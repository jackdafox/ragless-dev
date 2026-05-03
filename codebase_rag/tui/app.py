"""Ragless TUI — Textual app with Claude Code-inspired interface."""

from __future__ import annotations

import os
import sys

from textual.app import App, ComposeResult
from textual.widgets import Header, Static, RichLog, Input
from textual.containers import VerticalScroll, Container
from textual.binding import Binding
from textual.message import Message
from textual import on
from rich.markdown import Markdown

from .state import TUIState, Message as TuiMessage

# Whether to stream output (default: enabled)
_STREAM_OUTPUT = True

class RaglessApp(App):
    """Terminal TUI for ragless-dev — built with Textual."""

    CSS = """
    Screen {
        background: $surface;
    }
    #conversation {
        height: 1fr;
        border: none;
        padding: 0 1;
    }
    #context-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        padding: 0 1;
        border-top: solid $border;
    }
    #input-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl-c", "quit", "Quit", show=False),
        Binding("ctrl-q", "quit", "Quit", show=False),
    ]

    def __init__(self, root: str | None = None):
        super().__init__()
        self.state = TUIState()
        self.root = root or os.getcwd()

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="conversation", highlight=False, markup=True)
        yield Static("", id="context-bar")
        with Container(id="input-bar"):
            yield Input(placeholder="Ask about your codebase…", id="user-input")

    def on_mount(self):
        log = self.query_one("#conversation", RichLog)
        root_display = self.root if len(self.root) < 60 else "…" + self.root[-(60):]
        log.write(f"[bold blue]ragless-dev[/bold blue]  ·  [dim]{root_display}[/dim]")
        log.write("")
        log.write("[dim]Type a query and press Enter. Ctrl+C to quit.[/dim]")
        log.write("")
        self.query_one("#user-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted):
        user_input = event.value.strip()
        if not user_input:
            return

        log = self.query_one("#conversation", RichLog)
        input_widget = self.query_one("#user-input", Input)
        input_widget.clear()

        # Add user message
        self.state.messages.append(TuiMessage(role="user", content=user_input))
        log.write(f"[green]›[/green] {user_input}")
        self._update_context_bar()
        self._process_query(user_input, log)

    def _update_context_bar(self):
        bar = self.query_one("#context-bar", Static)
        parts = []

        if self.state.streaming:
            parts.append("[yellow]◐ processing…[/yellow]")
        else:
            parts.append("[green]●[/green]")

        parts.extend([
            f"[dim]step:{self.state.step}[/dim]",
            f"[dim]files:{len(self.state.discovered_files)}[/dim]",
            f"[dim]sigs:{len(self.state.extracted_signatures)}[/dim]",
            f"[dim]msgs:{len(self.state.messages)}[/dim]",
        ])
        if self.state.discovered_files:
            fps = ", ".join(fp.split("/")[-1] for fp in self.state.discovered_files[-4:])
            parts.append(f"[dim]📁 {fps}[/dim]")
        bar.update("  ".join(parts))

    def _on_step_update(self, step: int, discovered_files: list, extracted_signatures: list, agent_output: str = ""):
        """Called by the worker thread to report progress."""
        def _update():
            self.state.step = step
            self.state.discovered_files = discovered_files
            self.state.extracted_signatures = extracted_signatures
            self._update_context_bar()
            if agent_output:
                log = self.query_one("#conversation", RichLog)
                for line in agent_output.split("\n")[:5]:
                    log.write(f"  [dim]▌ {line}[/dim]")
        self.call_from_thread(_update)

    def _process_query(self, query: str, log: RichLog):
        self.state.streaming = True
        log.write("[yellow]◐[/yellow] processing…")
        self._update_context_bar()
        import threading
        t = threading.Thread(target=self._run_query_in_thread, args=(query, log), daemon=True)
        t.start()

    def _run_query_in_thread(self, query: str, log: RichLog):
        import traceback
        try:
            from codebase_rag.dev.coordinator import DevCoordinator

            coordinator = DevCoordinator(root=self.root)
            ctx = {}

            def report(step_info: str = ""):
                def _r():
                    self._update_context_bar()
                    if step_info:
                        log.write(f"  [dim]→ {step_info}[/dim]")
                self.call_from_thread(_r)

            # Stage 1: file discovery
            report("discovering files…")
            ctx = coordinator.get_context(query, skip_final_response=True)
            report("files found")

            # Stage 2: final LLM response — stream to log as it arrives
            from langchain_core.messages import HumanMessage
            from codebase_rag.dev.llm import get_llm

            retrieval_context = ctx.get("retrieval_context", "") or "(no relevant code found)"
            report("generating response…")

            answer_parts = []
            llm = get_llm()
            prompt = (
                "You are a helpful coding assistant. Based on the following "
                "codebase information, answer the user's query in a friendly, "
                "concise way. If no relevant code was found, say so.\n\n"
                f"Query: {query}\n\n"
                f"Codebase context:\n{retrieval_context}\n\n"
                "Answer:"
            )

            for chunk in llm.stream([HumanMessage(content=prompt)]):
                if hasattr(chunk, "content") and chunk.content:
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        answer_parts.append(text)

                        def write_chunk(t=text):
                            log.write(f"{t}")
                        self.call_from_thread(write_chunk)

            answer = "".join(answer_parts) or retrieval_context

            def update():
                self.state.messages.append(TuiMessage(role="agent", content=answer))
                self.state.streaming = False
                self.state.discovered_files = ctx.get("discovered_files", [])
                self.state.extracted_signatures = ctx.get("extracted_signatures", [])
                self.state.step = ctx.get("step", 0)
                log.write("")
                log.write(Markdown(answer))
                self._update_context_bar()

            self.call_from_thread(update)

        except Exception as e:
            def update_error():
                self.state.streaming = False
                tb = traceback.format_exc()
                log.write(f"\n  [red]✗ Error: {e}[/red]")
                log.write(f"  [dim]{tb}[/dim]")
                self._update_context_bar()
            self.call_from_thread(update_error)
