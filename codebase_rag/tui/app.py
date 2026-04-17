"""Main TUI application for ragless-dev — Claude Code-inspired chat interface."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .state import TUIState, Message


class RaglessApp:
    """Terminal TUI for ragless-dev — Claude Code-style chat."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.running = False

    def _clear_screen(self):
        self.console.print("\x1b[2J\x1b[H", end="")

    def _print_header(self):
        status = "●" if not self.state.streaming else "◐"
        sc = "green" if not self.state.streaming else "yellow"
        self.console.print(
            f"[bold blue]ragless-dev[/bold blue]  [dim]│[/dim]  "
            f"[{sc}]{status}[/{sc}]  "
            f"[dim]step:{self.state.step}  files:{len(self.state.discovered_files)}  "
            f"sigs:{len(self.state.extracted_signatures)}  msgs:{len(self.state.messages)}[/dim]"
        )
        self.console.print(Rule(style="dim"))
        self.console.print()

    def _print_history(self):
        for msg in self.state.messages[-20:]:
            if msg.role == "user":
                self.console.print(f"  [green]› {msg.content}[/green]")
                self.console.print()
            else:
                lines = msg.content.split("\n")
                for i, line in enumerate(lines[:10]):
                    prefix = "  " if i > 0 else "  "
                    self.console.print(f"{prefix}[blue]▌[/blue] {line}")
                if len(lines) > 10:
                    self.console.print(f"  [dim]▌ ... ({len(lines)-10} more lines)[/dim]")
                self.console.print()

    def _print_context_footer(self):
        if self.state.discovered_files:
            self.console.print(Rule(style="dim"))
            fps = ", ".join(fp.split("/")[-1] for fp in self.state.discovered_files[-4:])
            self.console.print(f"[dim]  📁 {fps}[/dim]")
            self.console.print()

    def _render(self):
        self._clear_screen()
        self._print_header()
        self._print_history()
        self._print_context_footer()

    def run(self):
        self.running = True
        self._render()

        while self.running:
            try:
                user_input = input("[bold cyan]›[/bold cyan] ").strip()
                if not user_input:
                    self._render()
                    continue

                self.state.messages.append(Message(role="user", content=user_input))
                self.state.streaming = True
                self._render()

                try:
                    from codebase_rag.dev.coordinator import DevCoordinator

                    coordinator = DevCoordinator()
                    result = coordinator.handle_query(user_input)

                    self.state.messages.append(Message(role="agent", content=result))
                    self.state.streaming = False

                    ctx = coordinator.get_context(user_input)
                    self.state.discovered_files = ctx.get("discovered_files", [])
                    self.state.extracted_signatures = ctx.get("extracted_signatures", [])
                    self.state.step = ctx.get("step", 0)

                    self._render()

                    # Print agent response after render
                    self.console.print()
                    lines = result.split("\n")
                    for i, line in enumerate(lines[:15]):
                        prefix = "  " if i > 0 else "  "
                        self.console.print(f"{prefix}[blue]▌[/blue] {line}")
                    if len(lines) > 15:
                        self.console.print(f"  [dim]▌ ... ({len(lines)-15} more lines)[/dim]")
                    self.console.print()

                except Exception as e:
                    self.state.streaming = False
                    self._render()
                    self.console.print(f"\n  [red]✗ {e}[/red]\n")

            except KeyboardInterrupt:
                self.running = False
                self.console.print("\n[dim]Goodbye![/dim]\n")
                break
            except EOFError:
                self.running = False
                break

    async def run_async(self):
        self.run()

    async def _handle_input(self, user_input: str):
        pass  # sync run() handles input directly