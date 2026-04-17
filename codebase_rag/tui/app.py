"""Main TUI application for ragless-dev."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.box import ROUNDED
from rich.layout import Layout

from .state import TUIState, Message


class RaglessApp:
    """Terminal TUI for ragless-dev — live display above the input line."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.running = False
        self._live: Live | None = None

    def _render_header(self) -> Panel:
        status = "[green]● ready[/green]"
        if self.state.streaming:
            status = "[yellow]◐ processing...[/yellow]"
        return Panel(
            "[bold blue]ragless-dev[/bold blue] — file-based RAG with LangGraph + MiniMax\n"
            f"status: {status}   |   step: {self.state.step}   |   "
            f"files: {len(self.state.discovered_files)}   |   "
            f"sigs: {len(self.state.extracted_signatures)}   |   "
            f"msgs: {len(self.state.messages)}",
            title="[b]ragless-dev[/b]",
            border_style="blue",
            box=ROUNDED,
            height=5,
            padding=(0, 1),
        )

    def _render_conversation(self) -> Panel:
        if not self.state.messages:
            lines = ["[dim]No messages yet — type a query to get started.[/dim]"]
        else:
            lines = []
            for msg in self.state.messages[-10:]:
                role = "[bold green]You[/bold green]" if msg.role == "user" else "[bold blue]Agent[/bold blue]"
                content = msg.content[:300] + ("..." if len(msg.content) > 300 else "")
                lines.append(f"{role}: {content}")
                lines.append("")
        return Panel(
            "\n".join(lines) or "[dim]No messages[/dim]",
            title=f"[b]Conversation[/b] ({len(self.state.messages)} msgs)",
            border_style="green",
            box=ROUNDED,
            height=14,
            padding=(0, 1),
        )

    def _render_context(self) -> Panel:
        if not self.state.discovered_files:
            lines = ["[dim]No files discovered yet.[/dim]"]
        else:
            lines = [
                f"[bold]{len(self.state.discovered_files)}[/bold] files, "
                f"[bold]{len(self.state.extracted_signatures)}[/bold] signatures",
            ]
            for fp in self.state.discovered_files[-6:]:
                short = fp.split("/")[-1]
                lines.append(f"  📄 [dim]{short}[/dim]")
            if self.state.extracted_signatures:
                lines.append("  [dim]Top signatures:[/dim]")
                for sig in self.state.extracted_signatures[:5]:
                    name = sig.get("name", "unknown") if isinstance(sig, dict) else "unknown"
                    lines.append(f"    [dim]•[/dim] {name}()")
        return Panel(
            "\n".join(lines),
            title="[b]File Context[/b]",
            border_style="yellow",
            box=ROUNDED,
            height=10,
            padding=(0, 1),
        )

    def _render_all(self) -> list:
        """Return panels as list for vertical stacking."""
        return [
            self._render_header(),
            self._render_conversation(),
            self._render_context(),
        ]

    def _render_status(self) -> Table:
        status = "[green]● ready[/green]"
        if self.state.streaming:
            status = "[yellow]◐ processing...[/yellow]"
        t = Table(box=ROUNDED, padding=(0, 2), show_header=False)
        t.add_column("k", style="dim")
        t.add_column("v")
        t.add_row("status", status)
        t.add_row("step", str(self.state.step))
        t.add_row("files", str(len(self.state.discovered_files)))
        t.add_row("sigs", str(len(self.state.extracted_signatures)))
        t.add_row("msgs", str(len(self.state.messages)))
        return t

    def _print_all(self):
        """Print all panels to console, called before each input."""
        self.console.print()
        for p in self._render_all():
            self.console.print(p)
        self.console.print(self._render_status())

    async def run_async(self):
        self.running = True
        self._print_all()

        while self.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Prompt.ask("[bold cyan]›[/bold cyan] ").strip()
                )
                if not user_input:
                    continue
                await self._handle_input(user_input)
                self._print_all()

            except KeyboardInterrupt:
                self.running = False
                self.console.print("\n[dim]Goodbye![/dim]")
                break
            except EOFError:
                self.running = False
                break

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass

    async def _handle_input(self, user_input: str):
        self.state.messages.append(Message(role="user", content=user_input))
        self.state.streaming = True
        self._print_all()

        try:
            from codebase_rag.dev.coordinator import DevCoordinator

            coordinator = DevCoordinator()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: coordinator.handle_query(user_input)
            )

            self.state.messages.append(Message(role="agent", content=result))
            self.state.streaming = False

            ctx = coordinator.get_context(user_input)
            self.state.discovered_files = ctx.get("discovered_files", [])
            self.state.extracted_signatures = ctx.get("extracted_signatures", [])
            self.state.step = ctx.get("step", 0)

        except Exception as e:
            self.state.streaming = False
            self.state.messages.append(Message(role="agent", content=f"[red]Error: {e}[/red]"))

        self._print_all()
        self.console.print(
            Panel(
                self.state.messages[-1].content[:600] + ("..." if len(self.state.messages[-1].content) > 600 else ""),
                title="[bold blue]Agent[/bold blue] response",
                border_style="blue",
                box=ROUNDED,
                width=self.console.width,
            )
        )