"""Main TUI application for ragless-dev."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from .state import TUIState, Message


class RaglessApp:
    """Terminal TUI for ragless-dev."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.running = False

    def _render_header(self) -> Panel:
        return Panel(
            "[bold blue]ragless-dev[/bold blue] — file-based RAG with LangGraph\n"
            "[dim]Powered by MiniMax + LangChain + LangGraph[/dim]",
            title="ragless-dev TUI",
            border_style="blue",
            height=3,
        )

    def _render_messages(self) -> Panel:
        if not self.state.messages:
            return Panel(
                "[dim]Ask a question to get started...[/dim]",
                title="History",
                border_style="dim",
                height=10,
            )

        lines = []
        for msg in self.state.messages[-10:]:
            role = "[bold green]You[/bold green]" if msg.role == "user" else "[bold blue]Agent[/bold blue]"
            snippet = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
            lines.append(f"{role}: {snippet}")
            lines.append("")

        return Panel(
            "\n".join(lines) or "[dim]No messages[/dim]",
            title=f"History ({len(self.state.messages)} messages)",
            border_style="green",
            height=15,
        )

    def _render_context(self) -> Panel:
        if not self.state.discovered_files:
            return Panel(
                "[dim]No files discovered yet[/dim]",
                title="File Context",
                border_style="yellow",
                height=6,
            )

        lines = [
            f"[bold]{len(self.state.discovered_files)} files[/bold], "
            f"[bold]{len(self.state.extracted_signatures)}[/bold] signatures"
        ]
        for fp in self.state.discovered_files[-5:]:
            lines.append(f"  [dim]{fp}[/dim]")

        return Panel(
            "\n".join(lines),
            title="File Context",
            border_style="yellow",
            height=8,
        )

    def _render_status(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("key", style="dim")
        table.add_column("value")

        status = "[green]ready[/green]"
        if self.state.streaming:
            status = "[yellow]streaming...[/yellow]"

        table.add_row("status", status)
        table.add_row("step", str(self.state.step))
        table.add_row("files", str(len(self.state.discovered_files)))
        table.add_row("sigs", str(len(self.state.extracted_signatures)))
        return table

    def render(self):
        """Render the full TUI layout."""
        self.console.print(self._render_header())
        self.console.print(self._render_messages())
        self.console.print(self._render_context())
        self.console.print(self._render_status())

    async def run_async(self):
        """Run the TUI with async input."""
        self.running = True
        self.console.print("[dim]Type your query and press Enter. Ctrl+C to exit.[/dim]\n")

        while self.running:
            try:
                self.render()
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Prompt.ask("[bold cyan]>[/bold cyan] ")
                )
                if user_input.strip():
                    await self._handle_input(user_input)
            except (KeyboardInterrupt, EOFError):
                self.running = False
                self.console.print("\n[dim]Goodbye![/dim]")
                break

    def run(self):
        """Run synchronously."""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass

    async def _handle_input(self, user_input: str):
        """Handle user query through the coordinator."""
        self.state.messages.append(Message(role="user", content=user_input))
        self.state.streaming = True
        self.render()

        try:
            from codebase_rag.dev.coordinator import DevCoordinator
            coordinator = DevCoordinator()

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: coordinator.handle_query(user_input)
            )

            self.state.messages.append(Message(role="agent", content=result))
            self.state.streaming = False
            self.render()

        except Exception as e:
            self.state.streaming = False
            self.state.messages.append(Message(role="agent", content=f"[red]Error: {e}[/red]"))
            self.render()
