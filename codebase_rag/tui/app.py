"""Main TUI application for ragless-dev."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.scroll import Scroll

from .state import TUIState, Message


class RaglessApp:
    """Terminal TUI for ragless-dev with persistent Live display."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.running = False
        self._live: Live | None = None

    def _build_layout(self) -> Table:
        """Build the full status table shown in the Live display."""
        table = Table(title="ragless-dev TUI", box=None, padding=(0, 1))

        status = "[green]ready[/green]"
        if self.state.streaming:
            status = "[yellow]streaming...[/yellow]"

        msgs = self.state.messages
        last_msg = msgs[-1].content[:300] if msgs else ""

        table.add_column("field", style="dim", width=12)
        table.add_column("value")

        table.add_row("status", status)
        table.add_row("step", str(self.state.step))
        table.add_row("files", str(len(self.state.discovered_files)))
        table.add_row("sigs", str(len(self.state.extracted_signatures)))
        table.add_row("msgs", str(len(msgs)))
        if last_msg:
            table.add_row("last", last_msg)

        return table

    async def run_async(self):
        """Run the TUI with a persistent Live display."""
        self.running = True

        with Live(self._build_layout(), console=self.console, refresh_per_second=4) as live:
            self._live = live

            intro = (
                "[bold blue]ragless-dev[/bold blue] — file-based RAG with LangGraph\n"
                "[dim]Press Ctrl+C to exit. Type a query and press Enter.[/dim]"
            )
            self.console.print(intro)

            while self.running:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: Prompt.ask("[bold cyan]>[/bold cyan] ").strip())
                    if not user_input:
                        continue
                    await self._handle_input(user_input, live)
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

    async def _handle_input(self, user_input: str, live: Live):
        """Handle query, update state, refresh Live display."""
        self.state.messages.append(Message(role="user", content=user_input))
        self.state.streaming = True
        live.update(self._build_layout())

        try:
            from codebase_rag.dev.coordinator import DevCoordinator
            coordinator = DevCoordinator()

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: coordinator.handle_query(user_input)
            )

            self.state.messages.append(Message(role="agent", content=result))
            self.state.streaming = False

            # Update from coordinator state
            ctx = coordinator.get_context(user_input)
            self.state.discovered_files = ctx.get("discovered_files", [])
            self.state.extracted_signatures = ctx.get("extracted_signatures", [])
            self.state.step = ctx.get("step", 0)

        except Exception as e:
            self.state.streaming = False
            self.state.messages.append(Message(role="agent", content=f"[red]Error: {e}[/red]"))

        live.update(self._build_layout())