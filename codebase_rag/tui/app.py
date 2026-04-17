"""Main TUI application for ragless-dev."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.box import ROUNDED

from .state import TUIState, Message


class RaglessApp:
    """Terminal TUI for ragless-dev with live-updating display."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.running = False
        self._live: Live | None = None

    def _render_all(self) -> Table:
        """Render everything as a single table for Live display."""
        status = "[green]● ready[/green]"
        if self.state.streaming:
            status = "[yellow]◐ processing...[/yellow]"

        msgs = self.state.messages
        last_msg = msgs[-1].content[:200] if msgs else ""

        table = Table(box=None, padding=(0, 1), show_header=False)
        table.add_column("field", style="dim", width=14)
        table.add_column("value", max_width=80)
        table.add_row("[bold blue]ragless-dev[/bold blue]", "file-based RAG with LangGraph + MiniMax")
        table.add_row("─" * 30, "─" * 30)
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

        intro = (
            "[bold blue]ragless-dev[/bold blue] — file-based RAG with LangGraph\n"
            "[dim]Type your query and press Enter. Ctrl+C to exit.[/dim]\n"
        )
        self.console.print(intro)

        with Live(self._render_all(), console=self.console, refresh_per_second=8, transient=False) as live:
            self._live = live

            while self.running:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: Prompt.ask("[bold cyan]›[/bold cyan] ").strip()
                    )
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
        """Handle query: add user message, run coordinator, show agent response."""
        self.state.messages.append(Message(role="user", content=user_input))
        self.state.streaming = True
        live.update(self._render_all())

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

            live.update(self._render_all())

            # Show agent response in a panel after update
            self.console.print()
            self.console.print(
                Panel(
                    result[:800] + ("..." if len(result) > 800 else ""),
                    title="[bold blue]Agent[/bold blue] response",
                    border_style="blue",
                    box=ROUNDED,
                    padding=(0, 1),
                    width=self.console.width,
                )
            )

        except Exception as e:
            self.state.streaming = False
            self.state.messages.append(Message(role="agent", content=f"[red]Error: {e}[/red]"))
            live.update(self._render_all())
            self.console.print()
            self.console.print(
                Panel(
                    str(e),
                    title="[red]Error[/red]",
                    border_style="red",
                    box=ROUNDED,
                    width=self.console.width,
                )
            )