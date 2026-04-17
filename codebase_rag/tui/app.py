"""Main TUI application for ragless-dev."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.box import ROUNDED

from .state import TUIState, Message


class RaglessApp:
    """Terminal TUI for ragless-dev with user-friendly layout."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self.running = False

    def _render_header(self) -> Panel:
        lines = [
            "[bold blue]ragless-dev[/bold blue]  ·  file-based RAG with LangGraph",
            "",
            "  [dim]Type your query below. Ctrl+C to exit.[/dim]",
        ]
        return Panel(
            "\n".join(lines),
            title="[b]ragless-dev[/b] TUI",
            border_style="blue",
            box=ROUNDED,
            padding=(0, 1),
            height=5,
        )

    def _render_messages(self) -> Panel:
        """Render conversation history with user/agent distinction."""
        if not self.state.messages:
            lines = ["[dim]No messages yet — ask a question to get started.[/dim]"]
        else:
            lines = []
            for i, msg in enumerate(self.state.messages[-20:]):
                role_label = "[bold green]You[/bold green]" if msg.role == "user" else "[bold blue]Agent[/bold blue]"
                content = msg.content[:500] + ("..." if len(msg.content) > 500 else "")
                lines.append(f"{role_label}: {content}")
                lines.append("")

        return Panel(
            "\n".join(lines) or "[dim]No messages[/dim]",
            title=f"[b]Conversation[/b] ({len(self.state.messages)} messages)",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1),
            height=18,
        )

    def _render_context(self) -> Panel:
        """Render discovered files and signatures context."""
        if not self.state.discovered_files:
            lines = ["[dim]No files discovered yet.[/dim]", "[dim]Files will appear here after your first query.[/dim]"]
        else:
            lines = [
                f"[bold]{len(self.state.discovered_files)}[/bold] files, "
                f"[bold]{len(self.state.extracted_signatures)}[/bold] signatures",
                "",
            ]
            for fp in self.state.discovered_files[-6:]:
                short = fp.split("/")[-1] if "/" in fp else fp
                lines.append(f"  📄 [dim]{short}[/dim]")

            if self.state.extracted_signatures:
                lines.append("")
                lines.append("  [dim]Top signatures:[/dim]")
                for sig in self.state.extracted_signatures[:5]:
                    name = sig.get("name", sig.get("fname", "unknown")) if isinstance(sig, dict) else "unknown"
                    lines.append(f"    [dim]•[/dim] {name}()")

        return Panel(
            "\n".join(lines),
            title=f"[b]File Context[/b]",
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 1),
            height=12,
        )

    def _render_status(self) -> Table:
        """Render bottom status bar."""
        table = Table(box=ROUNDED, padding=(0, 3), show_header=False)
        table.add_column("label", style="dim", min_width=10)
        table.add_column("value", min_width=8)
        table.add_column("label", style="dim", min_width=10)
        table.add_column("value", min_width=8)
        table.add_column("label", style="dim", min_width=10)
        table.add_column("value", min_width=8)

        status = "[green]● ready[/green]"
        if self.state.streaming:
            status = "[yellow]◐ processing...[/yellow]"

        table.add_row("status", status, "step", str(self.state.step), "msgs", str(len(self.state.messages)))
        return table

    def _render_input_prompt(self) -> str:
        return "[bold cyan]›[/bold cyan] "

    def render(self):
        """Render the complete TUI layout to the console."""
        self.console.print(self._render_header())
        # Side-by-side: messages (wider) + context panel
        self.console.print(self._render_messages())
        self.console.print(self._render_context())
        self.console.print(self._render_status())

    async def run_async(self):
        """Run the TUI — full redraw between prompts."""
        self.running = True
        self.console.print("\n")
        self.render()
        self.console.print(
            "\n[dim]────────────────────────────────────────────────────[/dim]"
        )

        while self.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Prompt.ask(self._render_input_prompt()).strip()
                )
                if not user_input:
                    continue

                self.console.print(
                    "\n[dim]────────────────────────────────────────────────────[/dim]"
                )
                await self._handle_input(user_input)
                self.console.print(
                    "\n[dim]────────────────────────────────────────────────────[/dim]"
                )
                self.render()
                self.console.print(
                    "\n[dim]────────────────────────────────────────────────────[/dim]"
                )

            except KeyboardInterrupt:
                self.running = False
                self.console.print("\n\n[dim]Goodbye![/dim]")
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
        """Handle a query: update state, run coordinator, show response."""
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

            ctx = coordinator.get_context(user_input)
            self.state.discovered_files = ctx.get("discovered_files", [])
            self.state.extracted_signatures = ctx.get("extracted_signatures", [])
            self.state.step = ctx.get("step", 0)

            # Print agent response in a visible box
            self.console.print(
                Panel(
                    result[:1000] + ("..." if len(result) > 1000 else ""),
                    title="[blue]Agent[/blue] response",
                    border_style="blue",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )

        except Exception as e:
            self.state.streaming = False
            self.state.messages.append(Message(role="agent", content=f"[red]Error: {e}[/red]"))
            self.console.print(
                Panel(
                    f"[red]Error:[/red] {e}",
                    title="[red]Error[/red]",
                    border_style="red",
                    box=ROUNDED,
                )
            )