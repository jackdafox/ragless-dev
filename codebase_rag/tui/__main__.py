"""CLI entry point for ragless-dev TUI."""

from __future__ import annotations

from .app import RaglessApp


def main() -> int:
    app = RaglessApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
