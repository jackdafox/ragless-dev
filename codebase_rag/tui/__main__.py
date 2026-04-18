"""CLI entry point for ragless-dev TUI."""

from __future__ import annotations

import argparse
import os

from .app import RaglessApp


def main() -> int:
    parser = argparse.ArgumentParser(description="ragless-dev TUI")
    parser.add_argument("--root", default=os.getcwd(), help="Root directory to scan")
    args = parser.parse_args()

    app = RaglessApp(root=args.root)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
