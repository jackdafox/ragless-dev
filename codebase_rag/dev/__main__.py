"""CLI entry point: rag dev --query "..." or rag dev "..." --files ..."""

from __future__ import annotations

import argparse
import sys

from .coordinator import DevCoordinator
from .context_builder import format_llm_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="rag dev without ChromaDB")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("--files", "-f", nargs="*", help="Explicit file paths")
    parser.add_argument("--print-context", action="store_true", help="Print context and exit")

    args = parser.parse_args()

    if not args.query and not args.files:
        print("Error: provide a query or --files", file=sys.stderr)
        return 1

    query = args.query or " ".join(args.files)
    explicit = args.files if args.files else None

    coordinator = DevCoordinator()
    ctx = coordinator.get_context(query, explicit_files=explicit)

    if ctx is None:
        print(f"No files found for: {query}", file=sys.stderr)
        return 1

    if args.print_context:
        print(format_llm_prompt(ctx))
    else:
        print(f"Found {len(ctx.signatures)} signatures across {len(ctx.full_files)} files")
        for sig in ctx.signatures:
            print(f"  {sig.file_path}:{sig.start_line} {sig.name}({', '.join(sig.params)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
