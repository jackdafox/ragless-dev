# ragless-dev

rag dev without ChromaDB — file-based RAG pipeline using LangGraph + LangChain.

On-demand file discovery and tree-sitter signature extraction. No pre-indexing, no vector DB.

## Stack

- **Python** 3.11+
- **LangGraph** — multi-agent replan loop
- **LangChain** — LLM chain orchestration
- **MiniMax 2.7** — model (via OpenAI-compatible API)
- **Poetry** — dependency management
- **pytest** — test suite

## Install

```bash
pip install poetry
poetry install --with dev
```

## Usage

```bash
# Query-based discovery
poetry run rag dev "add JWT refresh token to auth service"

# Explicit files
poetry run rag dev "query" --files src/auth.py src/token.py

# Print LLM context only
poetry run rag dev "query" --print-context
```

## TUI

Interactive terminal interface:

```bash
poetry run python -m codebase_rag.tui
```

Requires `MINIMAX_API_KEY` environment variable.

## Architecture

```
file_discover → extract_signatures → build_retrieval_context → agent → replan
    ↑                                                                      ↓
    ← ← ← ← ← ← ← (replan loop: max 3 iterations if needs_more_files) ← ← ←
    ↓
final_response_node
```

## Modules

| Module | Purpose |
|--------|---------|
| `file_finder` | Keyword-glob file discovery |
| `signature_extractor` | Tree-sitter or regex signature extraction |
| `context_builder` | Build LLM context from signatures + full files |
| `state` | LangGraph `TypedDict` state schema |
| `nodes` | 6 LangGraph nodes (discover, extract, build, agent, replan, final) |
| `graph` | `StateGraph` assembly + conditional edges |
| `tools` | ReAct agent tools (file discovery, get signatures) |
| `llm` | MiniMax LLM + ReAct agent factory |
| `tui` | Rich-based terminal interface (`RaglessApp`) |

## CI/CD

`.github/workflows/ci.yml` runs on every push:

- **Secret leak scan** — gitleaks
- **Dependency audit** — pip-audit
- **Tests** — pytest

## Test

```bash
poetry run pytest tests/ -v
```
