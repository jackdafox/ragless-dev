# ragless-dev — Project Conversation History

This file tracks the evolution of the ragless-dev project.

## 2026-05-03 — Session Summary

### Tech Stack
- **LangGraph** for the agent pipeline (StateGraph with 6 nodes)
- **MiniMax M2.7** via Anthropic SDK (`https://api.minimax.io/anthropic`)
- **Textual** for the TUI
- **Poetry** for dependency management

### Architecture

```
file_discover_node → extract_signatures_node → build_retrieval_context_node
                                                        ↓
                                                  agent_node (ReAct)
                                                        ↓
                                                  replan_node
                                                        ↓
                                              final_response_node
```

### Nodes (in order)
| Node | LLM Call | Purpose |
|------|----------|---------|
| `file_discover_node` | No | Glob-based file search |
| `extract_signatures_node` | No | Tree-sitter Python parsing |
| `build_retrieval_context_node` | No | Formats prompt string |
| `agent_node` | Yes | ReAct reasoning + 2 tools |
| `replan_node` | No | Loop control (max 3 iterations) |
| `final_response_node` | Yes | Natural language answer |

### Key Files
- `codebase_rag/dev/llm.py` — MiniMax LLM setup, cached instance
- `codebase_rag/dev/coordinator.py` — Graph runner, result cache (key = md5 of query + skip_final_response flag)
- `codebase_rag/dev/nodes.py` — All LangGraph nodes, `_get_agent()` cached at module level
- `codebase_rag/tui/app.py` — Textual TUI app, streams LLM chunks to RichLog
- `codebase_rag/dev/__main__.py` — CLI entry: `poetry run rag dev "query"`
- `codebase_rag/tui/__main__.py` — TUI entry: `poetry run python -m codebase_rag.tui --root /path`

### Caching
- LLM instance cached in `llm.py::get_llm()` (one per process)
- Agent cached in `nodes.py::_get_agent()`
- Graph results cached in `coordinator.py` by `md5(query + skip_final_response)`

### Known Issues / TODOs
- `create_react_agent` deprecated → move to `langchain.agents.create_agent`
- TUI streaming to stderr was removed — agent streams `[agent]` to stderr when `STREAM_OUTPUT=1`
- Cache key includes `skip_final_response` to prevent TUI cached results leaking to CLI

### Commands
```bash
# CLI
poetry run rag dev "query"

# TUI
poetry run python -m codebase_rag.tui --root /path/to/project

# Tests
poetry run pytest tests/ -v

# Timing debug
$env:DEBUG_TIMING = "1"; poetry run rag dev "query"
```

### CI/CD
- `.github/workflows/ci.yml` — pip-audit (vulnerability scan) + pytest
- gitleaks removed (requires paid license)