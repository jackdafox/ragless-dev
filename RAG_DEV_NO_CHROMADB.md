# rag dev without ChromaDB

## Problem
Pre-indexing with ChromaDB is too slow. `rag dev` should work without it.

## How Claude Code Does It
No pre-indexing. Files read on demand at query time. LLM sees relevant files directly in context window.

## Proposed Approach

### `rag dev` Flow (no ChromaDB)

1. User query: `"add JWT refresh token support to auth service"`
2. System uses **glob + tree-sitter** to find relevant files:
   - Glob: `**/*auth*.py`, `**/*token*.py`, `**/*jwt*.py`
   - Or: ask LLM which files to read based on query
3. Tree-sitter extracts **signatures** (function name, params, return type, docstring) from found files
4. LLM gets: query + signatures + optional full file content for top candidates
5. Architect plans. Engineer generates. Validator checks.

### What Gets Dropped
- `chromadb` dependency
- `embeddings/factory.py`
- `db/collections.py`
- Persistent vector index

### What Gets Added/Modified
- `codebase_rag/dev/file_finder.py` — glob + tree-sitter file discovery
- `codebase_rag/dev/signature_extractor.py` — extract signatures only (not full bodies)
- `codebase_rag/dev/context_builder.py` — format signatures + file content for LLM
- Modify `agents/coordinator.py` — skip RAG retrieval, inject file-based context

### File Discovery Strategies

**Option A: Glob patterns from query keywords**
```
query: "add JWT refresh to auth"
keywords: ["auth", "token", "jwt", "refresh"]
glob patterns: ["**/*auth*.py", "**/*token*.py", "**/*jwt*.py"]
```

**Option B: LLM selects files**
```
System asks: "Which files in the codebase are relevant to your request?"
LLM reads directory tree, picks files, returns paths
```

**Option C: Explicit file paths (user provides)**
```
rag dev "add refresh token" --files src/auth.py src/token.py
```

### Signature Extraction

Tree-sitter already parses files. Extract only:
```python
{
    "name": "create_access_token",
    "params": ["user_id: int", "expires_delta: timedelta"],
    "return_type": "str",
    "docstring": "Create a JWT access token...",
    "file_path": "src/auth.py",
    "start_line": 42,
}
```

Full bodies stored separately, read on demand.

### Context Window Budget

LLM context: ~128K tokens (claude-haiku) or ~200K (claude-sonnet-4-6)
- Query: ~100 tokens
- Signatures (50 functions across 10 files): ~5,000 tokens
- Full file content (selected 2-3 files): ~20,000 tokens
- System prompt + plan + validation: ~10,000 tokens

Well within budget. No embeddings needed.

### Implementation Order

1. `file_finder.py` — glob-based discovery from keywords
2. `signature_extractor.py` — reuse existing chunker, extract signatures only
3. `context_builder.py` — format signatures + full content for LLM
4. Modify `coordinator.py` — replace RAG retrieval with file-based context
5. Remove ChromaDB from `rag dev` command

### Risks

- Large codebases: glob may match too many files
- Keyword matching is brittle (query "caching" → files not named "cache")
- Mitigation: LLM selects from candidate files before reading full content

### Experiment Plan

Create a separate directory with modified `codebase_rag/dev/` module. Keep original project intact.

```
codebase_rag/dev/
    __init__.py
    file_finder.py    # glob + keyword discovery
    signatures.py     # signature extraction
    context.py        # build LLM context
    coordinator.py    # modified coordinator (no ChromaDB)
```

Test on existing codebase. Compare `rag dev` quality with vs without ChromaDB.
