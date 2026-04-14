"""Signature extraction from source files using tree-sitter."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class FunctionSignature:
    """Represents a function/method signature."""
    name: str
    params: list[str]
    return_type: str | None
    docstring: str | None
    file_path: str
    start_line: int
    end_line: int | None = None


def extract_docstring(node, source_bytes: bytes) -> str | None:
    """Extract docstring from an AST node."""
    try:
        doc = ""
        for child in node.children:
            if child.type == "string":
                raw = child.text.decode("utf-8")
                for prefix in ('"""', "'''", '"', "'"):
                    if raw.startswith(prefix) and raw.endswith(prefix):
                        raw = raw[len(prefix):-len(prefix)]
                        break
                doc = raw.strip()
                break
        return doc if doc else None
    except Exception:
        return None


def extract_signature(node, source_bytes: bytes) -> FunctionSignature | None:
    """Extract signature from a function definition node."""
    try:
        name_node = None
        params_node = None
        return_type_node = None

        for child in node.children:
            if child.type == "identifier":
                name_node = child
            elif child.type == "parameters":
                params_node = child
            elif child.type == "type_annotation":
                return_type_node = child

        if name_node is None:
            return None

        name = name_node.text.decode("utf-8")

        params = []
        if params_node:
            for child in params_node.children:
                if child.type == "identifier":
                    params.append(child.text.decode("utf-8"))
                elif child.type == "default_parameter":
                    for c in child.children:
                        if c.type == "identifier":
                            params.append(c.text.decode("utf-8"))
                            break

        return_type = None
        if return_type_node:
            return_type = return_type_node.text.decode("utf-8")

        docstring = extract_docstring(node, source_bytes)

        return FunctionSignature(
            name=name,
            params=params,
            return_type=return_type,
            docstring=docstring,
            file_path="",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )
    except Exception:
        return None


def parse_python_file(file_path: str) -> Iterator[FunctionSignature]:
    """Parse a Python file and yield function signatures."""
    try:
        import tree_sitter_python as tsp
    except ImportError:
        yield from _parse_python_regex(file_path)
        return

    with open(file_path, "rb") as f:
        source_bytes = f.read()

    try:
        parser = tsp.Parser()
        tree = parser.parse(source_bytes)
    except Exception:
        yield from _parse_python_regex(file_path)
        return

    def walk(node):
        if node.type in ("function_definition", "async_function_definition"):
            sig = extract_signature(node, source_bytes)
            if sig:
                sig.file_path = file_path
                yield sig
        elif node.type in ("class_definition", "module"):
            for child in node.children:
                walk(child)

    for child in tree.root_node.children:
        walk(child)


def _parse_python_regex(file_path: str) -> Iterator[FunctionSignature]:
    """Fallback regex-based parser when tree-sitter unavailable."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return

    func_pattern = re.compile(
        r"^(?:async\s+)?def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:",
        re.MULTILINE,
    )

    for match in func_pattern.finditer(source):
        name = match.group(1)
        params_raw = match.group(2)
        return_type = match.group(3)

        params = []
        for p in params_raw.split(","):
            p = p.strip()
            if p and "=" not in p:
                params.append(p.split(":")[0].strip())
            elif p:
                params.append(p.split(":")[0].strip())

        lines = source[:match.start()].count("\n") + 1

        yield FunctionSignature(
            name=name,
            params=params,
            return_type=return_type.strip() if return_type else None,
            docstring=None,
            file_path=file_path,
            start_line=lines,
        )


def extract_signatures(file_paths: list[str]) -> list[FunctionSignature]:
    """Extract signatures from all given files."""
    sigs = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            continue
        for sig in parse_python_file(fp):
            sig.file_path = fp
            sigs.append(sig)
    return sigs
