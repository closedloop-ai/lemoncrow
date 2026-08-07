"""Synthetic code-intel workspaces for the open ``infra.code_intel`` modules.

The real databases are built by the closed engine, so these tests construct them
directly using the engine's own DDL (copied verbatim from a live
``.lemoncrow/workspace/`` dump). If the engine ever changes its schema, these
fixtures drift from reality -- which is exactly what
``CodeIntelStore.snapshot()``'s ``index_version`` stamp exists to surface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ID = "testrepo00000001"

_CODE_CONTEXT_DDL = (
    """
    CREATE TABLE files (
        repo_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        language TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL DEFAULT 0,
        indexed_at TEXT NOT NULL,
        PRIMARY KEY (repo_id, file_path)
    )
    """,
    """
    CREATE TABLE symbols (
        symbol_id TEXT PRIMARY KEY,
        repo_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        language TEXT NOT NULL,
        symbol_name TEXT NOT NULL,
        qualified_name TEXT NOT NULL,
        kind TEXT NOT NULL,
        signature TEXT NOT NULL,
        start_byte INTEGER NOT NULL,
        end_byte INTEGER NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        parent_symbol TEXT,
        doc_summary TEXT,
        content_hash TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE imports (
        repo_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        raw_import TEXT NOT NULL,
        target_file TEXT,
        UNIQUE(repo_id, source_file, raw_import, target_file)
    )
    """,
    "CREATE TABLE engine_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

_INTEL_DDL = (
    """
    CREATE TABLE call_edges (
        repo_id TEXT NOT NULL,
        caller_symbol_name TEXT NOT NULL,
        caller_qualified_name TEXT NOT NULL,
        caller_file_path TEXT NOT NULL,
        caller_start_line INTEGER NOT NULL,
        caller_end_line INTEGER NOT NULL,
        callee_name TEXT NOT NULL,
        callee_short_name TEXT NOT NULL DEFAULT '',
        call_line INTEGER NOT NULL,
        call_column INTEGER NOT NULL,
        UNIQUE(repo_id, caller_qualified_name, caller_file_path, call_line, call_column, callee_name)
    )
    """,
    """
    CREATE TABLE "references" (
        repo_id TEXT NOT NULL,
        symbol_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        line INTEGER NOT NULL,
        column INTEGER NOT NULL,
        end_column INTEGER NOT NULL,
        enclosing_symbol_name TEXT,
        enclosing_qualified_name TEXT,
        snippet TEXT NOT NULL,
        UNIQUE(repo_id, symbol_name, file_path, line, column, enclosing_qualified_name)
    )
    """,
    """
    CREATE TABLE centrality_map (
        repo_id TEXT NOT NULL,
        name_key TEXT NOT NULL,
        score REAL NOT NULL,
        index_version INTEGER NOT NULL,
        PRIMARY KEY (repo_id, name_key)
    )
    """,
)

_FILE_DEFAULTS: dict[str, Any] = {
    "language": "python",
    "content_hash": "hash",
    "size_bytes": 100,
    "mtime_ns": 1_000,
    "indexed_at": "2026-01-01T00:00:00+00:00",
}

_SYMBOL_DEFAULTS: dict[str, Any] = {
    "language": "python",
    "kind": "function",
    "signature": "()",
    "start_byte": 0,
    "end_byte": 10,
    "start_line": 1,
    "end_line": 2,
    "parent_symbol": None,
    "doc_summary": None,
    "content_hash": "hash",
}

_CALL_EDGE_DEFAULTS: dict[str, Any] = {
    "caller_symbol_name": "caller",
    "caller_start_line": 1,
    "caller_end_line": 5,
    "callee_short_name": "",
    "call_line": 2,
    "call_column": 4,
}

_REFERENCE_DEFAULTS: dict[str, Any] = {
    "line": 1,
    "column": 0,
    "end_column": 5,
    "enclosing_symbol_name": None,
    "enclosing_qualified_name": None,
    "snippet": "",
}


def _merge(defaults: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(row)
    return merged


def _insert(conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    quoted = f'"{table}"' if table == "references" else table
    placeholders = ",".join("?" * len(columns))
    conn.executemany(
        f"INSERT INTO {quoted} ({','.join(columns)}) VALUES ({placeholders})",
        [tuple(row[column] for column in columns) for row in rows],
    )


WorkspaceFactory = Callable[..., Path]


@pytest.fixture
def make_workspace(tmp_path: Path) -> WorkspaceFactory:
    """Build a synthetic ``.lemoncrow/workspace/`` and return its repo root.

    Every collection takes partial dicts; the fixture fills in the columns a
    test does not care about.
    """

    def _make(
        files: Iterable[Mapping[str, Any]] = (),
        symbols: Iterable[Mapping[str, Any]] = (),
        imports: Iterable[Mapping[str, Any]] = (),
        call_edges: Iterable[Mapping[str, Any]] = (),
        references: Iterable[Mapping[str, Any]] = (),
        centrality: Iterable[Mapping[str, Any]] = (),
        index_version: int = 8,
        indexer_semantics_version: int = 2,
        repo_id: str = REPO_ID,
        with_intel: bool = True,
        name: str = "repo",
    ) -> Path:
        from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, INTEL_DB, workspace_dir

        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        ws = workspace_dir(root)
        ws.mkdir(parents=True, exist_ok=True)

        code = sqlite3.connect(ws / CODE_CONTEXT_DB)
        try:
            for statement in _CODE_CONTEXT_DDL:
                code.execute(statement)
            file_rows = [_merge(_FILE_DEFAULTS, {"repo_id": repo_id, **row}) for row in files]
            _insert(
                code,
                "files",
                (
                    "repo_id",
                    "file_path",
                    "language",
                    "content_hash",
                    "size_bytes",
                    "mtime_ns",
                    "indexed_at",
                ),
                file_rows,
            )
            symbol_rows = []
            for position, row in enumerate(symbols):
                merged = _merge(_SYMBOL_DEFAULTS, {"repo_id": repo_id, **row})
                merged.setdefault("symbol_id", f"sym-{position}")
                merged.setdefault("qualified_name", merged["symbol_name"])
                symbol_rows.append(merged)
            _insert(
                code,
                "symbols",
                (
                    "symbol_id",
                    "repo_id",
                    "file_path",
                    "language",
                    "symbol_name",
                    "qualified_name",
                    "kind",
                    "signature",
                    "start_byte",
                    "end_byte",
                    "start_line",
                    "end_line",
                    "parent_symbol",
                    "doc_summary",
                    "content_hash",
                ),
                symbol_rows,
            )
            import_rows = [
                _merge({"raw_import": "mod", "target_file": None}, {"repo_id": repo_id, **row}) for row in imports
            ]
            _insert(code, "imports", ("repo_id", "source_file", "raw_import", "target_file"), import_rows)
            code.executemany(
                "INSERT INTO engine_state (key, value) VALUES (?, ?)",
                [
                    ("index_version", str(index_version)),
                    ("indexer_semantics_version", str(indexer_semantics_version)),
                ],
            )
            code.commit()
        finally:
            code.close()

        if with_intel:
            intel = sqlite3.connect(ws / INTEL_DB)
            try:
                for statement in _INTEL_DDL:
                    intel.execute(statement)
                edge_rows = []
                for row in call_edges:
                    merged = _merge(_CALL_EDGE_DEFAULTS, {"repo_id": repo_id, **row})
                    merged.setdefault("caller_qualified_name", merged["caller_symbol_name"])
                    edge_rows.append(merged)
                _insert(
                    intel,
                    "call_edges",
                    (
                        "repo_id",
                        "caller_symbol_name",
                        "caller_qualified_name",
                        "caller_file_path",
                        "caller_start_line",
                        "caller_end_line",
                        "callee_name",
                        "callee_short_name",
                        "call_line",
                        "call_column",
                    ),
                    edge_rows,
                )
                _insert(
                    intel,
                    "references",
                    (
                        "repo_id",
                        "symbol_name",
                        "file_path",
                        "line",
                        "column",
                        "end_column",
                        "enclosing_symbol_name",
                        "enclosing_qualified_name",
                        "snippet",
                    ),
                    [_merge(_REFERENCE_DEFAULTS, {"repo_id": repo_id, **row}) for row in references],
                )
                _insert(
                    intel,
                    "centrality_map",
                    ("repo_id", "name_key", "score", "index_version"),
                    [
                        _merge({"score": 1.0, "index_version": index_version}, {"repo_id": repo_id, **row})
                        for row in centrality
                    ],
                )
                intel.commit()
            finally:
                intel.close()

        return root

    return _make
