"""F0: read-only accessors over the closed engine's code-intel databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.store import (
    CODE_CONTEXT_DB,
    INTEL_DB,
    CodeIntelStore,
    CodeIntelUnavailable,
    open_ro,
    workspace_dir,
)

# `make_workspace` comes from conftest.py, which pytest imports as a plugin --
# tests/infra/code_intel/ is not a package, so it cannot be imported by path.
WorkspaceFactory = Callable[..., Path]


def _populated(make_workspace: WorkspaceFactory) -> Path:
    return make_workspace(
        files=[
            {"file_path": "src/a.py", "content_hash": "aaa", "size_bytes": 10, "mtime_ns": 111},
            {"file_path": "src/b.py", "content_hash": "bbb", "size_bytes": 20, "mtime_ns": 222},
            {"file_path": "README.md", "language": "markdown"},
        ],
        symbols=[
            {"file_path": "src/a.py", "symbol_name": "alpha", "kind": "function", "start_line": 10},
            {"file_path": "src/a.py", "symbol_name": "Beta", "kind": "class", "start_line": 1},
            {"file_path": "src/b.py", "symbol_name": "gamma", "kind": "function", "start_line": 3},
        ],
        imports=[
            {"source_file": "src/b.py", "raw_import": "a", "target_file": "src/a.py"},
            {"source_file": "src/a.py", "raw_import": "os", "target_file": None},
        ],
        call_edges=[
            {"caller_file_path": "src/b.py", "caller_symbol_name": "gamma", "callee_name": "mod.alpha"},
            {
                "caller_file_path": "src/a.py",
                "caller_symbol_name": "alpha",
                "callee_name": "pkg.helper",
                "callee_short_name": "helper",
            },
        ],
        references=[
            {"symbol_name": "alpha", "file_path": "src/b.py", "line": 4},
            {"symbol_name": "alpha", "file_path": "src/a.py", "line": 10},
        ],
        centrality=[{"name_key": "alpha", "score": 9.5}, {"name_key": "gamma", "score": 1.5}],
        index_version=8,
    )


def test_workspace_dir_is_repo_local(tmp_path: Path) -> None:
    assert workspace_dir(tmp_path) == tmp_path.resolve() / ".lemoncrow" / "workspace"


def test_open_ro_reports_an_unindexed_workspace(tmp_path: Path) -> None:
    with pytest.raises(CodeIntelUnavailable, match="indexed"):
        open_ro(CODE_CONTEXT_DB, tmp_path)


def test_open_ro_connection_cannot_write(make_workspace: WorkspaceFactory) -> None:
    """The engine owns these files; read-only is structural, not a convention."""
    root = _populated(make_workspace)
    conn = open_ro(CODE_CONTEXT_DB, root)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM files")
    finally:
        conn.close()


def test_repo_id_is_discovered_from_the_index(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=[{"file_path": "src/a.py"}], repo_id="deadbeef")
    with CodeIntelStore(root) as store:
        assert store.repo_id == "deadbeef"


def test_empty_index_raises_rather_than_returning_nothing(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace()
    with CodeIntelStore(root) as store, pytest.raises(CodeIntelUnavailable, match="empty index"):
        store.repo_id  # noqa: B018 -- property access is the call under test


def test_files_accessor(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        assert {row.file_path for row in store.files()} == {"src/a.py", "src/b.py", "README.md"}
        only = store.files(paths=["src/a.py"])
        assert len(only) == 1
        assert only[0].content_hash == "aaa"
        assert only[0].mtime_ns == 111
        assert store.files(paths=[]) == []


def test_symbols_accessor_filters_and_orders(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        in_a = store.symbols(file_path="src/a.py")
        assert [row.symbol_name for row in in_a] == ["Beta", "alpha"]  # ordered by start_line
        assert [row.symbol_name for row in store.symbols(kind="CLASS")] == ["Beta"]
        assert len(store.symbols(limit=2)) == 2


def test_symbol_counts_by_file_omits_files_with_no_symbols(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        counts = store.symbol_counts_by_file()
    assert counts == {"src/a.py": 2, "src/b.py": 1}
    assert "README.md" not in counts


def test_imports_accessor_distinguishes_resolved_from_dangling(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        assert len(store.imports()) == 2
        resolved = store.imports(resolved_only=True)
    assert [(row.source_file, row.target_file) for row in resolved] == [("src/b.py", "src/a.py")]


def test_call_edges_match_qualified_or_short_callee(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        assert [row.caller_symbol_name for row in store.call_edges(callee_name="mod.alpha")] == ["gamma"]
        # `helper` only appears as callee_short_name -- the OR arm must fire.
        assert [row.caller_symbol_name for row in store.call_edges(callee_name="helper")] == ["alpha"]
        assert [row.callee_name for row in store.call_edges(caller_file_path="src/a.py")] == ["pkg.helper"]


def test_references_and_centrality(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        assert {row.file_path for row in store.references(symbol_name="alpha")} == {"src/a.py", "src/b.py"}
        assert [row.name_key for row in store.centrality()] == ["alpha", "gamma"]  # score DESC
        assert [row.name_key for row in store.centrality(limit=1)] == ["alpha"]


def test_snapshot_stamps_the_engine_generation(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        snapshot = store.snapshot()
    assert snapshot.index_version == 8
    assert snapshot.indexer_semantics_version == 2
    assert snapshot.files == 3
    assert snapshot.symbols == 3
    assert snapshot.imports == 2
    assert snapshot.imports_resolved == 1
    assert snapshot.intel_available is True
    assert snapshot.call_edges == 2
    assert snapshot.references == 2
    assert snapshot.centrality == 2


def test_missing_intel_db_degrades_instead_of_raising(make_workspace: WorkspaceFactory) -> None:
    """The call graph is a later pass than the symbol index and can be absent."""
    root = make_workspace(files=[{"file_path": "src/a.py"}], with_intel=False)
    assert not (workspace_dir(root) / INTEL_DB).exists()
    with CodeIntelStore(root) as store:
        assert store.intel_available is False
        assert store.call_edges() == []
        assert store.references() == []
        assert store.centrality() == []
        snapshot = store.snapshot()
    assert snapshot.intel_available is False
    assert snapshot.call_edges == 0
    assert snapshot.files == 1


def test_engine_state_falls_back_when_key_absent(make_workspace: WorkspaceFactory) -> None:
    root = _populated(make_workspace)
    with CodeIntelStore(root) as store:
        assert store.engine_state("nope", default=-1) == -1
