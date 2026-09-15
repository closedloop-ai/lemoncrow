"""No enumerative response claims completeness over data it never had.

PRD-739 FR1, FR3 and FR6. A consumer decides whether a list is whole by one
predicate -- ``objective == "exhaustive" and truncated is False`` -- and a
reviewer who sees it pass writes "nothing else calls this". So every
enumerative operation here gets its missing-data case built for real: a call
graph that was never built, an import table with no rows, an index that is
mid-rebuild or holds nothing. Each response must fail the predicate or be an
error. An empty answer that passes it is the failure being pinned.

Requests go through the registered MCP handlers, and once through the JSON-RPC
dispatcher, rather than the infra functions: a handler that caught the error
and answered with an empty result would pass the infra tests and fail these.

Organised by surface, so later contract checks extend this file.
"""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.infra.code_intel.completeness import OBJECTIVE_EXHAUSTIVE, OBJECTIVE_PARTIAL
from lemoncrow.infra.code_intel.freshness import IndexRebuilding, reset_readiness_probes
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, INTEL_DB, CodeIntelUnavailable, workspace_dir
from lemoncrow.pro.capabilities.code_context.call_graph import build_call_graph_payload, traverse_call_graph

_REPO_ID = "contract00000001"

_CODE_DDL = (
    "CREATE TABLE files (repo_id TEXT NOT NULL, file_path TEXT NOT NULL, language TEXT NOT NULL, "
    "content_hash TEXT NOT NULL, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL DEFAULT 0, "
    "indexed_at TEXT NOT NULL, PRIMARY KEY (repo_id, file_path))",
    "CREATE TABLE symbols (symbol_id TEXT PRIMARY KEY, repo_id TEXT NOT NULL, file_path TEXT NOT NULL, "
    "language TEXT NOT NULL, symbol_name TEXT NOT NULL, qualified_name TEXT NOT NULL, kind TEXT NOT NULL, "
    "signature TEXT NOT NULL, start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL, "
    "start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, parent_symbol TEXT, doc_summary TEXT, "
    "content_hash TEXT NOT NULL)",
    "CREATE TABLE imports (repo_id TEXT NOT NULL, source_file TEXT NOT NULL, raw_import TEXT NOT NULL, "
    "target_file TEXT)",
    "CREATE TABLE engine_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

_INTEL_DDL = (
    "CREATE TABLE call_edges (repo_id TEXT NOT NULL, caller_symbol_name TEXT NOT NULL, "
    "caller_qualified_name TEXT NOT NULL, caller_file_path TEXT NOT NULL, caller_start_line INTEGER NOT NULL, "
    "caller_end_line INTEGER NOT NULL, callee_name TEXT NOT NULL, callee_short_name TEXT NOT NULL DEFAULT '', "
    "call_line INTEGER NOT NULL, call_column INTEGER NOT NULL)",
    'CREATE TABLE "references" (repo_id TEXT NOT NULL, symbol_name TEXT NOT NULL, file_path TEXT NOT NULL, '
    "line INTEGER NOT NULL, column INTEGER NOT NULL, end_column INTEGER NOT NULL, enclosing_symbol_name TEXT, "
    "enclosing_qualified_name TEXT, snippet TEXT NOT NULL)",
    "CREATE TABLE centrality_map (repo_id TEXT NOT NULL, name_key TEXT NOT NULL, score REAL NOT NULL, "
    "index_version INTEGER NOT NULL)",
)

#: ``beta`` calls ``alpha`` on line 6.
_SOURCE = "def alpha():\n    return 1\n\n\ndef beta():\n    return alpha()\n"

_FILE_GRAPH_KINDS = ("blast_radius", "dead_code", "cycles", "coupling", "topology")

Invoke = Callable[[Path], dict[str, Any]]


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    """No throttled readiness probe or per-call engine survives from another test."""
    reset_readiness_probes()
    mcp_server._code_engine_for_current_call.value = None
    yield
    reset_readiness_probes()


def _claims_complete(payload: dict[str, Any]) -> bool:
    """The predicate a consumer evaluates -- the one these responses must not pass."""
    return payload.get("objective") == OBJECTIVE_EXHAUSTIVE and payload.get("truncated") is False


def _tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """A registered MCP tool, called through its argument-validating handler."""
    result = mcp_server.TOOLS[name]["handler"](arguments)
    assert isinstance(result, dict)
    return result


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )


def _repo_with_an_edit(tmp_path: Path) -> Path:
    """A git repository whose working tree changes ``alpha``'s body."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text(_SOURCE, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    (root / "a.py").write_text(_SOURCE.replace("return 1", "return 2"), encoding="utf-8")
    return root


def _write_index(
    root: Path,
    *,
    rows: bool = True,
    with_intel: bool = True,
    imports: bool = False,
    call_edges: bool = False,
) -> Path:
    """Lay a synthetic engine index over *root*.

    The defaults are the missing-data case: files and symbols are indexed, but
    there is no import row and no call edge. ``rows=False`` writes the schema
    with nothing in it -- an index the engine emptied for a migration.
    """
    ws = workspace_dir(root)
    ws.mkdir(parents=True, exist_ok=True)
    code = sqlite3.connect(ws / CODE_CONTEXT_DB)
    try:
        for statement in _CODE_DDL:
            code.execute(statement)
        code.execute("INSERT INTO engine_state VALUES ('index_version', '3')")
        if rows:
            code.executemany(
                "INSERT INTO files VALUES (?, ?, 'python', 'h', 1, 0, '2026-01-01T00:00:00+00:00')",
                [(_REPO_ID, "a.py"), (_REPO_ID, "b.py")],
            )
            code.executemany(
                "INSERT INTO symbols VALUES (?, ?, 'a.py', 'python', ?, ?, 'function', '()', 0, 1, ?, ?, NULL, NULL, 'h')",
                [("s-alpha", _REPO_ID, "alpha", "alpha", 1, 2), ("s-beta", _REPO_ID, "beta", "beta", 5, 6)],
            )
        if imports:
            code.execute("INSERT INTO imports VALUES (?, 'b.py', 'a', 'a.py')", (_REPO_ID,))
        code.commit()
    finally:
        code.close()
    if with_intel:
        intel = sqlite3.connect(ws / INTEL_DB)
        try:
            for statement in _INTEL_DDL:
                intel.execute(statement)
            if call_edges:
                intel.execute(
                    "INSERT INTO call_edges VALUES (?, 'beta', 'beta', 'a.py', 5, 6, 'alpha', 'alpha', 6, 11)",
                    (_REPO_ID,),
                )
            intel.commit()
        finally:
            intel.close()
    return root


def _tear(root: Path) -> None:
    """A reindex caught mid-write: symbol rows survive, file rows are gone."""
    conn = sqlite3.connect(workspace_dir(root) / CODE_CONTEXT_DB)
    try:
        conn.execute("DELETE FROM files")
        conn.commit()
    finally:
        conn.close()


def _graph(kind: str) -> Invoke:
    # `graph` takes no repo_root, so the file-graph kinds are driven through the
    # op it dispatches to.
    return lambda root: mcp_server._op_graph(kind=kind, path="a.py", repo_root=str(root))


# --------------------------------------------------------------------------- #
# relations: callers / callees / usages
# --------------------------------------------------------------------------- #

_TARGET: dict[str, Any] = {
    "symbol_id": "s-alpha",
    "symbol_name": "alpha",
    "qualified_name": "alpha",
    "file_path": "a.py",
    "kind": "function",
    "start_line": 1,
    "end_line": 2,
    "language": "python",
}


def _relations_payload(op: str, data_status: str) -> dict[str, Any]:
    """An engine-shaped relations payload whose edge data is in *data_status*.

    Callers and callees come from the engine's own traversal: a neighbour lookup
    returning ``None`` is how a missing call-edge store reaches it, and ``[]`` is
    "looked, found none". The engine's ``usages`` has no unavailable state of its
    own -- with no reference rows it answers from a text search -- so its payload
    carries the status directly, pinning the stamp for every relations op.
    """
    if op == "usages":
        return {
            "target": dict(_TARGET),
            "references": [],
            "reference_count": 0,
            "truncated": False,
            "data_status": data_status,
        }
    neighbours: list[Any] | None = None if data_status == "unavailable" else []
    traversal = traverse_call_graph(
        dict(_TARGET),
        direction=cast(Any, op),
        depth=1,
        limit=20,
        lookup_neighbors=lambda _symbol_id: neighbours,
    )
    assert traversal.data_status == data_status
    return build_call_graph_payload(dict(_TARGET), direction=cast(Any, op), depth=1, result=traversal)


@pytest.mark.parametrize("op", ["callers", "callees", "usages"])
def test_relations_with_unavailable_edge_data_is_partial_not_exhaustive(op: str) -> None:
    unavailable = mcp_server._maybe_attach_code_rendered(
        op, _relations_payload(op, "unavailable"), render_compact=False
    )
    empty = mcp_server._maybe_attach_code_rendered(op, _relations_payload(op, "empty"), render_compact=False)

    assert not _claims_complete(unavailable)
    assert unavailable["objective"] == OBJECTIVE_PARTIAL
    # Looked up and none found is an answer, and keeps the op's claim.
    assert empty["objective"] == OBJECTIVE_EXHAUSTIVE


# --------------------------------------------------------------------------- #
# code_query / code_changes / file-graph analytics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("with_intel", [False, True], ids=["intel-db-absent", "call-graph-never-built"])
@pytest.mark.parametrize("select", ["callers", "callees", "references"])
def test_code_query_intel_select_without_a_call_graph_never_claims_complete(
    tmp_path: Path, select: str, with_intel: bool
) -> None:
    root = _write_index(tmp_path / "repo", with_intel=with_intel)

    payload = _tool("code_query", {"select": select, "repo_root": str(root)})

    assert payload["rows"] == []
    assert payload["data_status"] == "unavailable"
    assert not _claims_complete(payload)


@pytest.mark.parametrize("with_intel", [False, True], ids=["intel-db-absent", "call-graph-never-built"])
def test_code_changes_without_a_call_graph_never_claims_complete(tmp_path: Path, with_intel: bool) -> None:
    root = _write_index(_repo_with_an_edit(tmp_path), with_intel=with_intel)

    payload = _tool("code_changes", {"repo_root": str(root)})

    assert [symbol["name"] for symbol in payload["changed_symbols"]] == ["alpha"]
    assert payload["changed_symbols"][0]["callers"] == 0
    assert payload["data_status"] == "unavailable"
    assert not _claims_complete(payload)


@pytest.mark.parametrize("kind", _FILE_GRAPH_KINDS)
def test_file_graph_without_import_rows_never_claims_complete(tmp_path: Path, kind: str) -> None:
    root = _write_index(tmp_path / "repo")

    payload = _graph(kind)(root)

    assert payload["data_status"] == "unavailable"
    assert not _claims_complete(payload)


_WITH_DATA: list[Any] = [
    pytest.param(lambda root: _tool("code_query", {"select": "callers", "repo_root": str(root)}), id="code_query"),
    pytest.param(lambda root: _tool("code_changes", {"repo_root": str(root)}), id="code_changes"),
    *(pytest.param(_graph(kind), id=f"graph-{kind}") for kind in _FILE_GRAPH_KINDS),
]


@pytest.mark.parametrize("invoke", _WITH_DATA)
def test_the_same_ops_still_claim_complete_once_the_data_exists(tmp_path: Path, invoke: Invoke) -> None:
    """Control: the cases above fail the predicate because data is missing, not always."""
    root = _write_index(_repo_with_an_edit(tmp_path), imports=True, call_edges=True)

    payload = invoke(root)

    assert payload["data_status"] == "available"
    assert _claims_complete(payload)


# --------------------------------------------------------------------------- #
# store-backed tools and index readiness
# --------------------------------------------------------------------------- #

_STORE_BACKED: list[Any] = [
    pytest.param(lambda root: _tool("code_changes", {"repo_root": str(root)}), id="code_changes"),
    pytest.param(lambda root: _tool("code_query", {"select": "callers", "repo_root": str(root)}), id="code_query"),
    pytest.param(
        lambda root: _tool("code_coverage_check", {"paths": ["a.py"], "repo_root": str(root)}),
        id="code_coverage_check",
    ),
    *(pytest.param(_graph(kind), id=f"graph-{kind}") for kind in _FILE_GRAPH_KINDS),
]


@pytest.mark.parametrize("invoke", _STORE_BACKED)
def test_store_backed_ops_raise_mid_rebuild_instead_of_answering(tmp_path: Path, invoke: Invoke) -> None:
    root = _write_index(_repo_with_an_edit(tmp_path), imports=True, call_edges=True)
    _tear(root)

    with pytest.raises(IndexRebuilding):
        invoke(root)


@pytest.mark.parametrize("emptied", [False, True], ids=["never-indexed", "index-emptied"])
@pytest.mark.parametrize("invoke", _STORE_BACKED)
def test_store_backed_ops_raise_on_an_absent_index_instead_of_answering(
    tmp_path: Path, invoke: Invoke, emptied: bool
) -> None:
    root = _repo_with_an_edit(tmp_path)
    if emptied:
        _write_index(root, rows=False)

    with pytest.raises(CodeIntelUnavailable):
        invoke(root)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("code_changes", {}),
        ("code_query", {"select": "callers"}),
        ("code_coverage_check", {"paths": ["a.py"]}),
    ],
)
def test_the_dispatcher_returns_an_unready_index_as_a_tool_error_not_an_empty_result(
    tmp_path: Path, name: str, arguments: dict[str, Any]
) -> None:
    root = _repo_with_an_edit(tmp_path)
    _write_index(root, rows=False)

    response = mcp_server._handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": {**arguments, "repo_root": str(root)}},
        }
    )

    assert isinstance(response, dict)
    result = response["result"]
    assert result["isError"] is True
    assert "being migrated" in result["content"][0]["text"]
