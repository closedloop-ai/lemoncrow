"""Code tools answer while a reindex holds the index lock, and say so.

On a large repo a reindex held the lock most of the time and every code tool
failed with "index is being rebuilt" for its whole length. A reindex commits in
one transaction, so the last committed index is whole and readable meanwhile;
the tools now serve it and append a note, so the model knows results may lag.
"""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.infra.code_intel.freshness import INDEX_LOCK_SUFFIX, reset_readiness_probes
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir
from lemoncrow.pro.capabilities.code_context import CodeContextEngine
from tests.helpers import init_store_at


@pytest.fixture()
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / ".lemoncrow"
    init_store_at(str(store))
    monkeypatch.setenv("LEMONCROW_ROOT", str(store))
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # setattr, not assignment: the calls below cache a ledger bound to this
    # test's store, and a leaked one reroutes later tests' model recommendations.
    monkeypatch.setattr(mcp_server._ledger, "_current_ledger", None)
    monkeypatch.setattr(mcp_server._ledger, "_realtime_ctx", None)
    remote = MagicMock()
    remote.get_context.return_value = {"context": "", "run_ledger": []}
    monkeypatch.setattr(mcp_server, "_remote_client", remote)
    mcp_server._RECENT_CODE_SEARCH_QUERIES.clear()
    (tmp_path / "billing.py").write_text("def reconcile_invoices():\n    return 1\n", encoding="utf-8")
    CodeContextEngine(tmp_path, autosync_enabled=False).index_repo(force=True)
    return tmp_path


@contextlib.contextmanager
def _reindex_in_progress(root: Path) -> Iterator[None]:
    lock_path = Path(str(workspace_dir(root) / CODE_CONTEXT_DB) + INDEX_LOCK_SUFFIX)
    lock_path.touch()
    with lock_path.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _call(name: str, args: dict[str, Any]) -> tuple[bool, str]:
    reset_readiness_probes()
    mcp_server._code_engine_cache.clear()
    resp = mcp_server._handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
    )
    assert isinstance(resp, dict) and "result" in resp, resp
    result = resp["result"]
    return bool(result.get("isError")), str(result["content"][0]["text"])


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("code_search", {"query": "reconcile_invoices"}),  # through the engine cache
        ("code_query", {"select": "symbols"}),  # through require_ready
    ],
)
def test_a_code_tool_answers_during_a_reindex_and_notes_it(indexed: Path, tool: str, args: dict[str, Any]) -> None:
    with _reindex_in_progress(indexed):
        is_error, text = _call(tool, args)

    assert not is_error, text
    assert "reconcile_invoices" in text
    assert mcp_server._INDEX_REFRESHING_NOTE in text

    is_error, text = _call(tool, args)
    assert not is_error, text
    assert mcp_server._INDEX_REFRESHING_NOTE not in text, "the refreshing note leaked into a later call"
