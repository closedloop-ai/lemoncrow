"""The MCP server's engine cache must not outlive the index it was built on (F11).

``_code_engine_cache`` was populate-once. A daemon built one ``CodeContextEngine``
per repo and reused it for its whole life: measured in production, one started at
``index_version 1`` was still serving at version 23 and returned empty results
for every query, with no error. These tests pin the two behaviours that fix it --
rebuild on a version bump, and raise rather than answer while the index is
mid-write -- plus the two that keep the fix from costing more than it saves:
no rebuild churn on a stable index, and one rebuild under concurrency.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.infra.code_intel.freshness import (
    FRESHNESS_REBUILT,
    IndexRebuilding,
    VersionedEngineCache,
)
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir

_DDL = (
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


class _FakeEngine:
    """Stands in for the closed ``CodeContextEngine``; identity is what we assert."""

    instances = 0

    def __init__(self, root: Path) -> None:
        type(self).instances += 1
        self.root = root
        self.db_path = str(root)

    def index_ready(self) -> bool:
        return True


@pytest.fixture
def indexed_repo(tmp_path: Path) -> Path:
    """A minimal but structurally faithful ``code_context.sqlite`` at version 1."""
    root = tmp_path / "repo"
    root.mkdir()
    ws = workspace_dir(root)
    ws.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ws / CODE_CONTEXT_DB)
    try:
        for statement in _DDL:
            conn.execute(statement)
        conn.execute(
            "INSERT INTO files VALUES ('r', 'a.py', 'python', 'h', 1, 0, '2026-01-01T00:00:00+00:00')",
        )
        conn.execute(
            "INSERT INTO symbols VALUES ('s0', 'r', 'a.py', 'python', 'alpha', 'alpha', 'function', "
            "'()', 0, 1, 1, 2, NULL, NULL, 'h')",
        )
        conn.execute("INSERT INTO engine_state VALUES ('index_version', '1')")
        conn.commit()
    finally:
        conn.close()
    return root


def _bump(root: Path, version: int) -> None:
    conn = sqlite3.connect(workspace_dir(root) / CODE_CONTEXT_DB)
    try:
        conn.execute("UPDATE engine_state SET value = ? WHERE key = 'index_version'", (str(version),))
        conn.commit()
    finally:
        conn.close()


def _tear(root: Path) -> None:
    """Reproduce the drop/rebuild window: files gone, symbols still present.

    This direction and not the other. Symbols with no files cannot be a resting
    state -- every symbol row references a file row -- whereas files with no
    symbols is exactly what a docs-only repo looks like.
    """
    conn = sqlite3.connect(workspace_dir(root) / CODE_CONTEXT_DB)
    try:
        conn.execute("DELETE FROM files")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a private cache and a fake engine so nothing leaks across tests."""
    import lemoncrow.pro.capabilities.code_context as code_context

    _FakeEngine.instances = 0
    monkeypatch.setattr(code_context, "CodeContextEngine", _FakeEngine)
    monkeypatch.setattr(mcp_server, "_code_engine_cache", VersionedEngineCache("test", recheck_seconds=0.0))
    monkeypatch.setattr(mcp_server, "_scoped_context_cache", {})
    mcp_server._code_index_freshness_for_current_call.value = None
    mcp_server._code_engine_for_current_call.value = None


def test_cache_is_version_stamped_not_a_plain_dict() -> None:
    """A bare dict is the defect; the type is load-bearing, so assert it."""
    from lemoncrow.gateway.adapters.mcp_server import _code_engine_cache

    assert isinstance(_code_engine_cache, VersionedEngineCache)


def test_stable_index_reuses_the_same_engine(indexed_repo: Path) -> None:
    first = mcp_server._code_context_engine(str(indexed_repo))
    second = mcp_server._code_context_engine(str(indexed_repo))
    assert first is second
    assert _FakeEngine.instances == 1


def test_index_version_bump_rebuilds_the_engine(indexed_repo: Path) -> None:
    first = mcp_server._code_context_engine(str(indexed_repo))
    _bump(indexed_repo, 23)
    second = mcp_server._code_context_engine(str(indexed_repo))

    assert second is not first, "engine survived a 22-generation index bump"
    assert _FakeEngine.instances == 2
    assert mcp_server._code_index_freshness_for_current_call.value == FRESHNESS_REBUILT


class _FakeScoped:
    """Stands in for the compiled ScopedContextCapability."""

    def __init__(self, engine: object) -> None:
        self.engine = engine


@pytest.fixture
def fake_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    import lemoncrow.pro.capabilities.scoped_context as scoped_context

    monkeypatch.setattr(scoped_context, "ScopedContextCapability", _FakeScoped)


def test_scoped_capability_is_rebuilt_when_its_engine_is(
    indexed_repo: Path,
    fake_scoped: None,
) -> None:
    """The capability captures its engine, so it goes stale in step with it."""
    first = mcp_server._scoped_context_capability(str(indexed_repo))
    _bump(indexed_repo, 23)
    second = mcp_server._scoped_context_capability(str(indexed_repo))

    assert second is not first
    assert second.engine is not first.engine


def test_scoped_capability_survives_a_stable_index(
    indexed_repo: Path,
    fake_scoped: None,
) -> None:
    first = mcp_server._scoped_context_capability(str(indexed_repo))
    assert mcp_server._scoped_context_capability(str(indexed_repo)) is first
    assert _FakeEngine.instances == 1


def test_scoped_capability_does_not_self_deadlock_on_a_rebuild(
    indexed_repo: Path,
    fake_scoped: None,
) -> None:
    """Regression: one thread, one non-reentrant lock, taken twice.

    `_scoped_context_capability` held `_scoped_context_cache_lock` while it
    built, and called `_code_context_engine` from inside that block. When the
    engine cache evicted on a version bump it reached for the same lock on the
    same thread -- a plain `threading.Lock`, no timeout -- and hung forever,
    taking every later caller with it.

    The original test for this exercised `_code_context_engine` directly and so
    never entered the deadlocking path at all. This one drives the real entry
    point, on a worker thread, and fails by timing out rather than by hanging
    the suite.
    """
    mcp_server._scoped_context_capability(str(indexed_repo))
    _bump(indexed_repo, 23)

    done = threading.Event()
    box: list[object] = []

    def call() -> None:
        box.append(mcp_server._scoped_context_capability(str(indexed_repo)))
        done.set()

    worker = threading.Thread(target=call, daemon=True)
    worker.start()

    assert done.wait(timeout=10), "_scoped_context_capability deadlocked on a rebuilt engine"
    assert box and box[0] is not None


def test_scoped_capability_never_holds_the_lock_across_the_engine_call(
    indexed_repo: Path,
    fake_scoped: None,
) -> None:
    """Pin the ordering rule, not just its absence of symptoms.

    Resolving the engine while holding the scoped-cache lock is what made the
    deadlock possible. Assert it directly so a future refactor that moves the
    call back inside the lock fails here instead of in production.
    """
    held: list[bool] = []
    original = mcp_server._code_context_engine

    def probe(repo_root: str = ".") -> object:
        held.append(mcp_server._scoped_context_cache_lock.locked())
        return original(repo_root)

    mcp_server._code_context_engine = probe  # type: ignore[assignment]
    try:
        mcp_server._scoped_context_capability(str(indexed_repo))
    finally:
        mcp_server._code_context_engine = original  # type: ignore[assignment]

    assert held == [False], "the engine was resolved while the scoped-cache lock was held"


def test_mid_rebuild_index_raises_instead_of_serving_empty(indexed_repo: Path) -> None:
    """The regression that matters most.

    An empty result set from a rebuilding index is indistinguishable from a
    true negative, so a reviewer reads "no matches" as "no callers" and files
    a wrong finding. Failing loud is the whole point.
    """
    mcp_server._code_context_engine(str(indexed_repo))
    _tear(indexed_repo)

    with pytest.raises(IndexRebuilding):
        mcp_server._code_context_engine(str(indexed_repo))


def test_finish_code_result_stamps_a_rebuild(indexed_repo: Path) -> None:
    mcp_server._code_context_engine(str(indexed_repo))
    _bump(indexed_repo, 5)
    mcp_server._code_context_engine(str(indexed_repo))

    result = mcp_server._finish_code_result({"symbols": []})

    assert result["index_state"] == FRESHNESS_REBUILT
    assert "rebuilt" in result["hint"]


def test_finish_code_result_is_silent_on_a_stable_index(indexed_repo: Path) -> None:
    mcp_server._code_context_engine(str(indexed_repo))
    mcp_server._code_context_engine(str(indexed_repo))

    result = mcp_server._finish_code_result({"symbols": []})

    assert "index_state" not in result


def test_freshness_does_not_leak_into_the_next_call(indexed_repo: Path) -> None:
    """One rebuild must stamp one response, not every response after it."""
    mcp_server._code_context_engine(str(indexed_repo))
    _bump(indexed_repo, 5)
    mcp_server._code_context_engine(str(indexed_repo))

    assert mcp_server._finish_code_result({})["index_state"] == FRESHNESS_REBUILT
    assert "index_state" not in mcp_server._finish_code_result({})


def test_runtime_cache_reset_clears_the_engine_cache(indexed_repo: Path) -> None:
    mcp_server._code_context_engine(str(indexed_repo))
    assert len(mcp_server._code_engine_cache) == 1

    mcp_server._code_engine_cache.clear()

    assert len(mcp_server._code_engine_cache) == 0
    mcp_server._code_context_engine(str(indexed_repo))
    assert _FakeEngine.instances == 2
