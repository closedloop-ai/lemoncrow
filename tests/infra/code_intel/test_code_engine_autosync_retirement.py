"""A daemon runs one autosync loop per repo, however often the index moves.

Measured on a live daemon: the symphony-alpha index-write lock was held for the
whole of a sampled window, and code_search answered "index is being rebuilt" for
most calls. Every index bump had left the replaced engine's autosync loop
running, and each leaked loop spawned its own reindex per change (reproduced: six
engines, one edited file, six `lc code index` processes). The fake-engine tests
pin the wiring; this one drives the real engine through the production cache, so
the live thread count is the evidence.
"""

from __future__ import annotations

import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest

from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.infra.code_intel.freshness import VersionedEngineCache
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir
from lemoncrow.pro.capabilities.code_context import CodeContextEngine


def _live_autosync_threads(name: str) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == name and t.is_alive()]


def _bump(root: Path) -> None:
    conn = sqlite3.connect(workspace_dir(root) / CODE_CONTEXT_DB)
    try:
        conn.execute("UPDATE engine_state SET value = CAST(value AS INTEGER) + 1 WHERE key = 'index_version'")
        conn.commit()
    finally:
        conn.close()


def test_index_bumps_leave_exactly_one_autosync_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    for i in range(3):
        (repo / "pkg" / f"mod{i}.py").write_text(f"def f{i}():\n    return {i}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    CodeContextEngine(repo, autosync_enabled=False).index_repo(force=True)

    # The suite disables autosync globally; this test is about autosync.
    monkeypatch.setenv("LEMONCROW_CODE_AUTOSYNC", "1")
    monkeypatch.setenv("LEMONCROW_CODE_FILE_WATCHER", "0")
    cache = VersionedEngineCache("test", recheck_seconds=0.0, on_evict=mcp_server._code_engine_cache.on_evict)

    engines: list[CodeContextEngine] = []
    try:
        for _ in range(4):
            engine, _freshness = cache.get(str(repo), repo, lambda: CodeContextEngine(repo))
            engines.append(engine)
            _bump(repo)
        assert len({id(e) for e in engines}) == 4, "the index bumps did not rebuild the engine"

        name = f"lemoncrow-code-autosync-{engines[0].repo_id[:8]}"
        deadline = time.monotonic() + 10
        while len(_live_autosync_threads(name)) > 1 and time.monotonic() < deadline:
            time.sleep(0.05)

        assert len(_live_autosync_threads(name)) == 1, "replaced engines kept their autosync loops"
    finally:
        for engine in engines:
            engine.stop_autosync()
