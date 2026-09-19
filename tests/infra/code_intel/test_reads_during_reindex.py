"""Readers keep the whole previous index while a reindex runs in another process.

The premise behind serving code tools during a reindex: the indexer writes in one
transaction, and the databases run in WAL mode, so until it commits a reader in
another process -- the MCP daemon, while autosync's `lc code index` subprocess
holds the lock -- sees the last committed index, not a torn one. These tests pause
a real indexer mid-write (its deletes executed, its inserts not yet) and look.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.freshness import IndexRebuilding, index_state, require_ready, reset_readiness_probes
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, FTS_DB, workspace_dir
from lemoncrow.pro.capabilities.code_context import CodeContextEngine

_WRITER = """
import sys, time
from pathlib import Path
from lemoncrow.pro.capabilities.code_context import CodeContextEngine

root, gate, force = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
original = CodeContextEngine._parallel_extract

def paused(self, *args, **kwargs):
    (gate / "mid-write").touch()
    deadline = time.monotonic() + 60
    while not (gate / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return original(self, *args, **kwargs)

CodeContextEngine._parallel_extract = paused
CodeContextEngine(root, autosync_enabled=False).index_repo(force=force)
"""


def _repo(tmp_path: Path) -> Path:
    root = (tmp_path / "repo").resolve()
    (root / "pkg").mkdir(parents=True)
    for i in range(4):
        (root / "pkg" / f"mod{i}.py").write_text(f"def alpha_{i}():\n    return {i}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def _start_writer(root: Path, gate: Path, *, force: bool) -> subprocess.Popen[bytes]:
    gate.mkdir()
    writer = subprocess.Popen([sys.executable, "-c", _WRITER, str(root), str(gate), "1" if force else "0"])
    deadline = time.monotonic() + 60
    while not (gate / "mid-write").exists():
        assert writer.poll() is None, "the writer exited before reaching its write"
        assert time.monotonic() < deadline, "the writer never reached its write"
        time.sleep(0.02)
    return writer


def _finish(writer: subprocess.Popen[bytes], gate: Path) -> None:
    (gate / "release").touch()
    assert writer.wait(timeout=120) == 0


def _committed(root: Path) -> tuple[set[str], int]:
    """Symbol names and alpha-bearing lines, as a separate reader connection sees them."""
    ws = workspace_dir(root)
    conn = sqlite3.connect(f"file:{ws / CODE_CONTEXT_DB}?mode=ro", uri=True)
    try:
        conn.execute("ATTACH DATABASE ? AS fts", (f"file:{ws / FTS_DB}?mode=ro",))
        names = {str(row[0]) for row in conn.execute("SELECT symbol_name FROM symbols")}
        lines = int(conn.execute("SELECT COUNT(*) FROM fts.file_line_fts WHERE text LIKE '%alpha_%'").fetchone()[0])
    finally:
        conn.close()
    return names, lines


@pytest.mark.parametrize("force", [True, False], ids=["full-rebuild", "incremental"])
def test_a_paused_reindex_leaves_readers_the_whole_previous_index(tmp_path: Path, force: bool) -> None:
    root = _repo(tmp_path)
    CodeContextEngine(root, autosync_enabled=False).index_repo(force=True)
    before_names, before_lines = _committed(root)
    assert {"alpha_0", "alpha_3"} <= before_names and before_lines == 4

    (root / "pkg" / "mod0.py").write_text("def omega_0():\n    return 0\n", encoding="utf-8")
    gate = tmp_path / "gate"
    writer = _start_writer(root, gate, force=force)
    try:
        reset_readiness_probes()
        state = index_state(root)
        assert state.refreshing, state
        require_ready(root)  # answers instead of raising IndexRebuilding
        assert _committed(root) == (before_names, before_lines), "a reader saw the reindex's uncommitted deletes"
    finally:
        _finish(writer, gate)

    after_names, after_lines = _committed(root)
    assert "omega_0" in after_names and "alpha_0" not in after_names
    assert after_lines == 3


def test_a_first_build_in_progress_still_refuses_to_answer(tmp_path: Path) -> None:
    """No committed index yet: an empty answer would read as "no such code", so raise."""
    root = _repo(tmp_path)
    gate = tmp_path / "gate"
    writer = _start_writer(root, gate, force=True)
    try:
        reset_readiness_probes()
        assert index_state(root).rebuilding
        with pytest.raises(IndexRebuilding):
            require_ready(root)
    finally:
        _finish(writer, gate)

    reset_readiness_probes()
    assert index_state(root).status == "ready"
