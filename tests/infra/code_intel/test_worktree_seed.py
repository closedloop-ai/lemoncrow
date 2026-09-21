"""A linked worktree's code index is seeded from its main checkout's (ISS-10812, PLN-2070 PR 1).

Each test builds a real git repository with a real linked worktree, indexes the
main checkout with the real engine, and opens the worktree through the daemon's
engine factory -- the path every code tool takes.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.infra.code_intel import worktree_seed
from lemoncrow.infra.code_intel.freshness import IndexRebuilding, VersionedEngineCache
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir
from lemoncrow.infra.code_intel.zoekt import adapter as zoekt_adapter
from lemoncrow.infra.code_intel.zoekt.server import ZoektServer, _read_git_head
from lemoncrow.pro.capabilities.code_context import CodeContextEngine

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

_GIT = ["git", "-c", "user.email=seed@test", "-c", "user.name=seed", "-c", "commit.gpgsign=false"]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> Iterator[None]:
    monkeypatch.setenv("LEMONCROW_CODE_AUTOSYNC", "0")
    monkeypatch.setenv("LEMONCROW_CODE_FILE_WATCHER", "0")
    monkeypatch.delenv(worktree_seed.WORKTREE_ENGINE_IDLE_ENV, raising=False)
    monkeypatch.setattr(worktree_seed, "_worktrees", {})
    monkeypatch.setattr(worktree_seed, "_checked_at", {})
    monkeypatch.setattr(zoekt_adapter, "_ROOT_OVERRIDES", {})
    monkeypatch.setattr(zoekt_adapter, "_SUPERVISORS", {})
    cache = VersionedEngineCache(
        "test",
        recheck_seconds=0.0,
        clock=clock,
        on_evict=mcp_server._retire_code_engine,
        retire=worktree_seed.retire_worktree_engine,
    )
    monkeypatch.setattr(mcp_server, "_code_engine_cache", cache)
    monkeypatch.setattr(mcp_server, "_scoped_context_cache", {})
    yield
    cache.stop_sweeper()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run([*_GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path]:
    """(main, worktree): main indexed at the current format, worktree freshly added."""
    main = (tmp_path / "main").resolve()
    (main / "pkg").mkdir(parents=True)
    for i in range(6):
        (main / "pkg" / f"mod{i}.py").write_text(f"def alpha_{i}():\n    return {i}\n", encoding="utf-8")
    (main / ".gitignore").write_text(".lemoncrow/\n", encoding="utf-8")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    worktree = (tmp_path / "wt").resolve()
    _git(main, "worktree", "add", "-q", "-b", "feature", str(worktree))
    CodeContextEngine(main, autosync_enabled=False).index_repo(force=True)
    return main, worktree


def _db(root: Path) -> Path:
    return workspace_dir(root) / CODE_CONTEXT_DB


def _state(root: Path) -> dict[str, str]:
    conn = sqlite3.connect(_db(root))
    try:
        return {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM engine_state")}
    finally:
        conn.close()


def _file_count(root: Path, repo_id: str | None = None) -> int:
    conn = sqlite3.connect(_db(root))
    try:
        if repo_id is None:
            return int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        return int(conn.execute("SELECT COUNT(*) FROM files WHERE repo_id = ?", (repo_id,)).fetchone()[0])
    finally:
        conn.close()


def _names(engine: Any, query: str) -> set[str]:
    return {s.symbol_name for s in engine.search_symbols(query, limit=50, auto_index=False)}


def _open(root: Path) -> Any:
    return mcp_server._code_context_engine(str(root))


@pytest.fixture
def extractions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Relative paths the engine re-extracts, in order."""
    seen: list[str] = []
    original = CodeContextEngine._parallel_extract

    def spy(self: Any, paths: list[Path], *args: Any, **kwargs: Any) -> Any:
        seen.extend(os.path.relpath(p, self.repo_root) for p in paths)
        return original(self, paths, *args, **kwargs)

    monkeypatch.setattr(CodeContextEngine, "_parallel_extract", spy)
    return seen


def test_opening_a_worktree_engine_seeds_it_by_clone_under_mains_lock(
    repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, extractions: list[str]
) -> None:
    main, worktree = repos
    main_id = worktree_seed.path_repo_id(main)
    main_store = workspace_dir(main)
    lock = main_store / (CODE_CONTEXT_DB + ".indexlock")
    clones: list[tuple[str, bool]] = []
    real_clone = worktree_seed._clone_file

    def spy_clone(src: Path, dst: Path) -> None:
        clones.append((src.name, worktree_seed._lock_held(lock)))
        real_clone(src, dst)

    monkeypatch.setattr(worktree_seed, "_clone_file", spy_clone)
    # A reader pinned on an old snapshot keeps the next commit's frames out of the
    # database file, so the seed has to carry main's WAL to include it.
    reader = sqlite3.connect(main_store / CODE_CONTEXT_DB, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM files").fetchone()
    writer = sqlite3.connect(main_store / CODE_CONTEXT_DB)
    writer.execute("INSERT INTO engine_state(key, value) VALUES ('probe', 'after-the-reader')")
    writer.commit()
    writer.close()
    try:
        assert (main_store / (CODE_CONTEXT_DB + "-shm")).exists()
        engine = _open(worktree)
    finally:
        reader.execute("ROLLBACK")
        reader.close()

    assert extractions == [], "opening a worktree ran an index build"
    state = _state(worktree)
    assert state[f"repo_id_alias:{worktree_seed.path_repo_id(worktree)}"] == main_id
    assert state["seeded_from"].startswith(f"{main}@")
    assert state["probe"] == "after-the-reader", "the seed lost a commit still in main's WAL"
    assert engine.repo_id == main_id
    assert _file_count(worktree) == _file_count(main, main_id) == 6
    assert "alpha_3" in _names(engine, "alpha_3")
    assert clones and all(held for _name, held in clones), clones
    assert (CODE_CONTEXT_DB + "-wal") in {name for name, _held in clones}
    assert not [name for name, _held in clones if name.endswith("-shm")]


def test_the_first_refresh_reextracts_only_what_differs_from_main(
    repos: tuple[Path, Path], extractions: list[str]
) -> None:
    main, worktree = repos
    engine = _open(worktree)
    assert "omega_wt" not in _names(engine, "omega_wt")
    (worktree / "pkg" / "mod1.py").write_text("def beta_1():\n    return 1\n", encoding="utf-8")
    (worktree / "pkg" / "only_here.py").write_text("def omega_wt():\n    return 9\n", encoding="utf-8")

    engine.index_repo(force=False)

    assert sorted(extractions) == ["pkg/mod1.py", "pkg/only_here.py"]
    assert "omega_wt" in _names(engine, "omega_wt")
    assert "alpha_1" not in _names(engine, "alpha_1")
    conn = sqlite3.connect(_db(worktree))
    try:
        stored = conn.execute("SELECT mtime_ns FROM files WHERE file_path = 'pkg/mod0.py'").fetchone()[0]
    finally:
        conn.close()
    assert stored == (worktree / "pkg" / "mod0.py").stat().st_mtime_ns, "an identical file kept main's mtime"
    assert "omega_wt" not in _names(_open(main), "omega_wt"), "the worktree's refresh wrote main's index"


def test_a_partial_worktree_index_is_replaced_by_a_seed(repos: tuple[Path, Path]) -> None:
    _main, worktree = repos
    partial = CodeContextEngine(worktree, autosync_enabled=False)
    partial._reindex_files([str(worktree / "pkg" / "mod0.py")])
    assert _file_count(worktree) == 1 and "seeded_from" not in _state(worktree)

    engine = _open(worktree)

    assert "seeded_from" in _state(worktree)
    assert _file_count(worktree) == 6
    assert "alpha_5" in _names(engine, "alpha_5")


def test_a_stale_seeded_index_is_reseeded_and_never_rebuilt(repos: tuple[Path, Path], extractions: list[str]) -> None:
    main, worktree = repos
    _open(worktree)
    conn = sqlite3.connect(_db(worktree))
    conn.execute("UPDATE engine_state SET value = '2' WHERE key = 'indexer_semantics_version'")
    conn.execute("INSERT INTO engine_state(key, value) VALUES ('stale-marker', 'x')")
    conn.commit()
    conn.close()
    version_before = int(_state(worktree)["index_version"])

    stale = CodeContextEngine(worktree, autosync_enabled=False)
    stale.index_repo(force=False)  # the stale format forces a rebuild -- refused for a seed
    stale.index_repo(force=True)
    assert extractions == [], "a seeded index was rebuilt"
    assert int(_state(worktree)["index_version"]) == version_before

    mcp_server._code_engine_cache.clear()
    engine = _open(worktree)

    state = _state(worktree)
    assert "stale-marker" not in state, "the stale index was not replaced"
    assert state["indexer_semantics_version"] == _state(main)["indexer_semantics_version"]
    assert extractions == []
    assert "alpha_2" in _names(engine, "alpha_2")


@pytest.mark.skipif(fcntl is None, reason="flock is POSIX-only")
def test_mains_lock_held_reports_seeding_then_the_next_call_seeds(
    repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    main, worktree = repos
    monkeypatch.setattr(worktree_seed, "SEED_LOCK_WAIT_S", 0.1)
    fd = os.open(workspace_dir(main) / (CODE_CONTEXT_DB + ".indexlock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with pytest.raises(IndexRebuilding, match="seeding the worktree index"):
            _open(worktree)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    engine = _open(worktree)
    assert "seeded_from" in _state(worktree)
    assert "alpha_0" in _names(engine, "alpha_0")


def test_an_alias_in_a_shared_database_leaves_the_other_repo_alone(repos: tuple[Path, Path], tmp_path: Path) -> None:
    main, worktree = repos
    other = (tmp_path / "other").resolve()
    other.mkdir()
    (other / "lib.py").write_text("def gamma():\n    return 0\n", encoding="utf-8")
    _git(other, "init", "-q")
    shared = _db(main)
    other_id = worktree_seed.path_repo_id(other)
    CodeContextEngine(other, db_path=shared, autosync_enabled=False).index_repo(force=False)
    other_rows = _file_count(main, other_id)
    assert other_rows == 1

    engine = _open(worktree)

    assert engine.repo_id == worktree_seed.path_repo_id(main)
    assert CodeContextEngine(other, db_path=shared, autosync_enabled=False).repo_id == other_id
    assert CodeContextEngine(other, db_path=_db(worktree), autosync_enabled=False).repo_id == other_id
    assert _file_count(main, other_id) == other_rows
    assert _file_count(worktree, other_id) == other_rows


def test_a_removed_worktrees_engine_retires_within_one_tick(repos: tuple[Path, Path]) -> None:
    main, worktree = repos
    main_engine = _open(main)
    engine = _open(worktree)
    shutil.rmtree(worktree)

    engine._autosync_tick(0)
    assert engine._autosync_stop.is_set(), "the engine's own tick kept running on a removed worktree"
    assert not (worktree / ".lemoncrow").exists(), "the tick recreated the removed worktree's store"

    retired = mcp_server._code_engine_cache.sweep()
    assert retired == [str(worktree)]
    assert str(worktree) not in mcp_server._code_engine_cache
    assert str(main) in mcp_server._code_engine_cache
    assert not main_engine._autosync_stop.is_set()
    assert zoekt_adapter._ROOT_OVERRIDES == {}


def test_an_idle_worktree_engine_retires_and_main_never_does(
    repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    main, worktree = repos
    monkeypatch.setenv(worktree_seed.WORKTREE_ENGINE_IDLE_ENV, "600")
    _open(main)
    engine = _open(worktree)

    clock.now += 599
    assert mcp_server._code_engine_cache.sweep() == []
    clock.now += 2
    assert mcp_server._code_engine_cache.sweep() == [str(worktree)]
    assert engine._autosync_stop.is_set()
    clock.now += 10_000
    assert mcp_server._code_engine_cache.sweep() == []
    assert str(main) in mcp_server._code_engine_cache


def test_a_seeded_engine_takes_zoekt_from_main_and_reads_its_own_head(repos: tuple[Path, Path]) -> None:
    main, worktree = repos
    _open(worktree)

    supervisor = zoekt_adapter.get_zoekt_supervisor(worktree)
    assert supervisor.repo_root == main
    assert supervisor.checkout_root == worktree
    assert supervisor._served_path(worktree / "pkg") == main / "pkg"

    (worktree / "pkg" / "new.py").write_text("x = 1\n", encoding="utf-8")
    _git(worktree, "add", "-A")
    _git(worktree, "commit", "-q", "-m", "wt")
    head = _git(worktree, "rev-parse", "HEAD")
    assert head != _git(main, "rev-parse", "HEAD")
    assert _read_git_head(worktree) == head
    assert ZoektServer(worktree).current_git_head() == head


def test_the_main_checkout_engine_is_unchanged(repos: tuple[Path, Path]) -> None:
    main, _worktree = repos
    before = _state(main)

    engine = _open(main)

    assert engine.repo_id == worktree_seed.path_repo_id(main)
    assert _state(main) == before
    assert not [key for key in before if key.startswith("repo_id_alias:") or key == "seeded_from"]
    assert zoekt_adapter.get_zoekt_supervisor(main).checkout_root == main
