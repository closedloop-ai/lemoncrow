"""Index-generation probing and the version-stamped engine cache (F11).

The defect these cover is not "the cache is slightly stale" -- it is that a
cached engine built against a superseded index returns **zero results for every
query** and reports no error at all. So the assertions here care as much about
what is *raised* as about what is returned: an empty list from a rebuilding
index is the failure, not the fallback.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.freshness import (
    FRESHNESS_FRESH,
    FRESHNESS_REBUILT,
    INDEX_LOCK_SUFFIX,
    LOCK_HELD,
    STATUS_ABSENT,
    STATUS_READY,
    STATUS_REBUILDING,
    IndexRebuilding,
    VersionedEngineCache,
    index_state,
    take_refreshing,
)
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir

WorkspaceFactory = Callable[..., Path]

_FILES = [{"file_path": "a.py"}]
_SYMBOLS = [{"file_path": "a.py", "symbol_name": "alpha"}]


def _code_db(root: Path) -> Path:
    return workspace_dir(root) / CODE_CONTEXT_DB


def _set_index_version(root: Path, version: int) -> None:
    conn = sqlite3.connect(_code_db(root))
    try:
        conn.execute("UPDATE engine_state SET value = ? WHERE key = 'index_version'", (str(version),))
        conn.commit()
    finally:
        conn.close()


def _drop_table(root: Path, table: str) -> None:
    conn = sqlite3.connect(_code_db(root))
    try:
        conn.execute(f"DROP TABLE {table}")
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# index_state
# --------------------------------------------------------------------------- #


def test_never_indexed_workspace_is_absent_not_rebuilding(tmp_path: Path) -> None:
    """A workspace with no database has not failed -- it has not started."""
    state = index_state(tmp_path)
    assert state.status == STATUS_ABSENT
    assert state.rebuilding is False
    assert state.index_version == 0


def test_populated_index_is_ready_and_reports_its_version(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=23)
    state = index_state(root)
    assert state.status == STATUS_READY
    assert state.index_version == 23
    assert state.detail == ""


def test_empty_index_is_absent_not_ready(make_workspace: WorkspaceFactory) -> None:
    """Zero rows is a legitimate state; it must not masquerade as a rebuild."""
    root = make_workspace(index_version=4)
    state = index_state(root)
    assert state.status == STATUS_ABSENT
    assert state.index_version == 4


def test_symbols_without_files_reads_as_rebuilding(make_workspace: WorkspaceFactory) -> None:
    """A torn view: every symbol row references a file row, so this cannot rest."""
    root = make_workspace(symbols=_SYMBOLS, index_version=9)
    state = index_state(root)
    assert state.status == STATUS_REBUILDING
    assert "partially populated" in state.detail


def test_files_without_symbols_is_a_resting_state_not_a_rebuild(
    make_workspace: WorkspaceFactory,
) -> None:
    """A docs-only repo indexes files and extracts nothing. That is not a fault.

    The check used to be symmetric, which made this permanent: nothing about a
    symbol-less index ever changes, so every code tool would raise
    ``IndexRebuilding`` forever on a workspace that is merely empty of symbols.
    Same wrong answer as the silent-empty bug, delivered louder. A workspace
    without the optional tree-sitter ``parsers`` extra lands here too.
    """
    root = make_workspace(files=_FILES, index_version=9)
    state = index_state(root)
    assert state.status == STATUS_READY
    assert state.rebuilding is False

    cache = VersionedEngineCache("test", recheck_seconds=0.0)
    value, freshness = cache.get("k", root, object)
    assert value is not None, "a symbol-less index must still serve"
    assert freshness == FRESHNESS_FRESH


def test_missing_table_reads_as_rebuilding(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS)
    _drop_table(root, "imports")
    state = index_state(root)
    assert state.status == STATUS_REBUILDING
    assert "imports" in state.detail


@contextlib.contextmanager
def _index_lock_held(root: Path) -> Iterator[None]:
    fcntl = pytest.importorskip("fcntl")
    lock_path = Path(str(_code_db(root)) + INDEX_LOCK_SUFFIX)
    lock_path.touch()
    handle = lock_path.open("r+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def test_held_index_lock_on_a_populated_index_is_ready_and_refreshing(make_workspace: WorkspaceFactory) -> None:
    """A reindex holds the lock but commits in one transaction; readers keep the last index."""
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=3)
    take_refreshing()

    free = index_state(root)
    assert free.status == STATUS_READY and not free.refreshing
    assert take_refreshing() is False

    with _index_lock_held(root):
        state = index_state(root)

    assert state.status == STATUS_READY
    assert state.refreshing
    assert state.lock == LOCK_HELD
    assert state.index_version == 3
    assert take_refreshing() is True
    assert take_refreshing() is False, "the mark must clear once taken"


def test_held_index_lock_on_an_empty_index_is_a_first_build(make_workspace: WorkspaceFactory) -> None:
    """Nothing is committed yet, so there is nothing to answer from: that still fails loud."""
    root = make_workspace(index_version=0)

    assert index_state(root).status == STATUS_ABSENT
    with _index_lock_held(root):
        state = index_state(root)

    assert state.status == STATUS_REBUILDING
    assert state.detail == "first index build in progress"


def test_the_engine_cache_serves_a_refreshing_index_and_marks_every_probe(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=3)
    clock = _Clock()
    cache = VersionedEngineCache("test", recheck_seconds=5.0, clock=clock)
    first, _ = cache.get("k", root, object)
    take_refreshing()

    with _index_lock_held(root):
        clock.now = 10.0
        during, _ = cache.get("k", root, object)
        assert take_refreshing() is True
        clock.now = 11.0  # inside the throttle window: the cached probe is reused
        cache.get("k", root, object)
        assert take_refreshing() is True, "a throttled probe of a refreshing index went unmarked"

    assert during is first


# --------------------------------------------------------------------------- #
# VersionedEngineCache
# --------------------------------------------------------------------------- #


class _Clock:
    """Manually advanced monotonic clock, so the throttle is tested not waited on."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_unchanged_version_returns_the_same_object(make_workspace: WorkspaceFactory) -> None:
    """No rebuild churn: a stable index must not cost a rebuild per call."""
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    cache = VersionedEngineCache("test", recheck_seconds=0.0)
    builds = []

    def build() -> object:
        made = object()
        builds.append(made)
        return made

    first, first_state = cache.get("k", root, build)
    second, second_state = cache.get("k", root, build)

    assert first is second
    assert first_state == FRESHNESS_FRESH
    assert second_state == FRESHNESS_FRESH
    assert len(builds) == 1
    assert cache.evictions == 0


def test_version_bump_evicts_and_rebuilds(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    cache = VersionedEngineCache("test", recheck_seconds=0.0)

    first, _ = cache.get("k", root, object)
    assert cache.version_of("k") == 1

    _set_index_version(root, 23)
    second, freshness = cache.get("k", root, object)

    assert second is not first
    assert freshness == FRESHNESS_REBUILT
    assert cache.version_of("k") == 23
    assert cache.evictions == 1


def test_a_superseded_entry_is_retired_once(make_workspace: WorkspaceFactory) -> None:
    """Eviction has to end what the evicted value started, not just forget it."""
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    retired: list[object] = []
    cache = VersionedEngineCache("test", recheck_seconds=0.0, on_evict=retired.append)

    first, _ = cache.get("k", root, object)
    cache.get("k", root, object)
    assert retired == [], "a stable index retired its own live entry"

    _set_index_version(root, 2)
    second, _ = cache.get("k", root, object)

    assert retired == [first]
    assert second is not first


def test_discard_and_clear_retire_what_they_drop(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    retired: list[object] = []
    cache = VersionedEngineCache("test", recheck_seconds=0.0, on_evict=retired.append)

    discarded, _ = cache.get("a", root, object)
    cache.discard("a")
    cleared, _ = cache.get("b", root, object)
    cache.clear()

    assert retired == [discarded, cleared]


def test_a_failing_retirement_does_not_fail_the_lookup(make_workspace: WorkspaceFactory) -> None:
    """The rebuilt value is already cached; a stop hook that raises must not lose it."""
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)

    def refuse(_value: object) -> None:
        raise RuntimeError("stop failed")

    cache = VersionedEngineCache("test", recheck_seconds=0.0, on_evict=refuse)
    first, _ = cache.get("k", root, object)
    _set_index_version(root, 2)
    second, freshness = cache.get("k", root, object)

    assert second is not first
    assert freshness == FRESHNESS_REBUILT
    assert cache.peek("k") is second


def test_first_build_is_fresh_not_rebuilt(make_workspace: WorkspaceFactory) -> None:
    """Nothing was superseded on a cold cache; only an eviction is 'rebuilt'."""
    root = make_workspace(files=_FILES, symbols=_SYMBOLS)
    cache = VersionedEngineCache("test", recheck_seconds=0.0)
    _, freshness = cache.get("k", root, object)
    assert freshness == FRESHNESS_FRESH


def test_recheck_throttle_defers_the_probe(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    clock = _Clock()
    cache = VersionedEngineCache("test", recheck_seconds=5.0, clock=clock)

    first, _ = cache.get("k", root, object)
    _set_index_version(root, 23)

    clock.now = 4.9
    within_window, _ = cache.get("k", root, object)
    assert within_window is first, "probe re-read the version inside the throttle window"

    clock.now = 5.0
    after_window, freshness = cache.get("k", root, object)
    assert after_window is not first
    assert freshness == FRESHNESS_REBUILT


def test_concurrent_callers_rebuild_once(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    cache = VersionedEngineCache("test", recheck_seconds=0.0)
    cache.get("k", root, object)
    _set_index_version(root, 2)

    builds = 0
    builds_lock = threading.Lock()
    start = threading.Barrier(8)

    def build() -> object:
        nonlocal builds
        with builds_lock:
            builds += 1
        return object()

    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        start.wait(timeout=10)
        value, _ = cache.get("k", root, build)
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert builds == 1, f"{builds} concurrent rebuilds; the double-check under the lock did not hold"
    assert len({id(value) for value in results}) == 1


def test_rebuilding_index_raises_instead_of_serving(make_workspace: WorkspaceFactory) -> None:
    """The regression that matters most: never hand back a torn or empty view."""
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    cache = VersionedEngineCache("test", recheck_seconds=0.0)
    cache.get("k", root, object)

    _drop_table(root, "symbols")

    with pytest.raises(IndexRebuilding) as excinfo:
        cache.get("k", root, object)
    assert "symbols" in excinfo.value.detail
    assert "retry shortly" in str(excinfo.value)


def test_rebuilding_index_raises_even_on_a_cold_cache(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(symbols=_SYMBOLS, index_version=1)  # torn: symbols, no files
    cache = VersionedEngineCache("test", recheck_seconds=0.0)
    with pytest.raises(IndexRebuilding):
        cache.get("k", root, object)
    assert len(cache) == 0


def test_index_rebuilding_is_not_a_code_intel_unavailable() -> None:
    """Existing ``CodeIntelUnavailable`` handlers degrade to empty results.

    Inheriting from it would let this fix be swallowed by the very code paths
    it exists to correct, so the relationship is asserted, not assumed.
    """
    from lemoncrow.infra.code_intel.store import CodeIntelUnavailable

    assert not issubclass(IndexRebuilding, CodeIntelUnavailable)


def test_clear_drops_entries_and_probes(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, index_version=1)
    cache = VersionedEngineCache("test", recheck_seconds=1000.0)
    first, _ = cache.get("k", root, object)
    assert "k" in cache

    cache.clear()
    assert len(cache) == 0
    assert cache.peek("k") is None

    _set_index_version(root, 7)
    rebuilt, freshness = cache.get("k", root, object)
    assert rebuilt is not first
    # A cleared cache has nothing to supersede, so this is a cold build.
    assert freshness == FRESHNESS_FRESH
    assert cache.version_of("k") == 7
