"""Detect when the closed engine's index has moved under a cached object.

The MCP server's ``_code_engine_cache`` was populate-once: one
``CodeContextEngine`` per repo path, built at whatever index generation happened
to be current, and then reused for the entire life of the daemon. Measured in
production, a four-hour-old daemon built at ``index_version 1`` was still
serving at version 23 -- twenty-two reindexes later -- and returned **empty
results for every query**, including symbols plainly present in the index. It
failed silently: no error, no warning, just nothing found.

Two mechanisms live here:

* :func:`index_state` -- a read-only probe of ``engine_state.index_version``
  plus enough structural checks to tell "the index is mid-write" apart from
  "the index is empty". Those are different answers and only one of them is an
  error.
* :class:`VersionedEngineCache` -- a process cache whose entries are stamped
  with the generation they were built against, and rebuilt on mismatch.

The rule both serve is **fail loud, never empty**. An empty result set from an
index that is mid-rebuild is indistinguishable from a true negative, which
makes it the worst failure this subsystem has available: a wrong answer
delivered with the same confidence as a right one. A reviewer reads "no
matches" as "this symbol has no callers" and files a finding on it. Callers get
an exception they can catch instead.

Nothing here writes to the engine's databases; see
:mod:`lemoncrow.infra.code_intel.store` for that boundary.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir

__all__ = [
    "DEFAULT_RECHECK_SECONDS",
    "FRESHNESS_FRESH",
    "FRESHNESS_REBUILT",
    "INDEX_LOCK_SUFFIX",
    "LOCK_FREE",
    "LOCK_HELD",
    "LOCK_UNKNOWN",
    "STATUS_ABSENT",
    "STATUS_READY",
    "STATUS_REBUILDING",
    "IndexRebuilding",
    "IndexState",
    "VersionedEngineCache",
    "index_state",
]

logger = logging.getLogger(__name__)

#: How long a version probe is reused before the on-disk value is re-read. A
#: hot ``code_search`` path must not pay a SQLite open per call; the cost is a
#: bounded, known staleness window in place of an unbounded one.
DEFAULT_RECHECK_SECONDS = 5.0

#: The engine's index-write lock sits beside ``code_context.sqlite``.
INDEX_LOCK_SUFFIX = ".indexlock"

STATUS_READY = "ready"
STATUS_REBUILDING = "rebuilding"
STATUS_ABSENT = "absent"

FRESHNESS_FRESH = "fresh"
FRESHNESS_REBUILT = "rebuilt"

LOCK_FREE = "free"
LOCK_HELD = "held"
LOCK_UNKNOWN = "unknown"

#: Tables ``code_context.sqlite`` always has once the engine has finished a
#: pass. A missing one means we caught the DDL mid-flight.
_REQUIRED_TABLES = frozenset({"files", "symbols", "imports", "engine_state"})

_PROBE_BUSY_TIMEOUT_MS = 2_000


class IndexRebuilding(RuntimeError):
    """The engine's index is mid-write, so any result would be incomplete.

    Deliberately **not** a subclass of
    :class:`~lemoncrow.infra.code_intel.store.CodeIntelUnavailable`. Handlers
    for that exception degrade to an empty result, which is exactly the outcome
    this error exists to prevent -- inheriting from it would let the fix be
    swallowed by the code it is fixing.
    """

    def __init__(self, repo_root: Path | str, detail: str) -> None:
        super().__init__(
            f"code index for {repo_root} is being rebuilt ({detail}); "
            "results would be incomplete -- retry shortly"
        )
        self.repo_root = str(repo_root)
        self.detail = detail


@dataclass(frozen=True)
class IndexState:
    """How the engine's index looks to a read-only observer.

    ``status`` is the field that gates behaviour:

    ``ready``
        The index can be read.
    ``rebuilding``
        Mid-write. A query against it would return a torn or empty view, so
        callers must raise rather than return what they find.
    ``absent``
        Never indexed, or indexed to nothing. An answer, not a failure -- the
        engine creates the databases on first use.
    """

    index_version: int
    status: str
    detail: str
    lock: str

    @property
    def rebuilding(self) -> bool:
        return self.status == STATUS_REBUILDING


def index_lock_path(repo_root: Path | str = ".") -> Path:
    """Path of the engine's index-write lock for *repo_root*."""
    return Path(str(workspace_dir(repo_root) / CODE_CONTEXT_DB) + INDEX_LOCK_SUFFIX)


def _probe_lock(lock_path: Path) -> str:
    """Best-effort answer to "is the index-write lock held right now?".

    The lock file is created once and persists across runs, so its *existence*
    proves nothing; only an attempted acquisition does. The probe takes a
    shared lock and releases it immediately, so it never blocks a writer for
    longer than the syscall pair.

    A probe that cannot run returns :data:`LOCK_UNKNOWN` and never
    :data:`LOCK_FREE`. Reporting a lock as free on the strength of a check that
    did not happen is how a silent wrong answer gets manufactured.
    """
    if not lock_path.exists():
        return LOCK_UNKNOWN
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        return LOCK_UNKNOWN
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except OSError:  # pragma: no cover - permission dependent
        return LOCK_UNKNOWN
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        return LOCK_HELD
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return LOCK_FREE
    finally:
        os.close(fd)


def _engine_int(conn: sqlite3.Connection, key: str) -> int:
    row = conn.execute("SELECT value FROM engine_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return 0
    try:
        return int(str(row[0]))
    except ValueError:
        return 0


def index_state(repo_root: Path | str = ".") -> IndexState:
    """Probe ``code_context.sqlite`` without opening it for write.

    The checks run cheapest-first and stop at the first one that settles the
    question:

    1. the database file is missing -> ``absent``
    2. a required table is missing -> ``rebuilding`` (caught mid-DDL)
    3. ``files`` and ``symbols`` disagree about being empty -> ``rebuilding``
       (a torn index; a genuinely empty one has neither)
    4. the index-write lock is held -> ``rebuilding``
    5. no rows at all -> ``absent``; otherwise ``ready``
    """
    db = workspace_dir(repo_root) / CODE_CONTEXT_DB
    lock = _probe_lock(Path(str(db) + INDEX_LOCK_SUFFIX))
    if not db.exists():
        return IndexState(0, STATUS_ABSENT, f"{db} does not exist", lock)
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error as exc:  # pragma: no cover - platform dependent
        return IndexState(0, STATUS_REBUILDING, f"cannot open index read-only: {exc}", lock)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_PROBE_BUSY_TIMEOUT_MS}")
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = sorted(_REQUIRED_TABLES - tables)
        if missing:
            return IndexState(0, STATUS_REBUILDING, f"tables missing: {', '.join(missing)}", lock)
        version = _engine_int(conn, "index_version")
        counts = conn.execute("SELECT (SELECT COUNT(*) FROM files), (SELECT COUNT(*) FROM symbols)").fetchone()
        files = int(counts[0])
        symbols = int(counts[1])
        if bool(files) != bool(symbols):
            return IndexState(
                version,
                STATUS_REBUILDING,
                f"index partially populated ({files} files, {symbols} symbols)",
                lock,
            )
        if lock == LOCK_HELD:
            return IndexState(version, STATUS_REBUILDING, "index-write lock is held", lock)
        if files == 0:
            return IndexState(version, STATUS_ABSENT, "index is empty", lock)
        return IndexState(version, STATUS_READY, "", lock)
    except sqlite3.Error as exc:
        # A torn database mid-rebuild reads as corruption. That is a rebuild in
        # progress, not a permanent failure, and it must not surface as empty.
        return IndexState(0, STATUS_REBUILDING, f"index unreadable: {exc}", lock)
    finally:
        conn.close()


@dataclass
class _Entry:
    value: Any
    index_version: int


@dataclass
class _Probe:
    state: IndexState
    checked_at: float


class VersionedEngineCache:
    """Per-repo object cache stamped with the index generation it was built at.

    Replaces the populate-once ``dict`` this subsystem used to keep. An entry is
    reused only while the on-disk ``engine_state.index_version`` still matches
    the version it was built against; on a mismatch the entry is dropped and
    rebuilt, and the eviction is logged at INFO -- the failure this fixes was
    invisible for hours, and the fix should not be equally quiet.

    Rebuilds happen under a lock with a double check, so N concurrent callers
    arriving at a version bump together produce one rebuild, not N.
    """

    def __init__(
        self,
        name: str,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.recheck_seconds = float(recheck_seconds)
        self.evictions = 0
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._probes: dict[str, _Probe] = {}

    # -- probing -----------------------------------------------------------

    def state_for(self, repo_root: Path | str) -> IndexState:
        """:func:`index_state`, re-read at most once per *recheck_seconds*."""
        key = str(repo_root)
        now = self._clock()
        probe = self._probes.get(key)
        if probe is not None and (now - probe.checked_at) < self.recheck_seconds:
            return probe.state
        state = index_state(repo_root)
        self._probes[key] = _Probe(state=state, checked_at=now)
        return state

    # -- cache -------------------------------------------------------------

    def get(self, key: str, repo_root: Path | str, build: Callable[[], Any]) -> tuple[Any, str]:
        """Return ``(value, freshness)``, rebuilding when the index has moved.

        *freshness* is :data:`FRESHNESS_FRESH` when the cached value was still
        valid (or was built for the first time) and :data:`FRESHNESS_REBUILT`
        when a superseded entry was evicted to produce it. Callers should carry
        that onto their response so a consumer can tell the two apart.

        Raises :class:`IndexRebuilding` when the index is mid-write. Returning
        results from a torn index is the failure mode this whole module exists
        to remove.
        """
        state = self.state_for(repo_root)
        if state.rebuilding:
            raise IndexRebuilding(repo_root, state.detail)
        entry = self._entries.get(key)
        if entry is not None and entry.index_version == state.index_version:
            return entry.value, FRESHNESS_FRESH
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.index_version == state.index_version:
                return entry.value, FRESHNESS_FRESH
            superseded = entry.index_version if entry is not None else None
            if superseded is not None:
                logger.info(
                    "%s: evicting cached entry for %s -- index moved %s -> %s",
                    self.name,
                    key,
                    superseded,
                    state.index_version,
                )
                self.evictions += 1
            value = build()
            self._entries[key] = _Entry(value=value, index_version=state.index_version)
            return value, FRESHNESS_FRESH if superseded is None else FRESHNESS_REBUILT

    def peek(self, key: str) -> Any | None:
        """The cached value for *key* without probing or building."""
        entry = self._entries.get(key)
        return None if entry is None else entry.value

    def version_of(self, key: str) -> int | None:
        """The index generation *key*'s cached value was built against."""
        entry = self._entries.get(key)
        return None if entry is None else entry.index_version

    def discard(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._probes.clear()
            self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries
