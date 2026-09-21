"""Seed a linked git worktree's code index from its main checkout's index.

A worktree index built from scratch costs a full build: minutes and gigabytes for
a large repo. The worktree's files are almost all identical to the main
checkout's, so its index starts as a copy-on-write clone of main's index
instead (``clonefile`` on APFS, ``--reflink`` on Linux, a plain copy
elsewhere). One incremental pass then re-extracts only the files whose content
differs.

The clone keeps main's ``repo_id``. Every row is keyed by it, and rewriting the
key would un-share every page of the clone, so the seed records an alias instead:
``engine_state['repo_id_alias:<worktree id>'] = <main id>``. The engine adopts
the alias when it opens the database (:func:`resolve_repo_id`). The alias is
keyed by the worktree's own id, so another repo sharing the same database file
keeps its own id and rows.

The seed also writes ``engine_state['seeded_from'] = <main root>@<main
index_version>``. A seeded index is never fully rebuilt or vacuumed: either
would rewrite every page of the clone. When its format goes stale, the factory
re-seeds it from main's rebuilt index.

This module also owns the worktree engines' lifecycle in the daemon: which cached
roots are worktree engines, when they retire, and where their Zoekt candidates
come from.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.paths import resolve_workspace_store_dir, workspace_store_dir
from lemoncrow.infra.code_intel.freshness import INDEX_LOCK_SUFFIX, IndexRebuilding
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, FTS_DB, INTEL_DB, VECTORS_DB

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "ALIAS_KEY_PREFIX",
    "BUSY",
    "INDEX_DBS",
    "SEEDED",
    "SEEDED_FROM_KEY",
    "UNAVAILABLE",
    "WORKTREE_ENGINE_IDLE_ENV",
    "SeedResult",
    "checkpoint_index",
    "ensure_seeded",
    "forget_worktree",
    "is_seeded_index",
    "linked_worktree_of",
    "main_root_of",
    "path_repo_id",
    "resolve_repo_id",
    "retire_worktree_engine",
    "seed_worktree_index",
    "seeded_main_root",
    "start_first_refresh",
    "worktree_engine_idle_s",
]

logger = logging.getLogger(__name__)

#: The databases an index consists of. Everything else in the store (session
#: state, blocks, rubrics, loop discipline, Zoekt shards) is never cloned.
INDEX_DBS: tuple[str, ...] = (FTS_DB, INTEL_DB, VECTORS_DB, CODE_CONTEXT_DB)

ALIAS_KEY_PREFIX = "repo_id_alias:"
SEEDED_FROM_KEY = "seeded_from"

SEEDED = "seeded"
BUSY = "busy"
UNAVAILABLE = "unavailable"

#: How long a seed waits for main's index-write lock before reporting busy.
SEED_LOCK_WAIT_S = 2.0
#: How long a seed waits for SQLite's own write lock on each of main's databases.
#: Writers outside the index flock (the retrieval cache, the centrality map) hold
#: it for one short transaction.
_WRITE_LOCK_WAIT_S = 2.0
#: How often the factory re-checks a worktree it already has an engine for, so a
#: worktree whose index went stale (main rebuilt to a new format) gets re-seeded.
SEED_RECHECK_S = 60.0

WORKTREE_ENGINE_IDLE_ENV = "LEMONCROW_WORKTREE_ENGINE_IDLE_S"
#: Seconds a worktree engine may go without a request before the daemon unloads
#: it (``code_context.worktree_engine_idle_s``). Measured on a symphony-alpha clone
#: (17,870 indexed files): five seeded worktree engines added 190 MiB RSS to a
#: process holding main's -- 140 MiB of it with the first, ~13 MiB for each one
#: after -- and reopening one takes about a second. Half an hour keeps an engine
#: through a pause in a session at that price.
DEFAULT_WORKTREE_ENGINE_IDLE_S = 1800.0

_STAGING_PREFIX = ".seed-"

_lock = threading.Lock()
#: Linked worktree root -> its main checkout, for every worktree engine the
#: factory has opened. The retire policy reads it: only these roots idle-retire.
_worktrees: dict[str, Path] = {}
#: Monotonic time a worktree's seed need was last checked.
_checked_at: dict[str, float] = {}


@dataclass(frozen=True)
class SeedResult:
    """How a seed attempt ended, with the timings the measurements report."""

    status: str
    detail: str = ""
    checkpoint_s: float = 0.0
    clone_s: float = 0.0
    wal_cloned: tuple[str, ...] = ()

    @property
    def seeded(self) -> bool:
        return self.status == SEEDED


@dataclass(frozen=True)
class _IndexFacts:
    files: int
    semantics: int | None
    index_version: int
    seeded_from: str | None


# -- repo identity -----------------------------------------------------------


def path_repo_id(root: Path) -> str:
    """The engine's ``repo_id`` for *root* before any alias: a hash of its resolved path."""
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]


def _open_ro(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def resolve_repo_id(db_path: Path, computed_id: str) -> str:
    """The ``repo_id`` an engine opening *db_path* as *computed_id* should use.

    A seeded worktree index carries an alias to its main checkout's id; every
    other database, including one shared by several repos, maps an id to itself.
    """
    if not db_path.is_file():
        return computed_id
    try:
        conn = _open_ro(db_path)
    except sqlite3.Error:
        return computed_id
    try:
        row = conn.execute("SELECT value FROM engine_state WHERE key = ?", (ALIAS_KEY_PREFIX + computed_id,)).fetchone()
    except sqlite3.Error:
        return computed_id
    finally:
        conn.close()
    return str(row[0]) if row is not None and row[0] else computed_id


def effective_repo_id(root: Path) -> str:
    """*root*'s ``repo_id`` in its own default index, after alias resolution."""
    return resolve_repo_id(workspace_store_dir(root) / CODE_CONTEXT_DB, path_repo_id(root))


def is_seeded_index(conn: sqlite3.Connection) -> bool:
    """Whether the index behind *conn* was seeded from a main checkout's."""
    try:
        return conn.execute("SELECT 1 FROM engine_state WHERE key = ?", (SEEDED_FROM_KEY,)).fetchone() is not None
    except sqlite3.Error:
        return False


# -- worktree geometry -------------------------------------------------------


def linked_worktree_of(workspace_root: Path, candidate_dir: Path) -> Path | None:
    """The linked worktree of ``workspace_root`` that contains ``candidate_dir``, else None.

    Detected without spawning git: a linked worktree's ``.git`` is a *file*
    holding ``gitdir: <path>``, and for a worktree of THIS repo that path lives
    under ``<workspace_root>/.git/worktrees/``. A normal checkout has ``.git``
    as a directory, which ends the walk immediately.

    Returns None for every uncertain case -- a plain directory, a worktree
    belonging to a different repo, the workspace root itself.
    """
    try:
        candidate = candidate_dir.expanduser().resolve()
        if not candidate.is_dir():
            return None
        root = workspace_root.resolve()
        worktrees_dir = (root / ".git" / "worktrees").resolve()
        for directory in (candidate, *candidate.parents):
            marker = directory / ".git"
            if marker.is_dir():
                return None  # a normal checkout, not a linked worktree
            if not marker.is_file():
                continue
            gitdir = marker.read_text(encoding="utf-8").strip()
            if not gitdir.startswith("gitdir:"):
                return None
            target = Path(gitdir.split(":", 1)[1].strip()).resolve()
            if not target.is_relative_to(worktrees_dir):
                return None  # a worktree, but of some other repo
            return None if directory == root else directory
    except OSError:
        return None
    return None


def main_root_of(worktree: Path) -> Path | None:
    """The main checkout *worktree* is a linked worktree of, else None.

    Reads ``<worktree>/.git`` (``gitdir: <main>/.git/worktrees/<name>``) and the
    admin directory's ``commondir``. None for a normal checkout, a submodule, a
    bare repository's worktree, or anything unreadable.
    """
    marker = worktree / ".git"
    try:
        if not marker.is_file():
            return None
        text = marker.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir:"):
            return None
        gitdir = Path(text.split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = worktree / gitdir
        gitdir = gitdir.resolve()
        if gitdir.parent.name != "worktrees":
            return None  # e.g. a submodule's .git/modules/<name>
        commondir_file = gitdir / "commondir"
        common = gitdir.parent.parent
        if commondir_file.is_file():
            raw = Path(commondir_file.read_text(encoding="utf-8").strip())
            common = (raw if raw.is_absolute() else gitdir / raw).resolve()
        if common.name != ".git" or not common.is_dir():
            return None  # a bare repository has no main checkout
        main = common.parent
        return main if main != worktree.resolve() else None
    except OSError:
        return None


# -- reading an index --------------------------------------------------------


def _current_semantics_version() -> int:
    from lemoncrow.pro.capabilities.code_context.engine import _CODE_INDEXER_SEMANTICS_VERSION

    return _CODE_INDEXER_SEMANTICS_VERSION


def _read_facts(db: Path, repo_id: str | None) -> _IndexFacts | None:
    """The index state a seed decision needs; None when *db* is missing or unreadable.

    *repo_id* None counts every file row, which is what a worktree's own index
    holds whatever key its rows were written under.
    """
    if not db.is_file():
        return None
    try:
        conn = _open_ro(db)
    except sqlite3.Error:
        return None
    try:
        state = {
            str(key): str(value)
            for key, value in conn.execute(
                "SELECT key, value FROM engine_state WHERE key IN (?, ?, ?)",
                ("indexer_semantics_version", "index_version", SEEDED_FROM_KEY),
            )
        }
        if repo_id is None:
            files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        else:
            files = int(conn.execute("SELECT COUNT(*) FROM files WHERE repo_id = ?", (repo_id,)).fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    semantics: int | None
    try:
        semantics = int(state["indexer_semantics_version"])
    except (KeyError, ValueError):
        semantics = None
    try:
        index_version = int(state.get("index_version", "0"))
    except ValueError:
        index_version = 0
    return _IndexFacts(
        files=files, semantics=semantics, index_version=index_version, seeded_from=state.get(SEEDED_FROM_KEY)
    )


def _main_unavailable(facts: _IndexFacts | None, current: int) -> str | None:
    """Why main's index cannot seed a worktree, or None when it can."""
    if facts is None or facts.files == 0:
        return "the main checkout has no index"
    if facts.semantics != current:
        return f"the main checkout's index is at semantics version {facts.semantics}, not {current}"
    return None


def _seed_reason(worktree_facts: _IndexFacts | None, main_facts: _IndexFacts | None, current: int) -> str | None:
    """Why a worktree index should be replaced by a seed, or None when it should stay."""
    if worktree_facts is None or worktree_facts.files == 0:
        return "missing"
    if worktree_facts.semantics != current:
        return "stale"
    if worktree_facts.seeded_from is None and main_facts is not None and worktree_facts.files < main_facts.files:
        return "partial"
    return None


# -- locking and cloning -----------------------------------------------------


@contextlib.contextmanager
def _flock(lock_path: Path, *, wait_s: float) -> Iterator[bool]:
    """The engine's cross-process index-write lock; yields whether it was acquired."""
    if fcntl is None:  # pragma: no cover - non-POSIX platforms
        yield True
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    try:
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        yield acquired
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _lock_held(lock_path: Path) -> bool:
    with _flock(lock_path, wait_s=0.0) as acquired:
        return not acquired


def _clone_file(src: Path, dst: Path) -> None:
    """Copy *src* to *dst*, sharing its blocks where the filesystem can."""
    if sys.platform == "darwin":
        command = ["cp", "-c", str(src), str(dst)]
    elif sys.platform.startswith("linux"):
        command = ["cp", "--reflink=auto", str(src), str(dst)]
    else:
        shutil.copyfile(src, dst)
        return
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        # clonefile needs both paths on one APFS volume; fall back to a real copy.
        shutil.copyfile(src, dst)


def _checkpoint(db: Path) -> bool:
    """Run a PASSIVE checkpoint on *db*; True when its WAL is now fully in the database file."""
    conn = sqlite3.connect(db, timeout=_WRITE_LOCK_WAIT_S)
    try:
        _busy, log_frames, checkpointed = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
    finally:
        conn.close()
    # (0, -1, -1) is a database that is not in WAL mode: there is no WAL to carry.
    return int(log_frames) == int(checkpointed)


def checkpoint_index(store: Path, *, attempts: int = 3, pause_s: float = 0.5) -> bool:
    """Fold the WALs of the index databases in *store* into their files; True when all are empty.

    Run after a reindex, so a seed usually finds a small WAL: a clone carries the
    un-checkpointed WAL along, and checkpointing it later rewrites -- un-shares --
    every page it touches. SQLite's own auto-checkpoint stops short while a reader
    holds an older snapshot and does not run again until the next commit, which
    left symphony-alpha with 195k frames (808 MB) in fts.sqlite's WAL: a 5.4 s
    checkpoint on the seed's path.
    """
    pending = [store / name for name in INDEX_DBS if (store / name).is_file()]
    for attempt in range(max(1, attempts)):
        if attempt:
            time.sleep(pause_s)
        remaining: list[Path] = []
        for db in pending:
            try:
                done = _checkpoint(db)
            except sqlite3.Error:
                done = False
            if not done:
                remaining.append(db)
        pending = remaining
        if not pending:
            return True
    return False


def _snapshot(main_store: Path, staging: Path) -> tuple[float, float, tuple[str, ...]]:
    """Clone main's index databases into *staging*; returns (checkpoint_s, clone_s, wal_cloned).

    Callers hold main's index flock, which stops every indexer. It does not stop
    the query-time writers (the retrieval cache in ``code_context.sqlite`` and the
    centrality map in ``intel.sqlite``), so each database is also held under
    SQLite's own write lock while it is cloned: with no commit in flight, the
    database file and its WAL are one consistent state. The bulk checkpoint runs
    before those locks are taken, so query-time writers wait only for the clone.
    """
    names = [name for name in INDEX_DBS if (main_store / name).is_file()]
    started = time.monotonic()
    for name in names:
        _checkpoint(main_store / name)
    checkpoint_s = time.monotonic() - started
    holders: list[sqlite3.Connection] = []
    wal_cloned: list[str] = []
    try:
        for name in names:
            holder = sqlite3.connect(main_store / name, timeout=_WRITE_LOCK_WAIT_S, isolation_level=None)
            holders.append(holder)
            holder.execute("BEGIN IMMEDIATE")
        cloning = time.monotonic()
        for name in names:
            source = main_store / name
            backfilled = _checkpoint(source)
            _clone_file(source, staging / name)
            wal = source.with_name(name + "-wal")
            if not backfilled and wal.is_file() and wal.stat().st_size > 0:
                _clone_file(wal, staging / wal.name)
                wal_cloned.append(name)
        clone_s = time.monotonic() - cloning
    finally:
        for holder in holders:
            with contextlib.suppress(sqlite3.Error):
                holder.execute("ROLLBACK")
            holder.close()
    return checkpoint_s, clone_s, tuple(wal_cloned)


def _mark(staging: Path, *, worktree_id: str, main_id: str, seeded_from: str, previous_version: int) -> None:
    """Stamp the staged clone as a seed and fold every cloned WAL into its database.

    The swap then moves database files only, so a WAL can never end up beside a
    database it does not belong to.

    The index_version is main's unless that equals the version the worktree index
    being replaced had: the daemon's engine cache rebuilds an engine only when the
    version it was built at moves, and an engine left on the replaced files would
    keep serving them.
    """
    code_db = staging / CODE_CONTEXT_DB
    conn = sqlite3.connect(code_db, timeout=5.0)
    try:
        upsert = (
            "INSERT INTO engine_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        conn.execute(upsert, (ALIAS_KEY_PREFIX + worktree_id, main_id))
        conn.execute(upsert, (SEEDED_FROM_KEY, seeded_from))
        row = conn.execute("SELECT value FROM engine_state WHERE key = 'index_version'").fetchone()
        version = int(row[0]) if row is not None else 0
        if version == previous_version:
            conn.execute(upsert, ("index_version", str(version + 1)))
        # Keyed by (repo_id, index_version), both of which the clone shares with
        # main: a cached payload would answer for the worktree with main's paths.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("DELETE FROM retrieval_cache")
        conn.commit()
    finally:
        conn.close()
    for name in INDEX_DBS:
        db = staging / name
        if db.is_file() and db.with_name(name + "-wal").exists():
            folder = sqlite3.connect(db, timeout=5.0)
            try:
                folder.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                folder.close()


def _sweep_staging(parent: Path) -> None:
    """Remove staging directories a crashed seed left behind."""
    with contextlib.suppress(OSError):
        for entry in parent.iterdir():
            if entry.name.startswith(_STAGING_PREFIX) and entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)


def _swap(staging: Path, store: Path, parent: Path) -> None:
    """Replace the worktree's index databases with the staged seed.

    Each database moves by one atomic rename over the old one, so its path is
    never missing and never half-copied. Its old WAL and shared-memory files are
    moved aside first: SQLite pairs a database with whatever ``-wal`` sits beside
    it, and the old one's frames would be applied to the seed.

    lc-debt: a reader already holding the replaced files open (another thread's
    in-flight query, another process) keeps reading them and can pair with the new
    ones until it reconnects; the daemon retires its own engine before the swap.
    Upgrade path: a versioned store directory switched by one rename.
    """
    aside = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX + "old-", dir=parent))
    try:
        for name in INDEX_DBS:
            for suffix in ("-wal", "-shm"):
                side = store / (name + suffix)
                if side.exists():
                    os.replace(side, aside / side.name)
            staged = staging / name
            if staged.is_file():
                os.replace(staged, store / name)
            elif (store / name).exists():
                os.replace(store / name, aside / name)
    finally:
        shutil.rmtree(aside, ignore_errors=True)


# -- seeding -----------------------------------------------------------------


def seed_worktree_index(
    worktree_root: Path,
    main_root: Path,
    *,
    lock_wait_s: float | None = None,
    before_swap: Callable[[], None] | None = None,
) -> SeedResult:
    """Replace *worktree_root*'s index with a clone of *main_root*'s.

    ``unavailable`` when main's index is empty or at an older format -- the caller
    serves the worktree as before. ``busy`` when main's index-write lock stays
    held past *lock_wait_s* (default :data:`SEED_LOCK_WAIT_S`), or the worktree's
    own index is being written.
    *before_swap* runs after the clone is staged and before it replaces anything,
    so an owner can retire an engine still reading the old files.
    """
    worktree = worktree_root.resolve()
    main = main_root.resolve()
    main_store = workspace_store_dir(main)
    main_db = main_store / CODE_CONTEXT_DB
    main_lock = main_store / (CODE_CONTEXT_DB + INDEX_LOCK_SUFFIX)
    main_id = effective_repo_id(main)
    current = _current_semantics_version()
    unavailable = _main_unavailable(_read_facts(main_db, main_id), current)
    if unavailable is not None:
        if main_lock.exists() and _lock_held(main_lock):
            return SeedResult(BUSY, f"the main checkout's index is being rebuilt ({unavailable})")
        return SeedResult(UNAVAILABLE, unavailable)

    store = resolve_workspace_store_dir(workspace_root=worktree)
    store.mkdir(parents=True, exist_ok=True)
    parent = store.parent
    previous = _read_facts(store / CODE_CONTEXT_DB, None)
    with _flock(store / (CODE_CONTEXT_DB + INDEX_LOCK_SUFFIX), wait_s=0.0) as own:
        if not own:
            return SeedResult(BUSY, "the worktree's own index is being written")
        _sweep_staging(parent)
        staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=parent))
        try:
            with _flock(main_lock, wait_s=SEED_LOCK_WAIT_S if lock_wait_s is None else lock_wait_s) as held:
                if not held:
                    return SeedResult(BUSY, "the main checkout's index is being written")
                facts = _read_facts(main_db, main_id)
                unavailable = _main_unavailable(facts, current)
                if facts is None or unavailable is not None:
                    return SeedResult(UNAVAILABLE, unavailable or "the main checkout has no index")
                try:
                    checkpoint_s, clone_s, wal_cloned = _snapshot(main_store, staging)
                except sqlite3.OperationalError as exc:
                    return SeedResult(BUSY, f"the main checkout's index is locked: {exc}")
            _mark(
                staging,
                worktree_id=path_repo_id(worktree),
                main_id=main_id,
                seeded_from=f"{main}@{facts.index_version}",
                previous_version=previous.index_version if previous is not None else -1,
            )
            if before_swap is not None:
                before_swap()
            _swap(staging, store, parent)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    logger.info(
        "worktree_seed: seeded %s from %s (checkpoint %.2fs, clone %.2fs, WAL cloned for %s)",
        worktree,
        main,
        checkpoint_s,
        clone_s,
        ", ".join(wal_cloned) or "none",
    )
    return SeedResult(SEEDED, "", checkpoint_s, clone_s, wal_cloned)


def _route_zoekt(worktree: Path, main: Path, *, seeded: bool) -> None:
    """Point a seeded worktree's Zoekt searches at its main checkout's server."""
    try:
        from lemoncrow.infra.code_intel.zoekt.adapter import clear_zoekt_root_override, set_zoekt_root_override
    except ImportError:  # pragma: no cover - zoekt adapter is part of the package
        return
    if seeded:
        set_zoekt_root_override(worktree, main)
    else:
        clear_zoekt_root_override(worktree)


def ensure_seeded(
    root: Path,
    *,
    cached: bool = False,
    reseed: bool = False,
    lock_wait_s: float | None = None,
    before_swap: Callable[[], None] | None = None,
) -> SeedResult | None:
    """Seed *root*'s index from its main checkout's when *root* is a linked worktree that needs one.

    None when *root* is not a linked worktree, or its index is complete and
    current. A missing index, a partial one (unseeded, fewer files than main's)
    and one at an older format are replaced; *reseed* replaces any. *cached* says
    the caller already holds an engine for *root*, which limits the check to one
    per :data:`SEED_RECHECK_S`.

    Raises :class:`IndexRebuilding` while main's index is being written: the
    worktree has nothing complete to answer from until the seed lands, and the
    next call retries.
    """
    key = str(root)
    now = time.monotonic()
    with _lock:
        main = _worktrees.get(key)
        checked = _checked_at.get(key)
    if main is not None and cached and not reseed and checked is not None and now - checked < SEED_RECHECK_S:
        return None
    if main is None:
        main = main_root_of(root)
        if main is None:
            return None
    with _lock:
        _worktrees[key] = main
        _checked_at[key] = now
    current = _current_semantics_version()
    worktree_facts = _read_facts(workspace_store_dir(root) / CODE_CONTEXT_DB, None)
    main_facts = _read_facts(workspace_store_dir(main) / CODE_CONTEXT_DB, effective_repo_id(main))
    reason = "reseed" if reseed else _seed_reason(worktree_facts, main_facts, current)
    if reason is None:
        _route_zoekt(root, main, seeded=worktree_facts is not None and worktree_facts.seeded_from is not None)
        return None
    result = seed_worktree_index(root, main, lock_wait_s=lock_wait_s, before_swap=before_swap)
    if result.status == BUSY:
        with _lock:
            _checked_at.pop(key, None)
        raise IndexRebuilding(root, f"seeding the worktree index from {main}: {result.detail}")
    if result.seeded:
        logger.info("worktree_seed: %s index was %s; seeded from %s", root, reason, main)
    seeded = result.seeded or (worktree_facts is not None and worktree_facts.seeded_from is not None)
    _route_zoekt(root, main, seeded=seeded)
    return result


def seeded_main_root(root: Path) -> Path | None:
    """The main checkout *root*'s index was seeded from, when *root* is a seeded linked worktree."""
    main = main_root_of(root)
    if main is None:
        return None
    facts = _read_facts(workspace_store_dir(root) / CODE_CONTEXT_DB, None)
    return main if facts is not None and facts.seeded_from is not None else None


def _first_refresh(engine: Any) -> None:
    try:
        with engine._autosync_lock:
            engine._maybe_autosync_reindex_locked(known_change="seeded")
    except Exception:
        logger.exception("worktree_seed: first refresh after the seed failed")


def start_first_refresh(engine: Any) -> None:
    """Bring a just-seeded engine's index to the worktree's files now, off the request path.

    Queries answer from the seed meanwhile, marked refreshing. Without this the
    worktree's own edits would wait for autosync's next full-tree poll.
    """
    if not getattr(engine, "_autosync_enabled", False):
        return
    threading.Thread(target=_first_refresh, args=(engine,), name="lemoncrow-worktree-seed-refresh", daemon=True).start()


# -- lifecycle ---------------------------------------------------------------


def worktree_engine_idle_s() -> float:
    """``code_context.worktree_engine_idle_s``: idle seconds before a worktree engine unloads."""
    raw = os.environ.get(WORKTREE_ENGINE_IDLE_ENV, "").strip()
    if not raw:
        return DEFAULT_WORKTREE_ENGINE_IDLE_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_WORKTREE_ENGINE_IDLE_S


def forget_worktree(root: Path | str) -> None:
    """Drop everything this module remembers about the worktree engine at *root*."""
    key = str(root)
    with _lock:
        main = _worktrees.pop(key, None)
        _checked_at.pop(key, None)
    if main is not None:
        _route_zoekt(Path(key), main, seeded=False)


def retire_worktree_engine(key: str, idle_s: float) -> bool:
    """Retire policy for the daemon's engine cache: True retires the entry at *key*.

    Only worktree engines retire: when the worktree is gone, or when they have had
    no request for :func:`worktree_engine_idle_s`. A main checkout's engine never
    does. A retiring worktree is forgotten here, Zoekt routing included.

    Gone means its ``.git`` file is gone, not its directory: a reindex still
    running when ``git worktree remove`` deleted the checkout writes its store
    back, and that recreates the directory.
    """
    with _lock:
        tracked = key in _worktrees
    if not tracked:
        return False
    if (Path(key) / ".git").is_file() and idle_s < worktree_engine_idle_s():
        return False
    forget_worktree(key)
    return True
