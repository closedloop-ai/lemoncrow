"""Which checkout a daemon request works in: its session root (ISS-10812).

Every session on a repo shares the main checkout's MCP daemon, whose workspace
root is fixed at launch. A session that enters a linked git worktree keeps
calling that daemon, so unless a request is routed its searches answer from the
main checkout's index and its relative paths resolve into the main checkout.

A request's root is decided by this precedence:

1. an explicit ``repo_root`` / ``root`` argument, which the tool honors before
   asking this module;
2. absolute code-tool ``paths`` that all fall inside one linked worktree
   (:func:`code_root`);
3. the session's recorded ``cwd``. The plugin's PreToolUse hook writes it to
   ``<store root>/session_cwd/<session id>`` before every lc call. It counts only
   when it is the workspace root or inside one of its linked worktrees;
4. for relative edit and read paths only, the session's last explicit ``bash``
   ``cwd`` (:func:`path_root`);
5. the workspace root.

Workspace-keyed state (ledger, handover, compression, the workflow executor,
auto-init, warmers, savings) never reads the session root.

Known limitations:

* A session and its subagents share one session id. A subagent working in its
  own worktree (Agent ``isolation: "worktree"``) and a parent in the main
  checkout overwrite each other's recorded ``cwd``, so a call that names no
  checkout follows whichever of them called last. A call that names the worktree
  through absolute ``paths`` or ``repo_root`` still routes correctly, and an edit
  whose relative path names an existing file in both checkouts is refused as
  ambiguous instead of guessed.
* A worktree added with ``git worktree add --relative-paths`` records a relative
  ``gitdir``, which :func:`linked_worktree_of` does not resolve. Such a worktree
  is never accepted, and its session's calls fall back to the workspace root.
* Hosts without the hook (Codex, Cursor) get steps 1, 2, 4 and 5.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.worktree_seed import linked_worktree_of

__all__ = [
    "BASH_CWD",
    "BASH_CWD_IDLE_S",
    "MAX_SESSIONS",
    "PATHS",
    "PATH_ROUTED_TOOLS",
    "SESSION_CWD",
    "SESSION_CWD_DIRNAME",
    "WORKSPACE",
    "BashCwds",
    "PathsSpanWorktrees",
    "SessionRoot",
    "accept_session_cwd",
    "begin_request",
    "code_root",
    "disclose",
    "end_request",
    "path_root",
    "record_bash_cwd",
    "routed_paths",
    "session_cwd_file",
    "session_worktree",
    "worktree_of_paths",
]

#: Directory under the store root holding one file per session, named by its id.
#: The plugin's ``mcp_read_allow.py`` writes it and ``session_start.py`` prunes it.
SESSION_CWD_DIRNAME = "session_cwd"
#: Sessions whose bash cwd (and recorded-cwd read) is remembered at once.
MAX_SESSIONS = 256
#: A session's bash cwd is forgotten after this long without use.
BASH_CWD_IDLE_S = 24 * 3600.0

#: :attr:`SessionRoot.source` values: which precedence step answered.
PATHS = "paths"
SESSION_CWD = "session_cwd"
BASH_CWD = "bash_cwd"
WORKSPACE = "workspace"

#: Tools whose ``paths`` / ``path`` arguments can route the call (step 2).
PATH_ROUTED_TOOLS: frozenset[str] = frozenset({"code_search", "code_coverage_check", "code_changes", "graph"})

# A session id names a file, so it must never carry a path separator or a dot.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")


@dataclass(frozen=True)
class SessionRoot:
    """The checkout a request answers from, and the precedence step that chose it.

    ``source`` is :data:`WORKSPACE` exactly when ``root`` is the workspace root.
    """

    root: Path
    source: str


class PathsSpanWorktrees(ValueError):
    """A code tool's absolute ``paths`` name files in more than one linked worktree."""


class BashCwds:
    """Each session's last explicit ``bash`` ``cwd``, bounded in count and idle time."""

    def __init__(
        self,
        *,
        max_sessions: int = MAX_SESSIONS,
        idle_s: float = BASH_CWD_IDLE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_sessions = max_sessions
        self._idle_s = idle_s
        self._clock = clock
        self._lock = threading.Lock()
        # Least recently used first; a value is (cwd, monotonic time of last use).
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def record(self, session_key: str, cwd: str) -> None:
        now = self._clock()
        with self._lock:
            self._entries[session_key] = (cwd, now)
            self._entries.move_to_end(session_key)
            self._expire(now)
            while len(self._entries) > self._max_sessions:
                self._entries.popitem(last=False)

    def get(self, session_key: str) -> str | None:
        now = self._clock()
        with self._lock:
            self._expire(now)
            entry = self._entries.get(session_key)
            if entry is None:
                return None
            self._entries[session_key] = (entry[0], now)
            self._entries.move_to_end(session_key)
            return entry[0]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _expire(self, now: float) -> None:
        # Entries are ordered by last use, so the idle ones are all at the front.
        while self._entries:
            key, (_cwd, last_used) = next(iter(self._entries.items()))
            if now - last_used < self._idle_s:
                return
            del self._entries[key]


class _SessionCwdFiles:
    """Recorded session cwds, re-read only when the file's inode or mtime moves.

    The hook replaces the file by rename on every change, so a new value always
    arrives under a new inode.
    """

    def __init__(self, max_entries: int = MAX_SESSIONS) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[int, int, str | None]] = OrderedDict()

    def read(self, path: Path) -> str | None:
        key = str(path)
        try:
            stat = path.stat()
        except OSError:
            with self._lock:
                self._cache.pop(key, None)
            return None
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] == stat.st_ino and hit[1] == stat.st_mtime_ns:
                self._cache.move_to_end(key)
                return hit[2]
        try:
            value: str | None = path.read_text(encoding="utf-8").strip() or None
        except (OSError, UnicodeDecodeError):
            value = None
        with self._lock:
            self._cache[key] = (stat.st_ino, stat.st_mtime_ns, value)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return value


class _Request:
    """What one request carries for routing, plus the answers already worked out."""

    def __init__(
        self,
        *,
        session_id: Callable[[], str],
        bash_key: str,
        paths: tuple[str, ...],
        store_root: Callable[[], Path],
    ) -> None:
        self.session_id = session_id
        self.bash_key = bash_key
        self.paths = paths
        self.store_root = store_root
        self.resolved_session_id: str | None = None
        self.checkouts: dict[str, Path | None] = {}
        self.code_roots: dict[str, SessionRoot] = {}
        #: The code root this request answered from, for :func:`disclose`.
        self.answered: SessionRoot | None = None


_request = threading.local()
_bash_cwds = BashCwds()
_cwd_files = _SessionCwdFiles()


def begin_request(
    *,
    session_id: Callable[[], str],
    bash_key: str,
    paths: tuple[str, ...],
    store_root: Callable[[], Path],
) -> _Request | None:
    """Scope routing to the request running on this thread; return the prior scope.

    *session_id* is called at most once, and only when a tool needs the recorded
    cwd. It returns "" when the request names no session. *bash_key* keys the
    per-session bash cwd map. *paths* are the call's routing paths
    (:func:`routed_paths`).
    """
    prior = _current()
    _request.value = _Request(session_id=session_id, bash_key=bash_key, paths=paths, store_root=store_root)
    return prior


def end_request(prior: _Request | None) -> None:
    _request.value = prior


def _current() -> _Request | None:
    value = getattr(_request, "value", None)
    return value if isinstance(value, _Request) else None


def record_bash_cwd(session_key: str, cwd: str) -> None:
    """Remember a session's explicit ``bash`` ``cwd`` (precedence step 4)."""
    _bash_cwds.record(session_key, cwd)


def routed_paths(tool: str, args: Mapping[str, Any]) -> tuple[str, ...]:
    """The ``paths`` / ``path`` arguments of a tool that routes by them, else ().

    A ``tool`` broker call is judged by the tool it calls.
    """
    if tool == "tool":
        inner_name = args.get("name")
        inner_args = args.get("arguments")
        if isinstance(inner_name, str) and inner_name != "tool" and isinstance(inner_args, dict):
            return routed_paths(inner_name, inner_args)
        return ()
    if tool not in PATH_ROUTED_TOOLS:
        return ()
    found: list[str] = []
    for key in ("paths", "path"):
        value = args.get(key)
        if isinstance(value, str):
            found.extend(part.strip() for part in value.split(",") if part.strip())
        elif isinstance(value, list):
            found.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    return tuple(found)


def session_cwd_file(store_root: Path, session_id: str) -> Path | None:
    """Where the hook records *session_id*'s cwd, or None for an id that cannot name a file."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        return None
    return store_root / SESSION_CWD_DIRNAME / session_id


def accept_session_cwd(workspace: Path, cwd: str) -> Path | None:
    """The checkout a recorded *cwd* puts the session in, else None.

    That is the linked worktree of *workspace* containing *cwd*, or *workspace*
    itself when *cwd* is exactly the workspace root. Anything else -- another
    repo, a plain directory, a missing or relative path -- is never used.
    """
    candidate = Path(cwd).expanduser()
    if not candidate.is_absolute():
        return None
    worktree = linked_worktree_of(workspace, candidate)
    if worktree is not None:
        return worktree
    try:
        return workspace if candidate.resolve() == workspace.resolve() else None
    except OSError:
        return None


def _nearest_dir(path: Path) -> Path | None:
    for directory in (path, *path.parents):
        if directory.is_dir():
            return directory
    return None


def worktree_of_paths(workspace: Path, paths: Iterable[str]) -> Path | None:
    """The one linked worktree every absolute path in *paths* falls inside, else None (step 2).

    Relative paths are ignored: they resolve against whichever root is chosen.
    Raises :class:`PathsSpanWorktrees` when the absolute paths name two or more
    worktrees, since one call answers from one index.
    """
    worktrees: dict[str, Path] = {}
    elsewhere = False
    for raw in paths:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            continue
        directory = _nearest_dir(candidate)
        worktree = linked_worktree_of(workspace, directory) if directory is not None else None
        if worktree is None:
            elsewhere = True
        else:
            worktrees[str(worktree)] = worktree
    if len(worktrees) > 1:
        raise PathsSpanWorktrees(
            "paths span more than one worktree ("
            + ", ".join(sorted(worktrees))
            + "); one call searches one worktree's index -- split the call per worktree"
        )
    if len(worktrees) == 1 and not elsewhere:
        return next(iter(worktrees.values()))
    return None


def _recorded_checkout(ctx: _Request, workspace: Path) -> Path | None:
    """Step 3: the checkout the session's recorded cwd puts it in, else None."""
    key = str(workspace)
    if key in ctx.checkouts:
        return ctx.checkouts[key]
    if ctx.resolved_session_id is None:
        ctx.resolved_session_id = ctx.session_id()
    found: Path | None = None
    path = session_cwd_file(ctx.store_root(), ctx.resolved_session_id) if ctx.resolved_session_id else None
    cwd = _cwd_files.read(path) if path is not None else None
    if cwd:
        found = accept_session_cwd(workspace, cwd)
    ctx.checkouts[key] = found
    return found


def _as_root(workspace: Path, checkout: Path, source: str) -> SessionRoot:
    try:
        same = checkout.resolve() == workspace.resolve()
    except OSError:
        same = False
    return SessionRoot(workspace, WORKSPACE) if same else SessionRoot(checkout, source)


def code_root(workspace: Path) -> SessionRoot:
    """Where this request's code tools answer from: steps 2, 3 and 5.

    The workspace root outside a request. Raises :class:`PathsSpanWorktrees`
    when the call's absolute paths name two worktrees.
    """
    ctx = _current()
    if ctx is None:
        return SessionRoot(workspace, WORKSPACE)
    key = str(workspace)
    found = ctx.code_roots.get(key)
    if found is None:
        worktree = worktree_of_paths(workspace, ctx.paths) if ctx.paths else None
        if worktree is not None:
            found = _as_root(workspace, worktree, PATHS)
        else:
            checkout = _recorded_checkout(ctx, workspace)
            found = (
                _as_root(workspace, checkout, SESSION_CWD)
                if checkout is not None
                else SessionRoot(workspace, WORKSPACE)
            )
        ctx.code_roots[key] = found
    ctx.answered = found
    return found


def path_root(workspace: Path) -> SessionRoot:
    """Where this request's relative edit and read paths resolve: steps 3, 4 and 5.

    Outside a request only step 4 applies, under the session key "" that a
    single-session stdio server records under.
    """
    ctx = _current()
    bash_key = ""
    if ctx is not None:
        checkout = _recorded_checkout(ctx, workspace)
        if checkout is not None:
            return _as_root(workspace, checkout, SESSION_CWD)
        bash_key = ctx.bash_key
    recorded = _bash_cwds.get(bash_key)
    worktree = linked_worktree_of(workspace, Path(recorded)) if recorded else None
    return SessionRoot(worktree, BASH_CWD) if worktree is not None else SessionRoot(workspace, WORKSPACE)


def session_worktree(workspace: Path) -> Path | None:
    """The linked worktree the session's recorded cwd puts it in (step 3 only), else None.

    A ``bash`` call with no ``cwd`` runs there.
    """
    ctx = _current()
    checkout = _recorded_checkout(ctx, workspace) if ctx is not None else None
    if checkout is None:
        return None
    found = _as_root(workspace, checkout, SESSION_CWD)
    return None if found.source == WORKSPACE else found.root


def disclose(result: object, rendered: str | None) -> str | None:
    """Name the checkout a code tool answered from when it is not the workspace root.

    Sets ``result["repo_root"]`` and returns *rendered* with one
    ``| repo_root: <path>`` suffix on its first line. Unchanged otherwise.
    """
    ctx = _current()
    answered = ctx.answered if ctx is not None else None
    if answered is None or answered.source == WORKSPACE:
        return rendered
    root = str(answered.root)
    if isinstance(result, dict):
        result["repo_root"] = root
    if not rendered:
        return rendered
    first, newline, rest = rendered.partition("\n")
    return f"{first} | repo_root: {root}{newline}{rest}"
