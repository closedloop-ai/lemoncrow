"""F2 -- a diff, mapped onto the symbols it changed and the callers it reaches.

Review tools ask the same question every time: *what does this change touch?*
Answering it by reading the diff alone stops at the file boundary. This module
carries it one step further -- changed line ranges become changed symbols, and
changed symbols become the callers that reference them.

The pipeline is deliberately boring:

1. ``git diff --unified=0`` against the merge base -> changed line ranges per file
2. ranges intersected with ``symbols.start_line``/``end_line`` -> changed symbols
3. ``call_edges`` and ``references`` reversed on the changed symbol's name
   -> impacted callers
4. repeat step 3 to *depth* hops
5. a stated risk rule per changed symbol

**Precision caveat, and it is not a footnote.** ``call_edges`` stores the callee
as raw dotted call text with no ``symbol_id``, so every reverse lookup here is
name-matched. A change to a method called ``run`` reports every ``.run()`` call
in the repository. The error is one-directional -- this over-reports and never
under-reports -- which is the safe direction for impact analysis, but callers
must be told which kind of match they got rather than left to assume. Every
edge therefore carries ``match_kind``, fixed at ``"name"`` until a resolution
sidecar (F9) exists to make ``"resolved"`` true.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.completeness import OBJECTIVE_EXHAUSTIVE
from lemoncrow.infra.code_intel.store import CodeIntelStore, SymbolRow

__all__ = [
    "MATCH_NAME",
    "MATCH_RESOLVED",
    "ChangeImpactReport",
    "ChangedSymbol",
    "FileChange",
    "ImpactedCaller",
    "LineRange",
    "analyze_changes",
]

#: Name-matched: the callee text equalled the symbol's name. Over-reports.
MATCH_NAME = "name"
#: Resolved to a symbol id. Nothing produces this until F9's sidecar lands.
MATCH_RESOLVED = "resolved"

STATUS_ADDED = "added"
STATUS_MODIFIED = "modified"
STATUS_DELETED = "deleted"
STATUS_RENAMED = "renamed"

RISK_HIGH = "high"
RISK_MEDIUM = "medium"
RISK_LOW = "low"

#: A changed symbol with at least this many callers is high risk when exported.
_HIGH_RISK_CALLERS = 5

_GIT_TIMEOUT_S = 30.0

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")

_TEST_DIR_MARKERS = ("tests/", "test/", "spec/", "__tests__/")
_TEST_FILE_SUFFIXES = ("_test.py", "_test.go", ".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.js")


class GitUnavailable(RuntimeError):
    """*repo_root* is not a git worktree, or git could not be run there."""


@dataclass(frozen=True)
class LineRange:
    """An inclusive line span in the post-change file."""

    start: int
    end: int

    def overlaps(self, start: int, end: int) -> bool:
        return self.start <= end and start <= self.end

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class FileChange:
    """One file in the diff, with the post-change line spans that moved.

    A deleted file has no ranges: nothing survives to intersect with. Its
    symbols come from the index instead, which is exactly the case worth
    reporting -- the index still holds symbols that no longer exist on disk.
    """

    path: str
    status: str
    ranges: tuple[LineRange, ...]
    old_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "status": self.status,
            "ranges": [line_range.to_dict() for line_range in self.ranges],
        }
        if self.old_path is not None:
            payload["old_path"] = self.old_path
        return payload


@dataclass(frozen=True)
class ChangedSymbol:
    """An indexed symbol whose body intersects the diff."""

    symbol_id: str
    symbol_name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    status: str
    exported: bool
    callers: int
    test_callers: int
    risk: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "name": self.symbol_name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "status": self.status,
            "exported": self.exported,
            "callers": self.callers,
            "test_callers": self.test_callers,
            "risk": self.risk,
        }


@dataclass(frozen=True)
class ImpactedCaller:
    """A site that reaches a changed symbol, and how far away it sits."""

    name: str
    qualified_name: str
    file_path: str
    line: int
    depth: int
    via: str
    match_kind: str
    is_test: bool
    changed_symbol: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "path": self.file_path,
            "line": self.line,
            "depth": self.depth,
            "via": self.via,
            "match_kind": self.match_kind,
            "is_test": self.is_test,
            "changed_symbol": self.changed_symbol,
        }


@dataclass(frozen=True)
class ChangeImpactReport:
    """Everything the diff touched, stamped with the index generation used."""

    repo_root: str
    base_ref: str
    diff_ref: str
    engine_index_version: int
    depth: int
    match_kind: str
    files: tuple[FileChange, ...]
    changed_symbols: tuple[ChangedSymbol, ...]
    impacted: tuple[ImpactedCaller, ...]
    impacted_total: int
    truncated: bool
    unindexed_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            # Name-keyed matching over-reports and never misses, so the caller
            # list is a superset of the truth -- exhaustive in the sense that
            # matters for impact analysis.
            "objective": OBJECTIVE_EXHAUSTIVE,
            "repo_root": self.repo_root,
            "base_ref": self.base_ref,
            "diff_ref": self.diff_ref,
            "engine_index_version": self.engine_index_version,
            "depth": self.depth,
            "match_kind": self.match_kind,
            "files": [change.to_dict() for change in self.files],
            "changed_symbols": [symbol.to_dict() for symbol in self.changed_symbols],
            "impacted": [caller.to_dict() for caller in self.impacted],
            "impacted_total": self.impacted_total,
            "truncated": self.truncated,
            "unindexed_paths": list(self.unindexed_paths),
        }


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitUnavailable(f"git failed in {repo_root}: {exc}") from exc


def _diff_ref(repo_root: Path, base_ref: str) -> str:
    """Resolve *base_ref* to the ref the diff should actually run against.

    Three-dot semantics: a review wants "what this branch did", not "how this
    branch differs from a base that also moved". ``git merge-base`` gives the
    fork point; diffing from there excludes commits the base gained
    independently. Falls back to *base_ref* itself when there is no common
    ancestor (unrelated histories, or a ref that is not a commit).
    """
    merge_base = _git(repo_root, "merge-base", base_ref, "HEAD")
    if merge_base.returncode == 0 and merge_base.stdout.strip():
        return merge_base.stdout.strip()
    return base_ref


def parse_diff(diff_text: str) -> list[FileChange]:
    """Parse ``git diff --unified=0`` output into per-file line spans.

    With zero context every hunk header is exactly the changed span, so no
    guessing is needed about which lines were context and which were content.

    A pure deletion arrives as ``+N,0`` -- zero lines at position N in the new
    file. There is no post-change span to point at, so the anchor is line N
    itself: whatever symbol still spans that point is the one that lost code.
    """
    changes: list[FileChange] = []
    path: str | None = None
    old_path: str | None = None
    status = STATUS_MODIFIED
    ranges: list[LineRange] = []

    def flush() -> None:
        nonlocal path, old_path, status, ranges
        if path is not None:
            changes.append(
                FileChange(path=path, status=status, ranges=tuple(ranges), old_path=old_path),
            )
        path = None
        old_path = None
        status = STATUS_MODIFIED
        ranges = []

    for line in diff_text.splitlines():
        header = _DIFF_HEADER.match(line)
        if header is not None:
            flush()
            path = header.group(2)
            continue
        if path is None:
            continue
        if line.startswith("new file mode"):
            status = STATUS_ADDED
            continue
        if line.startswith("deleted file mode"):
            status = STATUS_DELETED
            continue
        if line.startswith("rename from "):
            status = STATUS_RENAMED
            old_path = line[len("rename from ") :]
            continue
        if line.startswith("rename to "):
            status = STATUS_RENAMED
            path = line[len("rename to ") :]
            continue
        hunk = _HUNK.match(line)
        if hunk is None:
            continue
        start = int(hunk.group(3))
        count = int(hunk.group(4)) if hunk.group(4) is not None else 1
        if count == 0:
            anchor = max(1, start)
            ranges.append(LineRange(anchor, anchor))
        else:
            ranges.append(LineRange(start, start + count - 1))

    flush()
    return changes


def collect_changes(
    repo_root: Path, base_ref: str = "HEAD", paths: list[str] | None = None
) -> tuple[str, list[FileChange]]:
    """Run the diff and parse it. Returns ``(diff_ref, changes)``."""
    if not (repo_root / ".git").exists():
        probe = _git(repo_root, "rev-parse", "--git-dir")
        if probe.returncode != 0:
            raise GitUnavailable(f"{repo_root} is not a git worktree")
    diff_ref = _diff_ref(repo_root, base_ref)
    args = ["diff", "--unified=0", "--no-color", "--find-renames", diff_ref]
    if paths:
        args.extend(["--", *paths])
    result = _git(repo_root, *args)
    if result.returncode != 0:
        raise GitUnavailable(f"git diff against {base_ref} failed: {result.stderr.strip()}")
    return diff_ref, parse_diff(result.stdout)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name.startswith("test_") or name.endswith(_TEST_FILE_SUFFIXES):
        return True
    return any(marker in f"{normalized}/" for marker in _TEST_DIR_MARKERS)


def _is_exported(symbol_name: str, language: str) -> bool:
    """Whether *symbol_name* is part of its module's public surface.

    Language conventions, not visibility analysis: Go capitalises exported
    identifiers, everything else here treats a leading underscore as private.
    A wrong answer costs a risk grade, not correctness of the caller list.
    """
    if not symbol_name:
        return False
    if language.lower() == "go":
        return symbol_name[0].isupper()
    return not symbol_name.startswith("_")


def _risk(exported: bool, callers: int, status: str) -> str:
    """The stated rule, so a caller can disagree with it on the evidence.

    ``high``
        A deleted symbol that still has callers -- the one case that is
        actually broken rather than merely risky -- or an exported symbol with
        at least five callers.
    ``medium``
        Exported, or has any caller at all.
    ``low``
        Private with no callers found.
    """
    if status == STATUS_DELETED and callers > 0:
        return RISK_HIGH
    if exported and callers >= _HIGH_RISK_CALLERS:
        return RISK_HIGH
    if exported or callers > 0:
        return RISK_MEDIUM
    return RISK_LOW


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def _symbols_for_change(store: CodeIntelStore, change: FileChange) -> list[tuple[SymbolRow, str]]:
    """Indexed symbols this file change touches, paired with their status.

    A deleted or renamed file contributes every symbol the index still holds
    for the old path: the diff says they are gone, the index says they are
    there, and that disagreement is the finding.
    """
    if change.status == STATUS_DELETED:
        return [(row, STATUS_DELETED) for row in store.symbols(file_path=change.path)]

    touched: list[tuple[SymbolRow, str]] = []
    seen: set[str] = set()
    for row in store.symbols(file_path=change.path):
        if any(line_range.overlaps(row.start_line, row.end_line) for line_range in change.ranges):
            touched.append((row, change.status))
            seen.add(row.symbol_id)
    if change.status == STATUS_RENAMED and change.old_path:
        # The index still keys these under the old path until it catches up.
        for row in store.symbols(file_path=change.old_path):
            if row.symbol_id not in seen:
                touched.append((row, STATUS_RENAMED))
    return touched


def _callers_of(store: CodeIntelStore, name: str) -> list[tuple[str, str, str, int, str]]:
    """``(name, qualified_name, path, line, via)`` for everything referencing *name*.

    Both sources are name-keyed, so this is a superset of the truth. Call edges
    come first because they carry a caller identity; references fill in sites
    the call-graph pass did not attribute to an enclosing symbol.
    """
    found: list[tuple[str, str, str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for edge in store.call_edges(callee_name=name):
        key = (edge.caller_file_path, edge.call_line)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            (
                edge.caller_symbol_name,
                edge.caller_qualified_name,
                edge.caller_file_path,
                edge.call_line,
                "call_edge",
            )
        )
    for reference in store.references(symbol_name=name):
        key = (reference.file_path, reference.line)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            (
                reference.enclosing_symbol_name or "",
                reference.enclosing_qualified_name or "",
                reference.file_path,
                reference.line,
                "reference",
            )
        )
    return found


def analyze_changes(
    base_ref: str = "HEAD",
    paths: list[str] | None = None,
    depth: int = 1,
    limit: int = 100,
    repo_root: Path | str = ".",
) -> ChangeImpactReport:
    """Map the diff against *base_ref* onto changed symbols and their callers.

    *depth* expands the caller set by that many hops. The graph is name-keyed,
    so imprecision compounds with every hop -- depth 1 is the honest default and
    anything deeper should be read as "possibly related", not "impacted".

    *limit* bounds the returned caller list only. ``impacted_total`` reports
    what was found before truncation, so a clipped list is never mistaken for a
    complete one.

    A site that reaches two changed symbols is reported once per symbol. That is
    not double counting: "who calls what" has two answers there, and collapsing
    them would silently drop one.
    """
    root = Path(repo_root).resolve()
    depth = max(1, int(depth))
    limit = max(1, int(limit))
    diff_ref, changes = collect_changes(root, base_ref=base_ref, paths=paths)

    with CodeIntelStore(root) as store:
        index_version = store.engine_state("index_version")
        indexed_paths = {row.file_path for row in store.files()}

        pending: list[tuple[SymbolRow, str]] = []
        unindexed: list[str] = []
        for change in changes:
            touched = _symbols_for_change(store, change)
            if not touched and change.path not in indexed_paths and change.status != STATUS_DELETED:
                unindexed.append(change.path)
            pending.extend(touched)

        # Walk the name-keyed graph outwards, one hop at a time. `visited`
        # holds names already expanded so a cycle terminates; `sites` dedupes
        # by location so the same call reached two ways is reported once.
        # Keyed by the changed symbol as well as the location: one site can be a
        # direct caller of one changed symbol and a two-hop caller of another,
        # and a global key would let whichever symbol arrived first fix `depth`
        # and `changed_symbol` for both -- reporting a direct caller as remote
        # and dropping it from the symbol it actually calls.
        impacted: list[ImpactedCaller] = []
        sites: set[tuple[str, str, int]] = set()
        caller_counts: dict[str, int] = {}
        test_caller_counts: dict[str, int] = {}

        for row, _status in pending:
            frontier = {row.symbol_name}
            visited: set[str] = set()
            direct = 0
            direct_tests = 0
            for hop in range(1, depth + 1):
                next_frontier: set[str] = set()
                for name in sorted(frontier):
                    if name in visited or not name:
                        continue
                    visited.add(name)
                    for caller_name, qualified, path, line, via in _callers_of(store, name):
                        if path == row.file_path and row.start_line <= line <= row.end_line:
                            continue  # the symbol's own body, not an external caller
                        is_test = _is_test_path(path)
                        if hop == 1:
                            direct += 1
                            if is_test:
                                direct_tests += 1
                        site = (row.qualified_name, path, line)
                        if site not in sites:
                            sites.add(site)
                            impacted.append(
                                ImpactedCaller(
                                    name=caller_name,
                                    qualified_name=qualified,
                                    file_path=path,
                                    line=line,
                                    depth=hop,
                                    via=via,
                                    match_kind=MATCH_NAME,
                                    is_test=is_test,
                                    changed_symbol=row.qualified_name,
                                )
                            )
                        if caller_name:
                            next_frontier.add(caller_name)
                frontier = next_frontier - visited
                if not frontier:
                    break
            caller_counts[row.symbol_id] = direct
            test_caller_counts[row.symbol_id] = direct_tests

        changed_symbols = tuple(
            ChangedSymbol(
                symbol_id=row.symbol_id,
                symbol_name=row.symbol_name,
                qualified_name=row.qualified_name,
                kind=row.kind,
                file_path=row.file_path,
                start_line=row.start_line,
                end_line=row.end_line,
                language=row.language,
                status=status,
                exported=_is_exported(row.symbol_name, row.language),
                callers=caller_counts.get(row.symbol_id, 0),
                test_callers=test_caller_counts.get(row.symbol_id, 0),
                risk=_risk(
                    _is_exported(row.symbol_name, row.language),
                    caller_counts.get(row.symbol_id, 0),
                    status,
                ),
            )
            for row, status in pending
        )

    impacted.sort(key=lambda item: (item.depth, item.file_path, item.line))
    return ChangeImpactReport(
        repo_root=str(root),
        base_ref=base_ref,
        diff_ref=diff_ref,
        engine_index_version=index_version,
        depth=depth,
        match_kind=MATCH_NAME,
        files=tuple(changes),
        changed_symbols=changed_symbols,
        impacted=tuple(impacted[:limit]),
        impacted_total=len(impacted),
        truncated=len(impacted) > limit,
        unindexed_paths=tuple(unindexed),
    )
