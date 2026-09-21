"""F3 -- is this path actually in the code index?

The primitive that makes "I searched and found nothing" auditable. Without it,
an empty search result is indistinguishable from an unindexed file, and every
other code-intel feature inherits that ambiguity.

Five states, per path:

``indexed``
    In ``files`` and matching what is on disk.
``stale``
    In ``files`` but the bytes on disk have moved on -- including the case where
    the file has been deleted and the index has not caught up.
``missing``
    Not in ``files``: either not on disk at all, or on disk and selected by the
    indexer's file scan but absent from the last index run.
``excluded``
    On disk but passed over by one of the indexer's file-selection rules, which
    the verdict names in ``rule`` (see :data:`EXCLUSION_RULES`).
``unparsed``
    In ``files`` with zero rows in ``symbols`` -- indexed as a file, but no
    symbols were extracted from it.

Excluded and missing are decided by the indexer's own scan rather than a guess
at it: an on-disk path the index does not hold is ``missing`` exactly when
:func:`~lemoncrow.pro.capabilities.repo_map.graph.iter_source_files` selects it
and the free-tier cap would keep it. One rule cannot be read back:
``exclude_globs`` passed to an index run are not persisted, so a path they kept
out reports ``missing``, with a reason saying an index-time exclude may apply --
or ``excluded`` under ``free-tier-file-cap`` when that cap is engaged too, since
the cap is counted here over the whole scan where the index run counted it over
what the excludes left.

Both directions hold, because the index applies these same rules on the way in
(:mod:`lemoncrow.infra.code_intel.inclusion`, on each of its two entry points):
a path this report calls ``excluded`` has no rows in the index to find.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel import inclusion
from lemoncrow.infra.code_intel.completeness import OBJECTIVE_EXHAUSTIVE
from lemoncrow.infra.code_intel.freshness import require_ready
from lemoncrow.infra.code_intel.inclusion import (
    EXCLUSION_RULES,
    REASON_SOURCE_FILE_SCAN,
    RULE_FREE_TIER_FILE_CAP,
    RULE_SOURCE_FILE_SCAN,
    exclusion_rule,
    free_tier_selection,
    git_ignored,
    git_tracked,
    load_lemoncrow_ignore_spec,
    run_git,
    source_file_patterns,
)
from lemoncrow.infra.code_intel.languages import language_for_path
from lemoncrow.infra.code_intel.store import CodeIntelStore, FileRow

__all__ = [
    "EXCLUSION_RULES",
    "STATES",
    "CoverageReport",
    "PathCoverage",
    "check_coverage",
]

STATES: tuple[str, ...] = ("indexed", "stale", "missing", "excluded", "unparsed")

# `exclude_globs` given to an index run are not persisted, so a path the scan
# selects but the index does not hold may be excluded that way or not yet indexed.
_NOT_IN_LAST_RUN = "not in the last index run (an index-time exclude may apply)"

_HASH_READ_CHUNK = 1 << 20


@dataclass(frozen=True)
class PathCoverage:
    """One path's index state, with the reason it landed there."""

    path: str
    state: str
    reason: str
    language: str | None
    symbols: int
    #: The :data:`EXCLUSION_RULES` entry behind an ``excluded`` verdict, else ``None``.
    rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "state": self.state,
            "reason": self.reason,
            "language": self.language,
            "symbols": self.symbols,
        }
        if self.rule is not None:
            payload["rule"] = self.rule
        return payload


@dataclass(frozen=True)
class CoverageReport:
    """Per-path states plus the engine generation they were judged against."""

    repo_root: str
    engine_index_version: int
    #: The rules behind this report's ``excluded`` verdicts, in
    #: :data:`EXCLUSION_RULES` order and joined with `` + ``; empty when no path
    #: was excluded. A summary of the per-path ``rule`` fields, never a claim
    #: about rules no returned path used.
    exclusion_source: str
    totals: dict[str, int]
    paths: tuple[PathCoverage, ...]
    exclusion_rules: tuple[str, ...] = EXCLUSION_RULES

    def to_dict(self) -> dict[str, Any]:
        return {
            # An audit reports on every path it was asked about, or it is not
            # an audit.
            "objective": OBJECTIVE_EXHAUSTIVE,
            "repo_root": self.repo_root,
            "engine_index_version": self.engine_index_version,
            "exclusion_source": self.exclusion_source,
            "exclusion_rules": list(self.exclusion_rules),
            "totals": dict(self.totals),
            "paths": [entry.to_dict() for entry in self.paths],
        }


@dataclass(frozen=True)
class _IndexSelection:
    """What the indexer's file scan takes from this checkout as it stands."""

    #: Repo-relative paths :func:`iter_source_files` selects.
    selected: frozenset[str]
    #: The subset the free-tier cap keeps; all of *selected* when uncapped.
    kept: frozenset[str]
    #: The ``.lemoncrow/.ignore`` spec the scan applied, or ``None``.
    ignore_spec: Any | None


_NO_SELECTION = _IndexSelection(selected=frozenset(), kept=frozenset(), ignore_spec=None)


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_READ_CHUNK):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _tracked_files(root: Path) -> set[str]:
    output = run_git(root, "ls-files", "-z")
    if output is None:
        return set()
    return {entry for entry in output.split("\0") if entry}


def _relative(root: Path, raw: str) -> str:
    """Normalise *raw* to a repo-relative POSIX path, as stored in ``files``."""
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root)
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def _relative_set(root: Path, files: list[Path]) -> frozenset[str]:
    relative: set[str] = set()
    for path in files:
        try:
            relative.add(path.relative_to(root).as_posix())
        except ValueError:
            continue  # a symlink resolving outside the checkout
    return frozenset(relative)


def _index_selection(root: Path) -> _IndexSelection:
    """Run the indexer's own file scan, then its free-tier cap.

    Mirrors ``CodeContextEngine._index_repo_unsafe`` with no ``exclude_globs``:
    the same :func:`iter_source_files` call, then, without the
    ``context_engine`` feature, the same :func:`free_tier_selection` over it.
    Imported lazily, because the scan lists and pattern-matches every
    git-visible file and only runs when a queried path is not indexed.
    """
    from lemoncrow.core.capabilities import licensing
    from lemoncrow.pro.capabilities.repo_map.graph import iter_source_files

    files = iter_source_files(root)
    kept = files
    if not licensing.has_feature("context_engine"):
        kept = free_tier_selection(files, cap=inclusion.FREE_TIER_MAX_FILES)
    return _IndexSelection(
        selected=_relative_set(root, files),
        kept=_relative_set(root, kept),
        ignore_spec=load_lemoncrow_ignore_spec(root),
    )


def _exclusion(
    rel: str,
    selection: _IndexSelection,
    ignored: frozenset[str],
    tracked: frozenset[str],
    patterns: Sequence[str],
) -> tuple[str, str]:
    """``(rule, reason)`` for an on-disk path the indexer's scan passed over.

    The rules the index itself applies (:func:`exclusion_rule`) name all of them
    the index can name, its own glob gate included. The fallback is what is left
    when the scan passed a path over for a reason no rule sees -- an untracked
    file inside a submodule, which ``git ls-files --others`` does not recurse.
    """
    return exclusion_rule(
        rel, ignore_spec=selection.ignore_spec, ignored=ignored, tracked=tracked, patterns=patterns
    ) or (
        RULE_SOURCE_FILE_SCAN,
        REASON_SOURCE_FILE_SCAN,
    )


def _exclusion_source(entries: Sequence[PathCoverage]) -> str:
    fired = {entry.rule for entry in entries if entry.rule is not None}
    return " + ".join(rule for rule in EXCLUSION_RULES if rule in fired)


def _disk_matches(root: Path, rel: str, row: FileRow) -> bool:
    """True when the indexed row still describes what is on disk.

    Size and mtime are the fast path; the content hash is only computed when
    mtime disagrees, so a touched-but-unchanged file (a checkout, a rebase)
    does not get reported as stale.
    """
    absolute = root / rel
    try:
        stat = absolute.stat()
    except OSError:
        return False
    if stat.st_size != row.size_bytes:
        return False
    if row.mtime_ns and stat.st_mtime_ns == row.mtime_ns:
        return True
    return _sha256(absolute) == row.content_hash


def check_coverage(paths: list[str] | None = None, repo_root: Path | str = ".") -> CoverageReport:
    """Classify *paths* (or the whole repo) against the code index.

    With no *paths*, the candidate set is every git-tracked file plus everything
    already in the index -- not a filesystem walk, which would drag in build
    output and virtualenvs the indexer never looked at.

    An on-disk path the index does not hold is judged against the indexer's own
    file scan, run once and only when such a path is asked about. It is
    ``excluded``, naming the first rule that passed it over, or ``missing`` when
    the scan selects it. Index-time ``exclude_globs`` are not persisted, so a
    path one of them kept out cannot be told apart from a path not yet indexed:
    it reports ``missing``, with a reason saying an index-time exclude may apply,
    or ``free-tier-file-cap`` when that cap is engaged as well -- the cap counts
    the whole scan here, where the index run counted only what the excludes left.

    Raises :class:`~lemoncrow.infra.code_intel.freshness.IndexRebuilding` while
    the index is mid-write, and
    :class:`~lemoncrow.infra.code_intel.store.CodeIntelUnavailable` when it is
    absent: verdicts judged against a torn index report real files as missing,
    and an absent one has nothing to judge against.
    """
    root = Path(repo_root).expanduser().resolve()
    require_ready(root)

    with CodeIntelStore(root) as store:
        snapshot = store.snapshot()
        indexed: dict[str, FileRow] = {row.file_path: row for row in store.files()}
        symbol_counts = store.symbol_counts_by_file()

    if paths is None:
        candidates = sorted(set(indexed) | _tracked_files(root))
    else:
        candidates = sorted({_relative(root, raw) for raw in paths})

    unindexed_on_disk = [rel for rel in candidates if rel not in indexed and (root / rel).exists()]
    selection = _index_selection(root) if unindexed_on_disk else _NO_SELECTION
    passed_over = [rel for rel in unindexed_on_disk if rel not in selection.selected]
    ignored = git_ignored(root, passed_over)
    tracked = git_tracked(root, passed_over)
    patterns = source_file_patterns() if unindexed_on_disk else []

    entries: list[PathCoverage] = []
    totals: dict[str, int] = dict.fromkeys(STATES, 0)

    for rel in candidates:
        language = language_for_path(rel)
        language_name = language.name if language is not None else None
        row = indexed.get(rel)
        exists = (root / rel).exists()

        if row is not None:
            symbols = symbol_counts.get(rel, 0)
            if not exists:
                entry = PathCoverage(rel, "stale", "indexed but deleted from disk", row.language, symbols)
            elif not _disk_matches(root, rel, row):
                entry = PathCoverage(rel, "stale", "content changed since indexing", row.language, symbols)
            elif symbols == 0:
                entry = PathCoverage(
                    rel,
                    "unparsed",
                    "indexed as a file but no symbols were extracted",
                    row.language,
                    0,
                )
            else:
                entry = PathCoverage(rel, "indexed", "up to date", row.language, symbols)
        elif not exists:
            entry = PathCoverage(rel, "missing", "not on disk and not indexed", language_name, 0)
        elif rel not in selection.selected:
            rule, reason = _exclusion(rel, selection, ignored, tracked, patterns)
            entry = PathCoverage(rel, "excluded", reason, language_name, 0, rule)
        elif rel not in selection.kept:
            entry = PathCoverage(rel, "excluded", "free-tier file cap", language_name, 0, RULE_FREE_TIER_FILE_CAP)
        else:
            entry = PathCoverage(rel, "missing", _NOT_IN_LAST_RUN, language_name, 0)

        entries.append(entry)
        totals[entry.state] += 1

    return CoverageReport(
        repo_root=str(root),
        engine_index_version=snapshot.index_version,
        exclusion_source=_exclusion_source(entries),
        totals=totals,
        paths=tuple(entries),
    )
