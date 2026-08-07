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
    On disk, in a language the index supports, and simply not there.
``excluded``
    On disk but outside what the indexer takes: git-ignored, or a file type the
    language registry does not recognise.
``unparsed``
    In ``files`` with zero rows in ``symbols`` -- indexed as a file, but no
    symbols were extracted from it.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.completeness import OBJECTIVE_EXHAUSTIVE
from lemoncrow.infra.code_intel.languages import language_for_path
from lemoncrow.infra.code_intel.store import CodeIntelStore, FileRow

__all__ = [
    "STATES",
    "CoverageReport",
    "PathCoverage",
    "check_coverage",
]

STATES: tuple[str, ...] = ("indexed", "stale", "missing", "excluded", "unparsed")

# The engine's own ignore rules live in the closed indexer. This module states
# which rules IT applied instead of pretending to know the engine's, and reports
# that under `exclusion_source` so a caller can weigh the answer.
_EXCLUSION_SOURCE = "git-ignore + unrecognised-file-type"

_HASH_READ_CHUNK = 1 << 20


@dataclass(frozen=True)
class PathCoverage:
    """One path's index state, with the reason it landed there."""

    path: str
    state: str
    reason: str
    language: str | None
    symbols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "state": self.state,
            "reason": self.reason,
            "language": self.language,
            "symbols": self.symbols,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Per-path states plus the engine generation they were judged against."""

    repo_root: str
    engine_index_version: int
    exclusion_source: str
    totals: dict[str, int]
    paths: tuple[PathCoverage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            # An audit reports on every path it was asked about, or it is not
            # an audit.
            "objective": OBJECTIVE_EXHAUSTIVE,
            "repo_root": self.repo_root,
            "engine_index_version": self.engine_index_version,
            "exclusion_source": self.exclusion_source,
            "totals": dict(self.totals),
            "paths": [entry.to_dict() for entry in self.paths],
        }


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_READ_CHUNK):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _git(root: Path, *args: str, stdin: str | None = None) -> str | None:
    """Run git in *root*, returning stdout or ``None`` when git cannot answer.

    Fail-open by design: a non-git directory or a missing git binary must
    degrade the exclusion rule, not break the whole report.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # check-ignore exits 1 when nothing matched, which is a real answer.
    if completed.returncode not in (0, 1):
        return None
    return completed.stdout


def _tracked_files(root: Path) -> set[str]:
    output = _git(root, "ls-files", "-z")
    if output is None:
        return set()
    return {entry for entry in output.split("\0") if entry}


def _ignored_files(root: Path, candidates: list[str]) -> set[str]:
    if not candidates:
        return set()
    output = _git(root, "check-ignore", "--stdin", "-z", stdin="\0".join(candidates))
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
    """
    root = Path(repo_root).expanduser().resolve()

    with CodeIntelStore(root) as store:
        snapshot = store.snapshot()
        indexed: dict[str, FileRow] = {row.file_path: row for row in store.files()}
        symbol_counts = store.symbol_counts_by_file()

    if paths is None:
        candidates = sorted(set(indexed) | _tracked_files(root))
        ignored: set[str] = set()
    else:
        candidates = sorted({_relative(root, raw) for raw in paths})
        on_disk_unindexed = [rel for rel in candidates if rel not in indexed and (root / rel).exists()]
        ignored = _ignored_files(root, on_disk_unindexed)

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
        elif rel in ignored:
            entry = PathCoverage(rel, "excluded", "git-ignored", language_name, 0)
        elif language is None:
            entry = PathCoverage(rel, "excluded", "unrecognised file type", None, 0)
        else:
            entry = PathCoverage(rel, "missing", "on disk, supported language, not indexed", language_name, 0)

        entries.append(entry)
        totals[entry.state] += 1

    return CoverageReport(
        repo_root=str(root),
        engine_index_version=snapshot.index_version,
        exclusion_source=_EXCLUSION_SOURCE,
        totals=totals,
        paths=tuple(entries),
    )
