"""Which paths the code index is allowed to take.

Two entry points write ``files`` rows. ``CodeContextEngine._index_repo_unsafe``
builds its candidate set with the whole-repo source-file scan, so that scan's
rules bound it. ``CodeContextEngine._reindex_files`` re-extracts whatever paths
an edit just touched, and applied no selection rules at all: an edit to a file
the scan would never select -- one inside a nested git worktree under a
gitignored ``.claude/``, say -- put that file's symbols into the parent
checkout's index, where ``relations``, ``code_query`` and ``code_changes`` then
returned the same symbol once per copy.

The rules live here so both entry points and
:mod:`lemoncrow.infra.code_intel.coverage` apply one set, and a path coverage
reports ``excluded`` has no rows in the index to find. The scan itself calls
*down* into this module, which is why nothing here imports from
:mod:`lemoncrow.pro` at any scope -- the dependency runs one way.

Fail-open by design: a rule that cannot be evaluated -- git missing, the
directory not a repository, ``pathspec`` not installed -- admits the path, which
is what the scan's own non-git fallback (``_iter_glob_source_files``) does.
"""

from __future__ import annotations

import fnmatch
import subprocess
from collections.abc import Container, Iterable, Sequence
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.languages import LANGUAGES, language_for_path

__all__ = [
    "EXCLUSION_RULES",
    "FREE_TIER_MAX_FILES",
    "REASON_SOURCE_FILE_SCAN",
    "RULE_FREE_TIER_FILE_CAP",
    "RULE_GIT_IGNORE",
    "RULE_LEMONCROW_IGNORE",
    "RULE_SKIPPED_DIRECTORY",
    "RULE_SOURCE_FILE_SCAN",
    "RULE_UNRECOGNISED_FILE_TYPE",
    "SOURCE_FILE_PATTERNS",
    "exclusion_rule",
    "free_tier_selection",
    "git_ignored",
    "indexable_paths",
    "load_lemoncrow_ignore_spec",
    "run_git",
    "scan_selects",
    "should_skip_path",
    "should_skip_relative_path",
    "source_file_patterns",
]

RULE_SKIPPED_DIRECTORY = "skipped-directory"
RULE_LEMONCROW_IGNORE = "lemoncrow-ignore"
RULE_GIT_IGNORE = "git-ignore"
RULE_UNRECOGNISED_FILE_TYPE = "unrecognised-file-type"
RULE_FREE_TIER_FILE_CAP = "free-tier-file-cap"
RULE_SOURCE_FILE_SCAN = "source-file-scan"

REASON_SOURCE_FILE_SCAN = "not selected by the index's source-file scan"

#: The indexer's file-selection rules, first match wins. The free-tier cap only
#: ever applies to a path the scan selected, and every other rule only to a path
#: it did not. ``source-file-scan`` closes the ladder: the scan's own glob gate
#: does not take the path, and no narrower rule says why (an extension in the
#: wrong case, an untracked file inside a submodule).
EXCLUSION_RULES: tuple[str, ...] = (
    RULE_SKIPPED_DIRECTORY,
    RULE_LEMONCROW_IGNORE,
    RULE_GIT_IGNORE,
    RULE_UNRECOGNISED_FILE_TYPE,
    RULE_FREE_TIER_FILE_CAP,
    RULE_SOURCE_FILE_SCAN,
)

_SKIP_PARTS = {
    ".git",
    ".lemoncrow",
    ".bench-work",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    # Raw data/results dumps (fixtures, benchmark output, run logs) are never
    # source code, but a directory literally named this often holds thousands
    # of JSON/CSV files that DO match a data-language extension -- each one
    # gets symbol-extracted (every JSON key becomes a "variable" symbol),
    # ballooning index/embedding time and diluting search with noise.
    "results",
    "data",
}

_LEMONCROW_IGNORE_PATH = Path(".lemoncrow") / ".ignore"


def should_skip_relative_path(path: str) -> bool:
    return any(part in _SKIP_PARTS for part in Path(path).parts)


def should_skip_path(path: Path, *, repo_root: Path | None = None) -> bool:
    try:
        rel = path.relative_to(repo_root) if repo_root is not None else path
    except ValueError:
        rel = path
    return should_skip_relative_path(rel.as_posix())


def _build_source_file_patterns() -> tuple[str, ...]:
    seen: set[str] = set()
    patterns: list[str] = []
    for lang in LANGUAGES:
        for filename in sorted(lang.filenames):
            for pattern in (filename, f"**/{filename}"):
                if pattern not in seen:
                    seen.add(pattern)
                    patterns.append(pattern)
        for ext in sorted(lang.extensions, key=lambda e: (-len(e), e)):
            pattern = f"**/*{ext}"
            if pattern not in seen:
                seen.add(pattern)
                patterns.append(pattern)
    return tuple(patterns)


#: Built once at import: the language registry is frozen, and the rule ladder
#: consults these patterns on every candidate path.
SOURCE_FILE_PATTERNS: tuple[str, ...] = _build_source_file_patterns()


def source_file_patterns() -> list[str]:
    """Return glob patterns for all languages in the canonical registry."""
    return list(SOURCE_FILE_PATTERNS)


def scan_selects(rel: str, patterns: Sequence[str]) -> bool:
    """True when the scan's glob gate takes the repo-relative path *rel*.

    Matched against both the full repo-relative path and the bare basename:
    the default patterns are recursive globs like ``**/*.py`` and ``fnmatch``
    (unlike pathlib) does not treat ``**`` as "zero or more dirs", so
    ``fnmatch("run.py", "**/*.py")`` is False and root-level source files would
    be silently dropped from the git-visible index.

    Case-sensitive on a POSIX filesystem, which is the rule's only surprising
    consequence: ``src/SHOUT.PY`` resolves to a language by suffix and is still
    not a source file to the index, because the scan never selects it.
    """
    name = rel.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern.rsplit("/", 1)[-1]) for pattern in patterns
    )


#: Free-tier repo-size cap for the context engine (``context_engine`` is a Pro
#: feature at scale -- see licensing/features.py). Generous on purpose: this is
#: well past a typical solo/small-team repo, so Free stays "genuinely useful";
#: it's a real ceiling only for large monorepos, which is exactly what Pro's
#: uncapped large-repo indexing is for. The engine's index run and the coverage
#: verdict that predicts it both read it off this module at call time, so they
#: always cap at the same number.
FREE_TIER_MAX_FILES = 2_500


def free_tier_selection(files: Sequence[Path], *, cap: int) -> list[Path]:
    """The first *cap* paths of the sorted scan, or all of *files* when under it.

    One definition of the Free-tier cap, so an index run and the coverage
    verdict that has to predict it cannot drift apart. Callers pass
    :data:`FREE_TIER_MAX_FILES` and own the licensing check that decides whether
    it applies at all.
    """
    if len(files) <= cap:
        return list(files)
    return sorted(files)[:cap]


def load_lemoncrow_ignore_spec(repo_root: Path) -> Any | None:
    """Load ``.lemoncrow/.ignore`` and return a gitignore-syntax pathspec.

    Acts as a union with ``.gitignore``: git already excludes gitignored paths
    from ``git ls-files``, so this only needs to add the *extra* patterns from
    ``.lemoncrow/.ignore`` (e.g. tracked data files a user wants kept out of
    the index without untracking them). Returns None when the file is absent
    or pathspec can't be loaded -- indexing must never hard-fail on this.
    """
    ignore_path = repo_root / _LEMONCROW_IGNORE_PATH
    if not ignore_path.is_file():
        return None
    try:
        import pathspec
    except ImportError:
        return None
    try:
        lines = ignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.PathSpec.from_lines("gitignore", lines)


def run_git(root: Path, *args: str, stdin: str | None = None) -> str | None:
    """Run git in *root*, returning stdout or ``None`` when git cannot answer.

    Fail-open by design: a non-git directory or a missing git binary must
    degrade the rule that asked, not break the caller.
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


def git_ignored(root: Path, candidates: Sequence[str]) -> frozenset[str]:
    """The repo-relative *candidates* git excludes from this checkout.

    One call covers the whole batch. ``check-ignore`` consults the index first,
    so a tracked file matching an ignore pattern is not reported -- the same
    answer ``git ls-files`` gives the scan.
    """
    if not candidates:
        return frozenset()
    output = run_git(root, "check-ignore", "--stdin", "-z", stdin="\0".join(candidates))
    if output is None:
        return frozenset()
    return frozenset(entry for entry in output.split("\0") if entry)


def exclusion_rule(
    rel: str,
    *,
    ignore_spec: Any | None,
    ignored: Container[str],
    patterns: Sequence[str] | None = None,
) -> tuple[str, str] | None:
    """``(rule, reason)`` for the first rule that passes *rel* over, else ``None``.

    The order is fixed and the first match wins, so a git-ignored file under
    ``data/`` names the skipped directory. The last rung is the scan's own glob
    gate, which is what makes ``None`` mean "every rule the index applies admits
    this path" rather than "no narrower rule objects": without it a path the
    scan drops on its patterns alone -- ``src/SHOUT.PY`` -- was admitted by the
    incremental entry point while coverage called it ``excluded``.

    *patterns* defaults to the scan's own :data:`SOURCE_FILE_PATTERNS`; pass the
    list once when classifying a batch.
    """
    for part in Path(rel).parts:
        if should_skip_relative_path(part):
            return RULE_SKIPPED_DIRECTORY, f"skipped directory: {part}"
    if ignore_spec is not None and ignore_spec.match_file(rel):
        return RULE_LEMONCROW_IGNORE, ".lemoncrow/.ignore"
    if rel in ignored:
        return RULE_GIT_IGNORE, "git-ignored"
    if language_for_path(rel) is None:
        return RULE_UNRECOGNISED_FILE_TYPE, "unrecognised file type"
    if not scan_selects(rel, SOURCE_FILE_PATTERNS if patterns is None else patterns):
        return RULE_SOURCE_FILE_SCAN, REASON_SOURCE_FILE_SCAN
    return None


def indexable_paths(repo_root: Path, paths: Iterable[Path]) -> list[Path]:
    """The subset of *paths* the indexer's file-selection rules admit, in order.

    For the incremental entry point, where the candidates are the handful of
    files an edit touched rather than a whole scan: the ``.lemoncrow/.ignore``
    spec is loaded once and git is asked once for the batch, so the cost is one
    subprocess per reindex, not one per file. A path outside *repo_root* is
    dropped -- no scan of this checkout would ever list it.
    """
    root = repo_root.resolve()
    pairs: list[tuple[Path, str]] = []
    for path in paths:
        try:
            pairs.append((path, path.resolve().relative_to(root).as_posix()))
        except (OSError, ValueError):
            continue
    if not pairs:
        return []
    ignore_spec = load_lemoncrow_ignore_spec(root)
    ignored = git_ignored(root, [rel for _path, rel in pairs])
    patterns = source_file_patterns()
    return [
        path
        for path, rel in pairs
        if exclusion_rule(rel, ignore_spec=ignore_spec, ignored=ignored, patterns=patterns) is None
    ]
