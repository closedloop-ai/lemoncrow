"""Which paths the code index is allowed to take.

Two entry points write ``files`` rows. ``CodeContextEngine._index_repo_unsafe``
builds its candidate set with
:func:`~lemoncrow.pro.capabilities.repo_map.graph.iter_source_files`, so that
scan's rules bound it. ``CodeContextEngine._reindex_files`` re-extracts whatever
paths an edit just touched, and applied no selection rules at all: an edit to a
file the scan would never select -- one inside a nested git worktree under a
gitignored ``.claude/``, say -- put that file's symbols into the parent
checkout's index, where ``relations``, ``code_query`` and ``code_changes`` then
returned the same symbol once per copy.

The rules live here so both entry points and
:mod:`lemoncrow.infra.code_intel.coverage` apply one set, and a path coverage
reports ``excluded`` has no rows in the index to find.

Fail-open by design: a rule that cannot be evaluated -- git missing, the
directory not a repository, ``pathspec`` not installed -- admits the path, which
is what the scan's own non-git fallback (``_iter_glob_source_files``) does.
"""

from __future__ import annotations

import subprocess
from collections.abc import Container, Iterable, Sequence
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.languages import language_for_path

__all__ = [
    "EXCLUSION_RULES",
    "RULE_FREE_TIER_FILE_CAP",
    "RULE_GIT_IGNORE",
    "RULE_LEMONCROW_IGNORE",
    "RULE_SKIPPED_DIRECTORY",
    "RULE_SOURCE_FILE_SCAN",
    "RULE_UNRECOGNISED_FILE_TYPE",
    "exclusion_rule",
    "git_ignored",
    "indexable_paths",
    "run_git",
]

RULE_SKIPPED_DIRECTORY = "skipped-directory"
RULE_LEMONCROW_IGNORE = "lemoncrow-ignore"
RULE_GIT_IGNORE = "git-ignore"
RULE_UNRECOGNISED_FILE_TYPE = "unrecognised-file-type"
RULE_FREE_TIER_FILE_CAP = "free-tier-file-cap"
RULE_SOURCE_FILE_SCAN = "source-file-scan"

#: The indexer's file-selection rules, first match wins. The free-tier cap only
#: ever applies to a path the scan selected, and every other rule only to a path
#: it did not. ``source-file-scan`` is the remainder a coverage verdict needs:
#: the scan passed the path over and no narrower rule says why (an extension in
#: the wrong case, an untracked file inside a submodule).
EXCLUSION_RULES: tuple[str, ...] = (
    RULE_SKIPPED_DIRECTORY,
    RULE_LEMONCROW_IGNORE,
    RULE_GIT_IGNORE,
    RULE_UNRECOGNISED_FILE_TYPE,
    RULE_FREE_TIER_FILE_CAP,
    RULE_SOURCE_FILE_SCAN,
)


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


def exclusion_rule(rel: str, *, ignore_spec: Any | None, ignored: Container[str]) -> tuple[str, str] | None:
    """``(rule, reason)`` for the first rule that passes *rel* over, else ``None``.

    The order is fixed and the first match wins, so a git-ignored file under
    ``data/`` names the skipped directory. ``None`` means no rule here excludes
    the path: the index takes it, and a coverage verdict falls through to its
    own remainder.
    """
    from lemoncrow.pro.capabilities.repo_map.graph import should_skip_relative_path

    for part in Path(rel).parts:
        if should_skip_relative_path(part):
            return RULE_SKIPPED_DIRECTORY, f"skipped directory: {part}"
    if ignore_spec is not None and ignore_spec.match_file(rel):
        return RULE_LEMONCROW_IGNORE, ".lemoncrow/.ignore"
    if rel in ignored:
        return RULE_GIT_IGNORE, "git-ignored"
    if language_for_path(rel) is None:
        return RULE_UNRECOGNISED_FILE_TYPE, "unrecognised file type"
    return None


def indexable_paths(repo_root: Path, paths: Iterable[Path]) -> list[Path]:
    """The subset of *paths* the indexer's file-selection rules admit, in order.

    For the incremental entry point, where the candidates are the handful of
    files an edit touched rather than a whole scan: the ``.lemoncrow/.ignore``
    spec is loaded once and git is asked once for the batch, so the cost is one
    subprocess per reindex, not one per file. A path outside *repo_root* is
    dropped -- no scan of this checkout would ever list it.
    """
    from lemoncrow.pro.capabilities.repo_map.graph import load_lemoncrow_ignore_spec

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
    return [path for path, rel in pairs if exclusion_rule(rel, ignore_spec=ignore_spec, ignored=ignored) is None]
