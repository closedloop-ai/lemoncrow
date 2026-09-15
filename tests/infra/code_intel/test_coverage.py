"""F3: index coverage -- absent must be distinguishable from not-indexed."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lemoncrow.infra.code_intel.coverage import EXCLUSION_RULES, STATES, CoverageReport, check_coverage
from lemoncrow.infra.code_intel.freshness import IndexRebuilding

WorkspaceFactory = Callable[..., Path]


def _write(root: Path, rel: str, body: str) -> dict[str, Any]:
    """Write *rel* and return the ``files`` row that would index it faithfully."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    raw = body.encode("utf-8")
    return {
        "file_path": rel,
        "content_hash": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mtime_ns": target.stat().st_mtime_ns,
    }


def _state_of(report: CoverageReport, path: str) -> str:
    for entry in report.paths:
        if entry.path == path:
            return entry.state
    raise AssertionError(f"{path} not in report: {[entry.path for entry in report.paths]}")


def _git_init(root: Path, *add: str) -> None:
    try:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True, timeout=30)
        if add:
            subprocess.run(["git", "add", *add], cwd=root, check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git is present in CI
        pytest.skip("git unavailable")


def test_all_five_states_are_reported(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    indexed = _write(workspace_root, "src/indexed.py", "def alpha():\n    return 1\n")
    stale = _write(workspace_root, "src/stale.py", "def beta():\n    return 2\n")
    unparsed = _write(workspace_root, "src/unparsed.py", "# no symbols here\n")
    _write(workspace_root, "src/missing.py", "def gamma():\n    return 3\n")
    _write(workspace_root, "assets/logo.svg", "<svg/>")

    root = make_workspace(
        files=[indexed, stale, unparsed],
        symbols=[
            {"file_path": "src/indexed.py", "symbol_name": "alpha"},
            {"file_path": "src/stale.py", "symbol_name": "beta"},
        ],
        index_version=8,
    )
    assert root == workspace_root

    # Mutate one file *after* it was indexed -- the definition of stale.
    (root / "src/stale.py").write_text("def beta():\n    return 999\n", encoding="utf-8")

    report = check_coverage(
        paths=["src/indexed.py", "src/stale.py", "src/unparsed.py", "src/missing.py", "assets/logo.svg"],
        repo_root=root,
    )

    assert _state_of(report, "src/indexed.py") == "indexed"
    assert _state_of(report, "src/stale.py") == "stale"
    assert _state_of(report, "src/unparsed.py") == "unparsed"
    assert _state_of(report, "src/missing.py") == "missing"
    assert _state_of(report, "assets/logo.svg") == "excluded"

    assert report.engine_index_version == 8
    assert set(report.totals) == set(STATES)
    assert sum(report.totals.values()) == 5


def test_a_touched_but_unchanged_file_is_not_stale(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    """A checkout or rebase rewrites mtime without changing bytes.

    Reporting that as stale would make the whole signal noise, so mtime
    disagreement falls through to a content hash rather than deciding.
    """
    row = _write(workspace_root, "src/a.py", "def alpha():\n    return 1\n")
    root = make_workspace(files=[row], symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}])

    bumped = int(row["mtime_ns"]) + 10**9
    os.utime(root / "src/a.py", ns=(bumped, bumped))

    assert _state_of(check_coverage(paths=["src/a.py"], repo_root=root), "src/a.py") == "indexed"


def test_a_deleted_but_still_indexed_file_is_stale(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    row = _write(workspace_root, "src/gone.py", "def alpha():\n    return 1\n")
    root = make_workspace(files=[row], symbols=[{"file_path": "src/gone.py", "symbol_name": "alpha"}])
    (root / "src/gone.py").unlink()

    report = check_coverage(paths=["src/gone.py"], repo_root=root)
    assert _state_of(report, "src/gone.py") == "stale"
    assert report.paths[0].reason == "indexed but deleted from disk"


def test_size_change_alone_is_enough_to_be_stale(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    row = _write(workspace_root, "src/a.py", "def alpha():\n    return 1\n")
    root = make_workspace(files=[row], symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}])

    # Restore the recorded mtime so only the length differs.
    (root / "src/a.py").write_text("def alpha():\n    return 1  # longer\n", encoding="utf-8")
    recorded = int(row["mtime_ns"])
    os.utime(root / "src/a.py", ns=(recorded, recorded))

    assert _state_of(check_coverage(paths=["src/a.py"], repo_root=root), "src/a.py") == "stale"


def test_absolute_paths_are_normalised_to_repo_relative(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    row = _write(workspace_root, "src/a.py", "def alpha():\n    return 1\n")
    root = make_workspace(files=[row], symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}])

    report = check_coverage(paths=[str(root / "src/a.py")], repo_root=root)
    assert [entry.path for entry in report.paths] == ["src/a.py"]
    assert report.paths[0].state == "indexed"


def test_git_ignored_files_are_excluded_not_missing(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    # Not `build/`: the indexer skips that directory by name, and a verdict names
    # the first rule that matches.
    _write(workspace_root, ".gitignore", "generated/\n")
    _write(workspace_root, "generated/output.py", "def alpha():\n    return 1\n")
    # An index with no files is absent, and absent raises; index one real file so
    # the verdict under test is the git-ignore rule, not the index's readiness.
    indexed = _write(workspace_root, "src/a.py", "def beta():\n    return 2\n")
    root = make_workspace(files=[indexed], symbols=[{"file_path": "src/a.py", "symbol_name": "beta"}])
    _git_init(root)

    report = check_coverage(paths=["generated/output.py"], repo_root=root)
    assert _state_of(report, "generated/output.py") == "excluded"
    assert report.paths[0].reason == "git-ignored"
    assert report.paths[0].rule == "git-ignore"


def test_whole_repo_mode_covers_tracked_and_indexed_files(
    workspace_root: Path, make_workspace: WorkspaceFactory
) -> None:
    tracked = _write(workspace_root, "src/a.py", "def alpha():\n    return 1\n")
    _write(workspace_root, "src/untracked.py", "def beta():\n    return 2\n")
    root = make_workspace(files=[tracked], symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}])
    _git_init(root, "src/a.py")

    seen = {entry.path for entry in check_coverage(repo_root=root).paths}
    assert "src/a.py" in seen
    # Never a filesystem walk: an untracked, unindexed file is out of scope.
    assert "src/untracked.py" not in seen


def test_report_states_which_exclusion_rules_it_applied(make_workspace: WorkspaceFactory) -> None:
    """The rules are listed beside ``exclusion_source``, which keeps the value consumers captured."""
    root = make_workspace(files=[{"file_path": "src/a.py"}])
    report = check_coverage(paths=["src/a.py"], repo_root=root)
    assert report.exclusion_source == "git-ignore + unrecognised-file-type"
    assert report.to_dict()["exclusion_rules"] == list(EXCLUSION_RULES)
    assert report.repo_root == str(root)


def test_rebuilding_index_raises(make_workspace: WorkspaceFactory, tear_index: Callable[[Path], None]) -> None:
    """Verdicts judged against a torn index would call real, indexed files missing."""
    root = make_workspace(
        files=[{"file_path": "src/a.py"}],
        symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}],
    )
    tear_index(root)
    with pytest.raises(IndexRebuilding):
        check_coverage(paths=["src/a.py"], repo_root=root)


# --------------------------------------------------------------------------- #
# excluded vs missing: the indexer's own file-selection rules
# --------------------------------------------------------------------------- #


def _index_one_file(workspace_root: Path, make_workspace: WorkspaceFactory) -> Path:
    """A ready index holding one real file, so the verdict under test is a rule, not readiness."""
    row = _write(workspace_root, "src/a.py", "def alpha():\n    return 1\n")
    return make_workspace(files=[row], symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}])


def test_skipped_directory_reports_excluded_with_rule(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    """A supported language under ``data/`` is still never indexed: excluded, not missing."""
    _write(workspace_root, "data/fixture.py", "def rows():\n    return []\n")
    root = _index_one_file(workspace_root, make_workspace)

    (entry,) = check_coverage(paths=["data/fixture.py"], repo_root=root).paths

    assert (entry.state, entry.rule, entry.reason) == ("excluded", "skipped-directory", "skipped directory: data")
    assert entry.to_dict()["rule"] == "skipped-directory"


def test_lemoncrow_ignore_reports_excluded(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    _write(workspace_root, ".lemoncrow/.ignore", "vendored/\n")
    _write(workspace_root, "vendored/lib.py", "def helper():\n    return 1\n")
    root = _index_one_file(workspace_root, make_workspace)

    (entry,) = check_coverage(paths=["vendored/lib.py"], repo_root=root).paths

    assert (entry.state, entry.rule, entry.reason) == ("excluded", "lemoncrow-ignore", ".lemoncrow/.ignore")


def test_free_tier_cap_reports_excluded(workspace_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``context_engine`` the indexer keeps the first N files of its sorted scan, and so does the verdict.

    Indexed by the real engine under the same cap, so the files the verdict says
    the cap kept are the files the engine actually indexed.
    """
    from lemoncrow.pro.capabilities.code_context import CodeContextEngine

    for name in ("c", "a", "b"):
        _write(workspace_root, f"src/{name}.py", f"def {name}_fn():\n    return 1\n")
    monkeypatch.setattr("lemoncrow.core.capabilities.licensing.has_feature", lambda _feature: False)
    monkeypatch.setattr("lemoncrow.pro.capabilities.code_context.engine._FREE_TIER_MAX_FILES", 2)
    CodeContextEngine(workspace_root).index_repo()

    report = check_coverage(paths=["src/a.py", "src/b.py", "src/c.py"], repo_root=workspace_root)

    assert [(entry.path, entry.state) for entry in report.paths] == [
        ("src/a.py", "indexed"),
        ("src/b.py", "indexed"),
        ("src/c.py", "excluded"),
    ]
    assert (report.paths[2].rule, report.paths[2].reason) == ("free-tier-file-cap", "free-tier file cap")

    # Uncapped, the same unindexed file is simply not in the last index run.
    monkeypatch.setattr("lemoncrow.core.capabilities.licensing.has_feature", lambda _feature: True)
    (entry,) = check_coverage(paths=["src/c.py"], repo_root=workspace_root).paths
    assert (entry.state, entry.reason) == ("missing", "not in the last index run (an index-time exclude may apply)")


def test_prompt_txt_reports_unrecognised_type(workspace_root: Path, make_workspace: WorkspaceFactory) -> None:
    """A tracked prompt file has no language, so the indexer never takes it."""
    _write(workspace_root, "prompts/prompt.txt", "Review this diff.\n")
    root = _index_one_file(workspace_root, make_workspace)
    _git_init(root, "prompts/prompt.txt")

    (entry,) = check_coverage(paths=["prompts/prompt.txt"], repo_root=root).paths

    assert (entry.state, entry.rule, entry.reason) == ("excluded", "unrecognised-file-type", "unrecognised file type")


_AGREEMENT_TREE: dict[str, str] = {
    ".gitignore": "ignored/\ndata/secret.py\n",
    ".lemoncrow/.ignore": "vendored/\n",
    "src/app.py": "def app():\n    return 1\n",
    "src/notes.md": "# Notes\n\nSome prose.\n",
    "src/SHOUT.PY": "def shout():\n    return 1\n",
    "data/rows.py": "def rows():\n    return []\n",
    "data/secret.py": "def secret():\n    return 1\n",
    "vendored/lib.py": "def helper():\n    return 1\n",
    "ignored/secret.py": "def secret():\n    return 1\n",
    "prompts/prompt.txt": "Review this diff.\n",
}

#: The rule each path the scan passes over must name; `data/secret.py` is also
#: git-ignored, and the skipped directory is tried first.
_EXPECTED_RULES: dict[str, str] = {
    ".gitignore": "unrecognised-file-type",
    ".lemoncrow/.ignore": "skipped-directory",
    "src/SHOUT.PY": "source-file-scan",
    "data/rows.py": "skipped-directory",
    "data/secret.py": "skipped-directory",
    "vendored/lib.py": "lemoncrow-ignore",
    "ignored/secret.py": "git-ignore",
    "prompts/prompt.txt": "unrecognised-file-type",
}


def test_verdicts_agree_with_iter_source_files(workspace_root: Path) -> None:
    """Excluded exactly when the indexer's scan passes a path over, and every exclusion names its rule."""
    from lemoncrow.pro.capabilities.code_context import CodeContextEngine
    from lemoncrow.pro.capabilities.repo_map.graph import iter_source_files

    root = workspace_root.resolve()
    for rel, body in _AGREEMENT_TREE.items():
        _write(root, rel, body)
    _git_init(root)
    CodeContextEngine(root).index_repo()
    _write(root, "src/new.py", "def new():\n    return 1\n")  # selected by the scan, never indexed

    queried = [*_AGREEMENT_TREE, "src/new.py"]
    report = check_coverage(paths=queried, repo_root=root)
    scanned = {path.relative_to(root).as_posix() for path in iter_source_files(root)}

    assert sorted(entry.path for entry in report.paths) == sorted(queried)
    for entry in report.paths:
        assert (entry.state == "excluded") is (entry.path not in scanned), entry
    assert {entry.path: entry.rule for entry in report.paths if entry.state == "excluded"} == _EXPECTED_RULES
    assert _state_of(report, "src/app.py") == "indexed"
    assert _state_of(report, "src/new.py") == "missing"
