"""F3: index coverage -- absent must be distinguishable from not-indexed."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lemoncrow.infra.code_intel.coverage import STATES, CoverageReport, check_coverage

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
    _write(workspace_root, ".gitignore", "build/\n")
    _write(workspace_root, "build/generated.py", "def alpha():\n    return 1\n")
    root = make_workspace()
    _git_init(root)

    report = check_coverage(paths=["build/generated.py"], repo_root=root)
    assert _state_of(report, "build/generated.py") == "excluded"
    assert report.paths[0].reason == "git-ignored"


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
    """The engine's real ignore rules are closed; say whose rules these are."""
    root = make_workspace(files=[{"file_path": "src/a.py"}])
    report = check_coverage(paths=["src/a.py"], repo_root=root)
    assert report.exclusion_source == "git-ignore + unrecognised-file-type"
    assert report.repo_root == str(root)
