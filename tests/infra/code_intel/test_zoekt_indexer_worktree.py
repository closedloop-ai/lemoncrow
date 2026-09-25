from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.zoekt.indexer import ZoektIndexer


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def linked_worktree(tmp_path: Path) -> Path:
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / "app.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    worktree = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "wt", str(worktree))
    return worktree


def test_a_linked_worktree_keeps_its_snapshot_in_its_own_git_admin_directory(
    linked_worktree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A linked worktree's `.git` is a file; reading a snapshot under it raised
    # NotADirectoryError and killed the snapshot thread on every engine start.
    first = ZoektIndexer(linked_worktree)
    assert first.ensure_snapshot().total_lines == 2

    admin_dir = Path(_git(linked_worktree, "rev-parse", "--absolute-git-dir"))
    assert (admin_dir / "lemoncrow" / "zoekt_snapshot.json").is_file()

    def no_rebuild(self: ZoektIndexer) -> None:
        raise AssertionError("the saved snapshot for this HEAD should be read back, not rebuilt")

    monkeypatch.setattr(ZoektIndexer, "_build_snapshot", no_rebuild)
    assert ZoektIndexer(linked_worktree).ensure_snapshot().total_lines == 2


def test_a_linked_worktree_snapshot_is_rebuilt_once_its_head_moves(linked_worktree: Path) -> None:
    ZoektIndexer(linked_worktree).ensure_snapshot()
    (linked_worktree / "app.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    _git(linked_worktree, "commit", "-qam", "grow")

    assert ZoektIndexer(linked_worktree).ensure_snapshot().total_lines == 3
