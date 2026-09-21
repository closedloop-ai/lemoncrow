"""git_tracked: which candidates git tracks, asked without listing the repository."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel import inclusion
from lemoncrow.infra.code_intel.inclusion import exclusion_rule, git_tracked


def _git(root: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
    }
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, timeout=30, env=env)


def _touch(root: Path, rel: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text("X = 1\n", encoding="utf-8")


def test_git_tracked_reads_bracketed_paths_literally(tmp_path: Path) -> None:
    """``[id]`` is a Next.js route segment, not a glob: only the literal path is tracked."""
    _git(tmp_path, "init", "-q")
    for rel in ("app/[id]/page.py", "app/i/page.py", "app/new.py"):
        _touch(tmp_path, rel)
    _git(tmp_path, "--literal-pathspecs", "add", "--", "app/[id]/page.py")

    assert git_tracked(tmp_path, ["app/[id]/page.py", "app/i/page.py", "app/new.py"]) == {"app/[id]/page.py"}


def test_git_tracked_answers_across_chunk_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunking bounds the argument list, never the answer: at, and one past, the chunk size."""
    _git(tmp_path, "init", "-q")
    rels = [f"build/m{index}.py" for index in range(5)]
    for rel in rels:
        _touch(tmp_path, rel)
    _git(tmp_path, "add", "--", *rels[:4])
    monkeypatch.setattr(inclusion, "_TRACKED_QUERY_CHUNK", 2)

    assert git_tracked(tmp_path, rels[:4]) == set(rels[:4])
    assert git_tracked(tmp_path, rels) == set(rels[:4])


def test_git_tracked_sees_into_submodules(tmp_path: Path) -> None:
    """The scan lists submodule files as tracked (``--recurse-submodules``); so must the per-edit check."""
    sub = tmp_path / "sub"
    top = tmp_path / "top"
    sub.mkdir()
    top.mkdir()
    _git(sub, "init", "-q")
    _touch(sub, "build/page.py")
    _git(sub, "add", "-A")
    _git(sub, "commit", "-qm", "sub")
    _git(top, "init", "-q")
    _git(top, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendor/sub")

    assert git_tracked(top, ["vendor/sub/build/page.py", "vendor/sub/build/other.py"]) == {"vendor/sub/build/page.py"}


def test_git_tracked_outside_a_repository_is_empty(tmp_path: Path) -> None:
    """Fail-open: nothing counts as tracked, so every candidate meets the full rule ladder."""
    _touch(tmp_path, "build/page.py")

    assert git_tracked(tmp_path, ["build/page.py"]) == frozenset()
    assert git_tracked(tmp_path, []) == frozenset()


def test_exclusion_rule_skips_the_directory_list_only_for_tracked_paths() -> None:
    rel = "apps/api/lib/build/prisma-cli-tracing.ts"

    assert exclusion_rule(rel, ignore_spec=None, ignored=set(), tracked={rel}) is None
    assert exclusion_rule(rel, ignore_spec=None, ignored=set(), tracked=set()) == (
        "skipped-directory",
        "skipped directory: build",
    )
