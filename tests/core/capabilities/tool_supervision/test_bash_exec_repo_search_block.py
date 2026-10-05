"""Repo-wide shell search (`grep -r`, `rg`, `git grep`) over the indexed repo is blocked in
favour of code_search; everything the index cannot answer keeps running."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import lemoncrow.pro.capabilities.tool_supervision.bash_exec as bx


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_text("def f():\n    pass\n")
    (root / ".gitignore").write_text(".claude/worktrees/\n.closedloop-ai/\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    _git(root, "worktree", "add", "-q", str(root / ".claude" / "worktrees" / "wt"))
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / ".closedloop-ai" / "campaigns").mkdir(parents=True)
    (tmp_path / "scratch").mkdir()
    return root


def _decide(command: str, repo: Path, cwd: Path | None = None) -> bx.CommandPolicyDecision:
    return bx.classify_command(command, cwd=cwd or repo, search_root=repo)


def _blocked(command: str, repo: Path, cwd: Path | None = None) -> bool:
    return _decide(command, repo, cwd).action == "block"


@pytest.mark.parametrize(
    "command",
    [
        "grep -rn 'def f' src",
        "grep -rn 'def f'",
        "grep -R --include=*.py f .",
        "grep --recursive -e f src",
        "grep -rniA3 f src",
        "grep -rn f src/*.py",
        "rg f",
        "rg -n -g '*.py' f src",
        "git grep -n f",
        "git grep -n f -- src",
        "git -C src grep f",
        "grep -rn f src | head -20",
        "cd src && grep -rn f .",
        "ls; grep -rn f src",
        "timeout 10 rg f src",
    ],
)
def test_a_recursive_search_of_the_repo_is_blocked_and_names_the_alternatives(repo: Path, command: str) -> None:
    decision = _decide(command, repo)
    assert decision.action == "block", command
    assert "code_search" in (decision.reason or "")
    assert bx.SHELL_SEARCH_OPT_IN in (decision.reason or "")


def test_a_search_inside_a_worktree_of_the_repo_is_blocked(repo: Path) -> None:
    worktree = repo / ".claude" / "worktrees" / "wt"
    assert _blocked("grep -rn f src", worktree, cwd=worktree)
    assert _blocked(f"grep -rn f {worktree / 'src'}", repo)


@pytest.mark.parametrize(
    "command",
    [
        "grep -n f src/pkg/mod.py",  # single file
        "grep -rn f src/pkg/mod.py",
        "grep f src/pkg/mod.py src/pkg/mod.py",  # not recursive
        "rg f src/pkg/mod.py",
        "grep -rn f node_modules/lib",  # outside the index
        "grep -rl f .closedloop-ai/campaigns",  # gitignored
        "rg -u f",
        "rg --no-ignore f src",
        "rg --files src",
        "git grep -n f origin/main",  # another revision
        "git grep -n f HEAD~1 -- src",
        "git log | grep f",  # stdin
        "cat src/pkg/mod.py | rg f",
        "grep -rl f src | xargs sed -i s/f/g/",  # output feeds a command
        "sed -i s/f/g/ $(grep -rl f src)",
        "grep -rn f $DIR",  # opaque path
        "cd - && grep -rn f .",
        "LEMONCROW_SHELL_SEARCH=1 grep -rn f src",  # explicit opt-in
        "cd src && LEMONCROW_SHELL_SEARCH=1 rg f",
        "echo 'grep -rn f src'",
        "grep -rn f does-not-exist",
    ],
)
def test_searches_the_index_cannot_answer_still_run(repo: Path, command: str) -> None:
    assert not _blocked(command, repo), command


def test_a_search_outside_the_repo_runs(repo: Path) -> None:
    scratch = repo.parent / "scratch"
    assert not _blocked(f"grep -rn f {scratch}", repo)
    assert not _blocked("grep -rn f .", repo, cwd=scratch)
    assert not _blocked(f"cd {scratch} && rg f", repo)


def test_without_a_search_root_nothing_is_blocked(repo: Path) -> None:
    assert bx.classify_command("grep -rn f src", cwd=repo).action != "block"


def test_a_workspace_outside_any_git_repo_blocks_nothing(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    (plain / "src").mkdir(parents=True)
    assert bx.classify_command("grep -rn f src", cwd=plain, search_root=plain).action != "block"


def test_the_mcp_bash_tool_returns_the_block_without_running_the_search(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lemoncrow.gateway.adapters.mcp import bash

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(repo))
    result = bash._run_bash_tool(command="grep -rn 'def f' src", cwd=str(repo))
    assert isinstance(result, dict)
    assert result["blocked"] is True
    assert "code_search" in result["blocked_reason"]

    allowed = bash._run_bash_tool(command="LEMONCROW_SHELL_SEARCH=1 grep -rn 'def f' src", cwd=str(repo))
    assert isinstance(allowed, dict)
    assert not allowed.get("blocked")
    assert "mod.py" in str(allowed.get("stdout"))
