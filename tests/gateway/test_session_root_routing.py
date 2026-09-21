"""Each session's calls route to the worktree it works in (ISS-10812, PLN-2070 PR 2).

Every test builds a real git repository with real linked worktrees, indexes the
main checkout with the real engine, and drives the tools through the daemon's
HTTP dispatcher, so each request carries its own ``Mcp-Session-Id`` exactly as a
bridge sends it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lemoncrow.gateway.adapters import mcp_http, mcp_server
from lemoncrow.gateway.adapters.mcp import session_root
from lemoncrow.infra.code_intel import worktree_seed
from lemoncrow.infra.code_intel.freshness import VersionedEngineCache
from lemoncrow.infra.code_intel.zoekt import adapter as zoekt_adapter
from lemoncrow.pro.capabilities.code_context import CodeContextEngine

_GIT = ["git", "-c", "user.email=route@test", "-c", "user.name=route", "-c", "commit.gpgsign=false"]
_HOOKS = Path(__file__).resolve().parents[2] / "integrations" / "claude" / "plugin" / "hooks"
# wt_a's copy of pkg/shared.py carries this many extra lines above shared_target.
_SHIFT = 5


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LEMONCROW_CODE_AUTOSYNC", "0")
    monkeypatch.setenv("LEMONCROW_CODE_FILE_WATCHER", "0")
    monkeypatch.setattr(worktree_seed, "_worktrees", {})
    monkeypatch.setattr(worktree_seed, "_checked_at", {})
    monkeypatch.setattr(zoekt_adapter, "_ROOT_OVERRIDES", {})
    monkeypatch.setattr(zoekt_adapter, "_SUPERVISORS", {})
    monkeypatch.setattr(session_root, "_bash_cwds", session_root.BashCwds())
    monkeypatch.setattr(session_root, "_cwd_files", session_root._SessionCwdFiles())
    cache = VersionedEngineCache(
        "test",
        recheck_seconds=0.0,
        on_evict=mcp_server._retire_code_engine,
        retire=worktree_seed.retire_worktree_engine,
    )
    monkeypatch.setattr(mcp_server, "_code_engine_cache", cache)
    monkeypatch.setattr(mcp_server, "_scoped_context_cache", {})
    yield
    cache.clear()
    cache.stop_sweeper()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run([*_GIT, *args], cwd=cwd, check=True, capture_output=True, text=True)


def _refresh(root: Path) -> None:
    """Open *root*'s engine through the daemon's factory (seeding a worktree) and bring it current."""
    mcp_server._code_context_engine(str(root)).index_repo(force=False)


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """(main, wt_a, wt_b): main is the workspace; each worktree holds a symbol only it has."""
    main = (tmp_path / "main").resolve()
    (main / "pkg").mkdir(parents=True)
    (main / "pkg" / "shared.py").write_text("def shared_target():\n    return 1\n", encoding="utf-8")
    (main / "pkg" / "base.py").write_text("def alpha_main():\n    return 0\n", encoding="utf-8")
    (main / ".gitignore").write_text(".lemoncrow/\n", encoding="utf-8")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    wt_a = (tmp_path / "wt_a").resolve()
    wt_b = (tmp_path / "wt_b").resolve()
    _git(main, "worktree", "add", "-q", "-b", "a", str(wt_a))
    _git(main, "worktree", "add", "-q", "-b", "b", str(wt_b))
    CodeContextEngine(main, autosync_enabled=False).index_repo(force=True)
    shifted = "".join(f"# a{i}\n" for i in range(_SHIFT)) + "def shared_target():\n    return 1\n"
    (wt_a / "pkg" / "shared.py").write_text(shifted, encoding="utf-8")
    (wt_a / "pkg" / "only_a.py").write_text("def omega_only_a():\n    return 'a'\n", encoding="utf-8")
    (wt_b / "pkg" / "only_b.py").write_text("def omega_only_b():\n    return 'b'\n", encoding="utf-8")
    for worktree in (wt_a, wt_b):
        _refresh(worktree)
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(main))
    return main, wt_a, wt_b


def _session() -> str:
    return f"s-{uuid.uuid4().hex[:12]}"


def _record_cwd(session_id: str, cwd: Path | str) -> None:
    """What the plugin's PreToolUse hook leaves behind for *session_id*."""
    directory = Path(os.environ["LEMONCROW_ROOT"]) / session_root.SESSION_CWD_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / session_id).write_text(str(cwd), encoding="utf-8")


def _call(session_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
    response = mcp_http._dispatch(request, session_id=session_id, host="claude", bridge_id="bridge-test")
    assert isinstance(response, dict) and "result" in response, response
    result = response["result"]
    assert isinstance(result, dict)
    return result


def _text(session_id: str, name: str, args: dict[str, Any]) -> str:
    result = _call(session_id, name, args)
    assert not result.get("isError"), result
    return str(result["content"][0]["text"])


def _search(session_id: str, query: str, **extra: Any) -> str:
    return _text(session_id, "code_search", {"query": query, **extra})


def test_two_concurrent_sessions_each_search_their_own_worktree(repos: tuple[Path, Path, Path]) -> None:
    """AC-2.1 / AC-2.2: no repo_root, each session answers from its worktree, with its line numbers."""
    main, wt_a, wt_b = repos
    session_a, session_b, session_main = _session(), _session(), _session()
    _record_cwd(session_a, wt_a / "pkg")  # a subdirectory still names its worktree
    _record_cwd(session_b, wt_b)
    _record_cwd(session_main, main)
    barrier = threading.Barrier(2)
    texts: dict[str, str] = {}

    def run(session_id: str, query: str) -> None:
        barrier.wait(timeout=30)
        texts[session_id] = _search(session_id, query)

    threads = [
        threading.Thread(target=run, args=(session_a, "omega_only_a")),
        threading.Thread(target=run, args=(session_b, "omega_only_b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert "omega_only_a" in texts[session_a], texts[session_a]
    assert f"repo_root: {wt_a}" in texts[session_a].splitlines()[0], texts[session_a]
    assert "omega_only_b" in texts[session_b], texts[session_b]
    assert f"repo_root: {wt_b}" in texts[session_b].splitlines()[0], texts[session_b]
    # Line numbers are the worktree's own: wt_a's shared_target sits _SHIFT lines lower.
    in_a = _search(session_a, "shared_target")
    in_b = _search(session_b, "shared_target")
    in_main = _search(session_main, "shared_target")
    assert f"shared_target L{1 + _SHIFT}-L{2 + _SHIFT}" in in_a, in_a
    assert "shared_target L1-L2" in in_b, in_b
    assert "shared_target L1-L2" in in_main and "repo_root:" not in in_main, in_main
    assert "omega_only_a" not in _search(session_b, "omega_only_a function")


def test_absolute_paths_inside_a_worktree_route_to_it_even_from_the_main_checkout(
    repos: tuple[Path, Path, Path],
) -> None:
    """AC-2.3: paths name the worktree, so its index answers whatever the session's cwd says."""
    main, wt_a, wt_b = repos
    session_id = _session()
    _record_cwd(session_id, main)

    text = _search(session_id, "omega_only_a", paths=[str(wt_a / "pkg")])

    assert "omega_only_a" in text, text
    assert f"repo_root: {wt_a}" in text.splitlines()[0], text
    refused = _call(session_id, "code_search", {"query": "shared_target", "paths": [str(wt_a), str(wt_b / "pkg")]})
    assert refused.get("isError") is True, refused
    message = refused["content"][0]["text"]
    assert str(wt_a) in message and str(wt_b) in message, message


def test_a_query_repeated_in_another_checkout_is_answered_and_blanked_only_in_the_same_one(
    repos: tuple[Path, Path, Path],
) -> None:
    """The near-repeat suppression counts a query as a repeat only against the checkout it was asked in."""
    main, wt_a, wt_b = repos
    session_id = _session()
    _record_cwd(session_id, main)
    assert "shared_target L1-L2" in _search(session_id, "shared_target")

    _record_cwd(session_id, wt_a)  # what EnterWorktree leaves behind
    in_a = _search(session_id, "shared_target")
    in_b = _search(session_id, "shared_target", paths=[str(wt_b / "pkg")])
    repeated_in_a = _search(session_id, "shared_target")

    assert f"shared_target L{1 + _SHIFT}-L{2 + _SHIFT}" in in_a, in_a
    assert "shared_target L1-L2" in in_b and f"repo_root: {wt_b}" in in_b, in_b
    assert repeated_in_a.startswith("no exact match") and "shared_target" not in repeated_in_a, repeated_in_a


def _fresh_worktree(main: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A worktree the daemon has never opened, with pkg/only_c.py (zeta_only_c) that only it has.

    Engines opened from here have autosync on (tests/gateway/conftest.py forces it
    off), so opening the worktree seeds it and runs the real post-seed first refresh.
    """
    forced_off = CodeContextEngine.__init__

    def with_autosync(self: CodeContextEngine, *args: Any, **kwargs: Any) -> None:
        forced_off(self, *args, **kwargs)
        self._autosync_enabled = True

    monkeypatch.setattr(CodeContextEngine, "__init__", with_autosync)
    worktree = (main.parent / "wt_c").resolve()
    _git(main, "worktree", "add", "-q", "-b", "c", str(worktree))
    (worktree / "pkg" / "only_c.py").write_text("def zeta_only_c():\n    return 'c'\n", encoding="utf-8")
    return worktree


def test_the_first_search_in_a_fresh_worktree_waits_for_its_first_refresh(
    repos: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-1.3: the first answer comes from the worktree's own files, not the main checkout's seed."""
    wt_c = _fresh_worktree(repos[0], monkeypatch)
    session_id = _session()
    _record_cwd(session_id, wt_c)

    first = _search(session_id, "zeta_only_c")

    assert "pkg/only_c.py" in first and f"repo_root: {wt_c}" in first.splitlines()[0], first
    assert mcp_server._INDEX_REFRESHING_NOTE not in first, first


def test_a_first_refresh_slower_than_the_wait_is_announced_and_a_retry_is_answered(
    repos: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    real_refresh = worktree_seed._first_refresh

    def slow_refresh(engine: Any, *args: Any) -> None:
        with engine._autosync_lock:  # keeps autosync's own poll off the index meanwhile
            release.wait(timeout=60)
            real_refresh(engine, *args)

    monkeypatch.setattr(worktree_seed, "_first_refresh", slow_refresh)
    monkeypatch.setattr(worktree_seed, "FIRST_REFRESH_WAIT_S", 0.2)
    wt_c = _fresh_worktree(repos[0], monkeypatch)
    session_id = _session()
    _record_cwd(session_id, wt_c)

    first = _search(session_id, "zeta_only_c")
    release.set()
    for thread in threading.enumerate():
        if thread.name == "lemoncrow-worktree-seed-refresh":
            thread.join(timeout=60)
    retry = _search(session_id, "where is zeta_only_c defined")

    assert mcp_server._INDEX_REFRESHING_NOTE in first, first
    assert "pkg/only_c.py" in retry, retry


def test_a_reworded_repeat_of_an_answer_from_the_refreshed_index_is_still_blanked(
    repos: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    wt_c = _fresh_worktree(repos[0], monkeypatch)
    session_id = _session()
    _record_cwd(session_id, wt_c)

    first = _search(session_id, "zeta_only_c")
    reworded = _search(session_id, "where is zeta_only_c defined")

    assert mcp_server._INDEX_REFRESHING_NOTE not in first, first
    assert reworded.startswith("no exact match") and "only_c" not in reworded, reworded


def _other_repos_worktree(tmp_path: Path) -> Path:
    other = tmp_path / "other"
    other.mkdir()
    (other / "f.txt").write_text("x\n", encoding="utf-8")
    _git(other, "init", "-q", "-b", "main")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "init")
    worktree = tmp_path / "other_wt"
    _git(other, "worktree", "add", "-q", "-b", "o", str(worktree))
    return worktree


@pytest.mark.parametrize("kind", ["another-repo", "plain-dir", "missing", "relative"])
def test_a_recorded_cwd_outside_the_workspace_is_never_used(
    repos: tuple[Path, Path, Path], tmp_path: Path, kind: str
) -> None:
    """AC-2.6: only the workspace root or one of its linked worktrees can route a call."""
    _main, wt_a, _wt_b = repos
    plain = tmp_path / "plain"
    plain.mkdir()
    cwd = {
        "another-repo": str(_other_repos_worktree(tmp_path)),
        "plain-dir": str(plain),
        "missing": str(tmp_path / "gone"),
        "relative": os.path.relpath(wt_a),
    }[kind]
    session_id = _session()
    _record_cwd(session_id, cwd)

    text = _search(session_id, "shared_target")

    assert "shared_target L1-L2" in text and "repo_root:" not in text, text
    assert session_root.accept_session_cwd(repos[0], cwd) is None


def test_relative_creates_land_in_each_calling_sessions_worktree(repos: tuple[Path, Path, Path]) -> None:
    """AC-2.2: a relative create follows the recorded cwd, or else the session's own last bash cwd."""
    main, wt_a, wt_b = repos
    hooked_a, hooked_b, bash_a, bash_b = _session(), _session(), _session(), _session()
    _record_cwd(hooked_a, wt_a)
    _record_cwd(hooked_b, wt_b)
    # Hosts without the hook: each session's bash cwd is its own, never the last one seen.
    _text(bash_a, "bash", {"command": "true", "cwd": str(wt_a)})
    _text(bash_b, "bash", {"command": "true", "cwd": str(wt_b)})

    for session_id, name in (
        (hooked_a, "hooked.txt"),
        (hooked_b, "hooked.txt"),
        (bash_a, "bash.txt"),
        (bash_b, "bash.txt"),
    ):
        _text(session_id, "edit", {"edits": [{"path": name, "new": f"{session_id}\n", "replace": True}]})

    assert (wt_a / "hooked.txt").read_text(encoding="utf-8") == f"{hooked_a}\n"
    assert (wt_b / "hooked.txt").read_text(encoding="utf-8") == f"{hooked_b}\n"
    assert (wt_a / "bash.txt").read_text(encoding="utf-8") == f"{bash_a}\n"
    assert (wt_b / "bash.txt").read_text(encoding="utf-8") == f"{bash_b}\n"
    assert not (main / "hooked.txt").exists() and not (main / "bash.txt").exists()


def test_reads_and_cwd_less_bash_follow_the_session_into_its_worktree(repos: tuple[Path, Path, Path]) -> None:
    main, wt_a, _wt_b = repos
    in_wt, in_main = _session(), _session()
    _record_cwd(in_wt, wt_a)
    _record_cwd(in_main, main)

    assert f"{1 + _SHIFT}\tdef shared_target" in _text(in_wt, "read", {"files": ["pkg/shared.py"]})
    assert _text(in_main, "read", {"files": ["pkg/shared.py"]}) == "def shared_target():\n    return 1\n"
    assert str(wt_a) in _text(in_wt, "bash", {"command": "pwd -P"})
    assert str(wt_a) not in _text(in_main, "bash", {"command": "pwd -P"})


def test_a_file_read_in_full_in_the_main_checkout_does_not_block_its_worktree_copys_cat(
    repos: tuple[Path, Path, Path],
) -> None:
    """The redundant-dump check resolves a relative path where the cwd-less command runs."""
    main, wt_a, _wt_b = repos
    session_id = _session()
    _record_cwd(session_id, main)
    _text(session_id, "read", {"files": ["pkg/shared.py:full"]})

    _record_cwd(session_id, wt_a)
    worktree_copy = _text(session_id, "bash", {"command": "cat pkg/shared.py"})
    _text(session_id, "read", {"files": ["pkg/shared.py:full"]})
    reread = _text(session_id, "bash", {"command": "cat pkg/shared.py"})

    assert "# a0" in worktree_copy, worktree_copy
    assert reread.startswith("[lc: already read in full"), reread


def test_a_missing_workspace_root_sends_the_dump_check_where_the_command_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLAUDE_WORKSPACE_ROOT that is not a directory falls back to the process cwd for both."""
    here = (tmp_path / "here").resolve()
    here.mkdir()
    (here / "notes.txt").write_text("already seen\n", encoding="utf-8")
    (tmp_path / "a-file").write_text("", encoding="utf-8")
    # Under a file, so nothing the daemon does can create it.
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path / "a-file" / "gone"))
    session_id = _session()
    _text(session_id, "read", {"files": [f"{here / 'notes.txt'}:full"]})
    monkeypatch.chdir(here)

    dumped = _text(session_id, "bash", {"command": "head notes.txt"})

    assert dumped.startswith("[lc: already read in full"), dumped


def test_the_resolved_against_note_appears_only_for_a_relative_path(repos: tuple[Path, Path, Path]) -> None:
    """AC-2.5: an absolute path owes nothing to the inferred worktree, so it says nothing about it."""
    _main, wt_a, wt_b = repos
    by_bash, by_hook = _session(), _session()
    _text(by_bash, "bash", {"command": "true", "cwd": str(wt_a)})
    _record_cwd(by_hook, wt_b)

    absolute = _text(
        by_bash,
        "edit",
        {"edits": [{"path": str(wt_a / "pkg" / "only_a.py"), "old": "'a'", "new": "'A'"}]},
    )
    relative = _text(by_bash, "edit", {"edits": [{"path": "pkg/only_a.py", "old": "'A'", "new": "'AA'"}]})
    hooked = _text(by_hook, "edit", {"edits": [{"path": "pkg/only_b.py", "old": "'b'", "new": "'B'"}]})

    assert "resolved against" not in absolute, absolute
    assert f"resolved against worktree {wt_a} (from last bash cwd)" in relative, relative
    assert f"resolved against worktree {wt_b} (from session cwd)" in hooked, hooked
    assert (wt_a / "pkg" / "only_a.py").read_text(encoding="utf-8").endswith("'AA'\n")


_GOLDEN_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("code_search", {"query": "shared_target"}),
    ("code_query", {"select": "symbols", "where": {"name_regex": "^(alpha|shared)"}}),
    ("relations", {"symbol": "shared_target", "kind": "self"}),
    ("read", {"files": ["pkg/shared.py"]}),
)


def test_a_main_checkout_session_answers_as_an_unrouted_call(
    repos: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-2.4: a session in the main checkout gets byte-identical answers to a call routing never touched."""
    main, _wt_a, _wt_b = repos
    routed = _session()
    _record_cwd(routed, main)
    routed_texts = [_text(routed, name, args) for name, args in _GOLDEN_CALLS]

    monkeypatch.setattr(session_root, "_current", lambda: None)
    unrouted = _session()
    unrouted_texts = [_text(unrouted, name, args) for name, args in _GOLDEN_CALLS]

    assert routed_texts == unrouted_texts
    assert all("repo_root:" not in text for text in routed_texts)


def test_the_bash_cwd_map_is_bounded_and_forgets_idle_sessions() -> None:
    """AC-2.7: at most 256 sessions, least recently used out first; 24 idle hours forgets one."""
    now = [0.0]
    cwds = session_root.BashCwds(clock=lambda: now[0])
    for i in range(session_root.MAX_SESSIONS + 10):
        cwds.record(f"s{i}", f"/cwd/{i}")
    assert len(cwds) == 256
    assert cwds.get("s0") is None and cwds.get("s9") is None
    assert cwds.get("s10") == "/cwd/10"

    now[0] = 23 * 3600.0
    assert cwds.get("s11") == "/cwd/11"  # used, so its idle clock restarts
    now[0] = 24 * 3600.0
    assert cwds.get("s12") is None
    assert cwds.get("s11") == "/cwd/11"
    assert len(cwds) == 1


def _run_hook(script: str, payload: dict[str, Any], store: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_HOOKS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**os.environ, "LEMONCROW_ROOT": str(store), **env},
        timeout=60,
    )


def test_the_hook_writes_the_cwd_on_a_change_and_skips_an_unchanged_one(tmp_path: Path) -> None:
    store = tmp_path / "store"
    payload = {"session_id": "sess-1", "cwd": "/work/one", "tool_name": "mcp__lc__edit", "tool_input": {}}
    record = store / "session_cwd" / "sess-1"

    assert _run_hook("mcp_read_allow.py", payload, store).returncode == 0
    assert record.read_text(encoding="utf-8") == "/work/one"
    assert (record.parent.stat().st_mode & 0o777) == 0o700
    first = record.stat()
    os.utime(record, ns=(first.st_atime_ns, first.st_mtime_ns - 10**9))
    written = record.stat().st_mtime_ns

    _run_hook("mcp_read_allow.py", payload, store)
    assert record.stat().st_mtime_ns == written, "an unchanged cwd was rewritten"

    _run_hook("mcp_read_allow.py", {**payload, "cwd": "/work/two"}, store)
    assert record.read_text(encoding="utf-8") == "/work/two"
    assert sorted(p.name for p in record.parent.iterdir()) == ["sess-1"], "a temp file was left behind"


def test_the_hook_records_before_every_early_return_and_never_changes_its_decision(tmp_path: Path) -> None:
    store = tmp_path / "store"
    edit = {"session_id": "sess-2", "cwd": "/work/edit", "tool_name": "mcp__lc__edit", "tool_input": {}}
    read = {**edit, "cwd": "/work/read", "tool_name": "mcp__lc__read"}

    killed = _run_hook("mcp_read_allow.py", edit, store, LEMONCROW_MCP_READ_ALLOW="0")
    assert killed.stdout == "" and (store / "session_cwd" / "sess-2").read_text(encoding="utf-8") == "/work/edit"

    allowed = _run_hook("mcp_read_allow.py", read, store)
    assert json.loads(allowed.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert (store / "session_cwd" / "sess-2").read_text(encoding="utf-8") == "/work/read"

    blocked = tmp_path / "not-a-dir"
    blocked.write_text("", encoding="utf-8")
    failed = _run_hook("mcp_read_allow.py", read, blocked)
    assert failed.returncode == 0
    assert failed.stdout == allowed.stdout
    assert "session cwd not recorded" in failed.stderr


def test_session_start_prunes_cwd_records_older_than_seven_days(tmp_path: Path) -> None:
    """AC-2.7: SessionStart drops what no session has refreshed in 7 days."""
    store = tmp_path / "store"
    records = store / "session_cwd"
    records.mkdir(parents=True)
    week = 7 * 24 * 3600
    now = time.time()
    for name, age in (("stale", week + 60), ("fresh", week - 60)):
        (records / name).write_text("/work", encoding="utf-8")
        os.utime(records / name, (now - age, now - age))

    started = _run_hook(
        "session_start.py",
        {"session_id": "sess-3", "cwd": str(tmp_path), "source": "resume"},
        store,
        CLAUDE_WORKSPACE_ROOT=str(tmp_path),
    )

    assert started.returncode == 0, started.stderr
    assert sorted(p.name for p in records.iterdir()) == ["fresh"]
