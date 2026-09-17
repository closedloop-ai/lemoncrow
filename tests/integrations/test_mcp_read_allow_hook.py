"""Tests for the Plan-Mode-safe PreToolUse allow hook for lc's read-only MCP tools.

The hook is a standalone script reading a JSON payload on stdin and printing an
optional JSON allow decision on stdout, so it is exercised as a subprocess with
crafted payloads (same shape as test_agent_redirect_hook.py).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "integrations" / "claude" / "plugin" / "hooks" / "mcp_read_allow.py"


def _hook_module() -> types.ModuleType:
    """Import the hook by path: it is a plugin script, not part of the package."""
    spec = importlib.util.spec_from_file_location("lemoncrow_mcp_read_allow_under_test", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    payload: dict, env_extra: dict | None = None, stdin_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin_text if stdin_text is not None else json.dumps(payload),
        capture_output=True,
        text=True,
        env={**os.environ, **(env_extra or {})},
        timeout=30,
    )


@pytest.mark.parametrize(
    "tool_name",
    [
        "mcp__lc__read",
        "mcp__lc__code_search",
        "mcp__lc__grep",
        "mcp__lc__relations",
        "mcp__lc__web_fetch",
        "mcp__lemoncrow__read",
    ],
)
def test_allows_read_only_tools(tool_name: str) -> None:
    proc = _run({"tool_name": tool_name, "tool_input": {}})
    assert proc.returncode == 0, proc.stderr
    hook_out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "PreToolUse"
    assert hook_out["permissionDecision"] == "allow"


@pytest.mark.parametrize(
    "tool_name",
    [
        "mcp__lc__edit",
        "mcp__lc__bash",
        "mcp__lc__sql",
        "mcp__lc__codemod",
        "mcp__lc__memory",
        "mcp__lc__verify",
        # `search` caches every query in the workspace search cache and
        # `context` records the task on the session ledger, so PRD-739 FR4
        # classifies both as writers -- Plan Mode must not auto-pass them.
        "mcp__lc__search",
        "mcp__lc__context",
        # The op-dispatcher can reach write-capable tools by name, so it is not
        # laundered through one allow decision.
        "mcp__lc__tool",
    ],
)
def test_stays_silent_for_write_capable_tools(tool_name: str) -> None:
    proc = _run({"tool_name": tool_name, "tool_input": {}})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytest.mark.parametrize(
    "tool_name",
    ["Read", "Bash", "mcp__other__read", "mcp__lc__read__extra", "lc__read", ""],
)
def test_stays_silent_for_foreign_tool_names(tool_name: str) -> None:
    proc = _run({"tool_name": tool_name, "tool_input": {}})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_opt_out_env_var_disables_allow() -> None:
    proc = _run({"tool_name": "mcp__lc__read"}, env_extra={"LEMONCROW_MCP_READ_ALLOW": "0"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_malformed_stdin_exits_zero_with_no_output() -> None:
    proc = _run({}, stdin_text="not json at all {{{")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytest.mark.parametrize(
    "tool_input",
    [{}, {"kind": "blast_radius", "path": "a.py"}, {"kind": "dead_code"}],
    ids=["default-kind", "blast_radius", "dead_code"],
)
def test_allows_graph_kinds_that_only_read(tool_input: dict) -> None:
    proc = _run({"tool_name": "mcp__lc__graph", "tool_input": tool_input})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"


@pytest.mark.parametrize(
    "tool_input",
    [
        {"kind": "pr_risk", "paths": ["a.py"]},
        {"kind": "index_docs"},
        {"kind": "recall_docs", "query": "design"},
        {"kind": "dead_code", "enable": True},
    ],
    ids=["pr_risk", "index_docs", "recall_docs", "enable"],
)
def test_stays_silent_for_graph_kinds_that_write(tool_input: dict) -> None:
    """`graph` is one name over many operations; the writing ones keep prompting."""
    proc = _run({"tool_name": "mcp__lc__graph", "tool_input": tool_input})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_stays_silent_for_graph_when_the_arguments_are_unreadable() -> None:
    """No arguments means no kind to judge, and the writing kinds look just like this."""
    proc = _run({"tool_name": "mcp__lc__graph"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


# --------------------------------------------------------------------------- #
# PRD-739 FR4 -- the hook and the broker do not drift                          #
# --------------------------------------------------------------------------- #

# The hook is a standalone script Claude Code runs as a subprocess, in installs
# where `lemoncrow` is not importable, so it cannot import the broker's policy.
# These tests pin the DIRECTION instead of equality: the broker legitimately
# runs more than Plan Mode auto-passes (code_query, code_changes,
# code_coverage_check), and `web_fetch` is auto-passed while the broker refuses
# it -- for outbound network reach, not for writing anything, and PRD-739 Open
# Question 2 leaves that undecided. Equality would fail on both.
_ALLOWED_BUT_NOT_BROKERED = frozenset({"web_fetch"})


def test_no_tool_the_broker_calls_a_writer_is_auto_allowed() -> None:
    from lemoncrow.gateway.adapters.mcp.broker_policy import BROKER_READ_ONLY

    drifted = _hook_module()._READ_ONLY_TOOLS - BROKER_READ_ONLY - _ALLOWED_BUT_NOT_BROKERED
    assert not drifted, (
        "Plan Mode auto-allows lc tools the broker will not run: "
        f"{sorted(drifted)}. Remove them from the hook, or classify them read-only in broker_policy."
    )


def test_no_graph_kind_the_broker_refuses_is_auto_allowed() -> None:
    from lemoncrow.gateway.adapters.mcp.broker_policy import GRAPH_DEFAULT_KIND, GRAPH_READ_ONLY_KINDS

    hook = _hook_module()
    drifted = hook._GRAPH_READ_ONLY_KINDS - GRAPH_READ_ONLY_KINDS
    assert not drifted, f"Plan Mode auto-allows graph kinds the broker refuses: {sorted(drifted)}"
    # An omitted kind has to mean the same operation on both sides, or the hook
    # judges one call and the server runs another.
    assert hook._GRAPH_DEFAULT_KIND == GRAPH_DEFAULT_KIND


def test_registered_in_plugin_hooks_json() -> None:
    hooks = json.loads((HOOK.parent / "hooks.json").read_text(encoding="utf-8"))
    commands = [hook["command"] for entry in hooks["hooks"]["PreToolUse"] for hook in entry["hooks"]]
    assert any("mcp_read_allow.py" in command for command in commands)
