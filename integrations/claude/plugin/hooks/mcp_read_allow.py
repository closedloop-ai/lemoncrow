"""PreToolUse auto-allow for lc's genuinely read-only MCP tools.

Claude Code's Plan Mode runs its own approval gate, independent of
``permissions.allow``. It auto-passes only the built-in read tools
(Read/Grep/Glob/WebFetch); a third-party MCP tool is never in that set, so
``mcp__lc__read`` & friends prompt on every call even when ``lc init`` /
``install_claude.sh`` already whole-tool-allowed them. There is no MCP-side
annotation Claude Code honours for this -- the only thing that suppresses the
prompt in *every* mode is a PreToolUse hook returning
``hookSpecificOutput.permissionDecision: "allow"``.

Scope is deliberately narrow. Only lookup-only tools are named here; anything
that can mutate a file, a store, or a shell (``edit``, ``bash``, ``sql``,
``codemod``, ``memory``, ``compact``, ``verify``, ``agent``, ``workflow``) is
omitted and keeps prompting normally. ``tool`` is omitted too: it dispatches to
any rarely-used lc tool by name, including write-capable ones, so allowing it
would launder the whole surface through one decision. ``search`` (which caches
every query in the workspace search cache) and ``context`` (which records the
task on the session ledger) are writers under PRD-739 FR4, so they are omitted
as well.

What this list holds has to stay inside what the MCP ``tool`` broker will run,
network tools aside -- the broker refuses those for reach, not for writes. That
agreement is pinned by ``tests/integrations/test_mcp_read_allow_hook.py``, not
by a shared import: Claude Code runs this file as a subprocess in plugin
installs where ``lemoncrow`` is not importable, so it stays standard-library
only.

Stays silent (no decision at all) for every other tool, so it never overrides a
user's own deny rule for something outside this list.

Fail-open; opt-out via LEMONCROW_MCP_READ_ALLOW=0.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# MCP server key as registered by install_claude.sh (user scope: "lc",
# workspace .mcp.json / plugin mcp.json: "lemoncrow").
_SERVERS = frozenset({"lc", "lemoncrow"})

# Read-only lc tools. Each only reads the workspace, the index, or the network;
# none writes files, the store, or runs a command.
_READ_ONLY_TOOLS = frozenset(
    {
        "blame",
        "code_search",
        "graph",
        "grep",
        "orient",
        "read",
        "relations",
        "web_fetch",
    }
)

# ``graph`` is one name over many operations, so the tool alone does not say
# whether a call reads: ``index_docs`` writes the design-doc store, ``pr_risk``
# folds each changed file into the machine-wide semantic file index,
# ``recall_docs`` embeds its query through the configured embedder, and
# ``enable`` switches doc indexing on. Only the kinds that read the code index
# or git history are auto-allowed; the rest keep prompting.
_GRAPH_READ_ONLY_KINDS = frozenset(
    {
        "blast_radius",
        "centrality",
        "commit_provenance",
        "coupling",
        "cycles",
        "dead_code",
        "design_gaps",
        "topology",
        "verify_design",
    }
)
_GRAPH_DEFAULT_KIND = "blast_radius"


def _graph_reads_only(tool_input: Any) -> bool:
    """True when this ``graph`` call names a kind that only reads.

    Unreadable arguments count as not read-only: an omitted ``tool_input`` would
    otherwise auto-allow whatever kind the call actually carried. ``kind`` is
    raw model-supplied JSON, so a non-string one is judged rather than hashed --
    the same shape the broker's own vetting uses -- because a ``list``/``dict``
    would raise out of the membership test and out of the hook.
    """
    if not isinstance(tool_input, dict) or "enable" in tool_input:
        return False
    kind = tool_input.get("kind", _GRAPH_DEFAULT_KIND)
    return isinstance(kind, str) and kind in _GRAPH_READ_ONLY_KINDS


def _read_only_tool(tool_name: str, tool_input: Any = None) -> str | None:
    """Return the bare lc tool name when this call is an allowed read call."""
    parts = tool_name.split("__")
    if len(parts) != 3 or parts[0] != "mcp" or parts[1] not in _SERVERS:
        return None
    tool = parts[2]
    if tool not in _READ_ONLY_TOOLS:
        return None
    if tool == "graph" and not _graph_reads_only(tool_input):
        return None
    return tool


def _allow(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main() -> int:
    if os.environ.get("LEMONCROW_MCP_READ_ALLOW", "1") == "0":
        return 0
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, TypeError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    tool = _read_only_tool(str(payload.get("tool_name") or ""), payload.get("tool_input"))
    if tool is None:
        return 0
    _allow(f"lc {tool} is read-only (no writes, no shell); auto-allowed in every mode including Plan Mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
