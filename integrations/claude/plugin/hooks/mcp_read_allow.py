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

It also records the session's working directory for every lc call, read-only or
not and whatever the opt-out says: ``<store root>/session_cwd/<session id>``
holds the payload's ``cwd``, which follows ``EnterWorktree``. The shared MCP
daemon reads it to route the session's calls to the worktree it works in
(``lemoncrow.gateway.adapters.mcp.session_root``). The record never changes
this hook's output and never raises; SessionStart prunes records older than
7 days.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
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


# A session id names a file, so it must never carry a path separator or a dot.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _session_cwd_dir() -> Path:
    """``<store root>/session_cwd``, the store root resolved as ``lemoncrow`` resolves it."""
    configured = os.environ.get("LEMONCROW_ROOT", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".lemoncrow"
    return root / "session_cwd"


def _record_session_cwd(payload: dict[str, Any]) -> None:
    """Write the payload's cwd to its session's file when it changed; never raise."""
    session_id = payload.get("session_id")
    cwd = payload.get("cwd")
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        return
    if not isinstance(cwd, str) or not cwd.strip():
        return
    try:
        directory = _session_cwd_dir()
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)
        target = directory / session_id
        try:
            if target.read_text(encoding="utf-8") == cwd:
                return
        except (OSError, UnicodeDecodeError):
            pass
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{session_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(cwd)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:  # the permission decision must survive any failure here
        print(f"lc: session cwd not recorded: {exc}", file=sys.stderr)


def main() -> int:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, TypeError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    _record_session_cwd(payload)
    if os.environ.get("LEMONCROW_MCP_READ_ALLOW", "1") == "0":
        return 0
    tool = _read_only_tool(str(payload.get("tool_name") or ""), payload.get("tool_input"))
    if tool is None:
        return 0
    _allow(f"lc {tool} is read-only (no writes, no shell); auto-allowed in every mode including Plan Mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
