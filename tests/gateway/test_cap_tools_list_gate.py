"""Open-source runtime: the MCP tool surface is NEVER gated or hidden.

The former savings-cap dormancy gate on tools/list and tools/call was removed
(see docs/maintenance-mode-transition.md). Every tool is always advertised and
callable, regardless of any legacy over-cap subscription state left on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _seed_legacy_over_cap(root: Path) -> None:
    # A leftover "over cap" flag from a legacy install must have NO effect.
    from lemoncrow.core.capabilities.plugin_runtime import _write_json, subscription_state_path

    _write_json(subscription_state_path(root), {"plan": "free", "savingsOverCap": True})


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LEMONCROW_ROOT", str(tmp_path))


def _list() -> list[dict]:
    from lemoncrow.gateway.adapters import mcp_server

    resp = mcp_server._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert isinstance(resp, dict)
    return resp["result"]["tools"]


def test_tools_always_listed(tmp_path: Path) -> None:
    tools = _list()
    assert len(tools) > 0
    assert any(t["name"] in {"read", "code_search", "bash", "edit"} for t in tools)


def test_tools_listed_even_with_legacy_over_cap_state(tmp_path: Path) -> None:
    _seed_legacy_over_cap(tmp_path)
    tools = _list()
    assert len(tools) > 0
    assert any(t["name"] in {"read", "code_search", "bash", "edit"} for t in tools)


def test_tools_call_never_rejected_by_cap(tmp_path: Path) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    _seed_legacy_over_cap(tmp_path)
    resp = mcp_server._handle(
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "read", "arguments": {"path": "x"}}}
    )
    assert isinstance(resp, dict)
    # Never the old "anonymous savings cap reached" rejection.
    assert "cap reached" not in str(resp).lower()


def test_crossing_legacy_cap_state_has_no_effect(tmp_path: Path) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    mcp_server._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert len(_list()) > 0
    _seed_legacy_over_cap(tmp_path)
    assert len(_list()) > 0


def test_core_profile_keeps_normal_tools_eager_and_brokers_rare_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    monkeypatch.setattr(mcp_server, "mcp_tool_visible_to_llm", lambda _name: True)
    names = {tool["name"] for tool in _list()}

    assert {"code_search", "read", "edit", "bash", "web_fetch", "tool"} <= names
    assert "blame" not in names

    result = mcp_server._TOOL_BROKER_SPEC["handler"]({"action": "search", "query": "blame"})
    assert result["matches"]
    assert result["matches"][0]["name"] == "blame"


def test_full_profile_does_not_advertise_unnecessary_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "full")
    monkeypatch.setattr(mcp_server, "mcp_tool_visible_to_llm", lambda _name: True)
    names = {tool["name"] for tool in _list()}
    assert "sql" in names
    assert "tool" not in names


# --------------------------------------------------------------------------- #
# F4 -- the broker reaches every unadvertised tool, not just one               #
# --------------------------------------------------------------------------- #


def _broker(payload: dict) -> dict:
    from lemoncrow.gateway.adapters import mcp_server

    result = mcp_server._TOOL_BROKER_SPEC["handler"](payload)
    assert isinstance(result, dict)
    return result


def _stub_handler(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Swap one tool's handler for a sentinel.

    The point under test is the broker's guard, not the tool body -- and the
    real `relations`/`graph` handlers reach into the compiled engine.
    """
    from lemoncrow.gateway.adapters import mcp_server

    spec = dict(mcp_server.TOOLS[name])
    spec["handler"] = lambda args, _n=name: {"called": _n, "args": args}
    monkeypatch.setitem(mcp_server.TOOLS, name, spec)


@pytest.mark.parametrize("name", ["blame", "graph", "grep", "orient", "search", "statusline_segment"])
def test_broker_calls_tools_that_are_hidden_under_the_core_profile(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """A read-only tool hidden from tools/list must still be reachable through the broker.

    The parametrization is every allow-listed tool the core profile hides.

    The old guard refused a tool as "already exposed" whenever it sat in
    _CORE_MCP_TOOLS, even when HIDDEN_LLM_TOOLS meant nothing ever advertised
    it -- so it was unreachable by every route, leaving `statusline_segment` as
    the only tool the broker could reach.

    `relations` used to be the headline case here. It is now advertised outright
    (see ``_FORCE_VISIBLE_TOOLS``), so it has moved to
    :func:`test_relations_is_advertised_under_every_profile` and the guard is
    pinned with tools that are still hidden.
    """
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    assert name not in {tool["name"] for tool in _list()}
    _stub_handler(monkeypatch, name)

    assert _broker({"action": "call", "name": name, "arguments": {"op": "callers"}}) == {
        "called": name,
        "args": {"op": "callers"},
    }


@pytest.mark.parametrize("profile", ["core", "full"])
def test_relations_is_advertised_under_every_profile(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    """The only enumerative symbol tool has to be visible to be routed to.

    Hidden, an agent sees one code-intel tool under the core profile --
    `code_search`, which ranks -- and reads its top-N as the complete caller
    set. `code_changes` does not substitute: a builder about to edit a symbol
    has a symbol, not a diff.
    """
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", profile)
    advertised = {tool["name"] for tool in _list()}
    assert "relations" in advertised
    assert "code_coverage_check" in advertised, "a negative result must stay auditable under both profiles"


@pytest.mark.parametrize("profile", ["core", "full"])
@pytest.mark.parametrize("name", ["code_changes", "code_query"])
def test_the_review_surface_is_advertised_under_every_profile(
    monkeypatch: pytest.MonkeyPatch, profile: str, name: str
) -> None:
    """Both were registered-but-hidden under `core`, reachable only via broker.

    "they are in the full profile" was not an answer. The profile is read from
    the *daemon's* environment, so a long-lived daemon started without
    LEMONCROW_MCP_TOOL_PROFILE serves core to every client no matter what the
    caller exports -- which is why both profiles looked identical from outside.

    Nor is the broker an acceptable substitute here. Its purpose is calling
    arbitrary registered tools by exact name, so routing a review agent through
    it to analyse one diff widens a trust boundary to work around a visibility
    list.
    """
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", profile)
    assert name in {tool["name"] for tool in _list()}


def test_advertised_relations_is_not_also_broker_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """One route per tool: the broker exists for what tools/list does not show."""
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    assert not mcp_server._broker_reachable("relations", mcp_server.TOOLS["relations"])


def test_broker_search_returns_hidden_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """`search` used to filter to *visible* tools, so it could never match."""
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    from lemoncrow.gateway.adapters import mcp_server

    found = {match["name"] for match in _broker({"action": "search", "query": ""})["matches"]}
    assert found
    assert found <= mcp_server._BROKER_READ_ONLY
    assert "blame" in {match["name"] for match in _broker({"action": "search", "query": "blame"})["matches"]}


def test_broker_search_never_returns_an_advertised_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    advertised = {tool["name"] for tool in _list()}
    found = {match["name"] for match in _broker({"action": "search", "query": ""})["matches"]}
    assert not (found & advertised)


def test_broker_runs_an_already_advertised_tool_and_points_at_the_direct_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broker must not dead-end a caller whose tool list predates the tool.

    This used to raise "already exposed; call it directly" -- advice the caller
    could not act on. A host captures tools/list once at connect time, so a tool
    added since (``relations``, as it happens) is advertised by the server and
    absent from a live session's list; refusing left it reachable by no route,
    and the reviewer that hit this had to hand-drive raw JSON-RPC.
    """
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    assert "relations" in {tool["name"] for tool in _list()}
    _stub_handler(monkeypatch, "relations")

    result = _broker({"action": "call", "name": "relations", "arguments": {"symbol": "merge"}})

    assert result["called"] == "relations"
    assert result["args"] == {"symbol": "merge"}
    assert "reconnect" in result["broker_note"]


def test_broker_note_is_absent_when_the_tool_is_genuinely_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """The note is a nudge toward a route that exists -- not boilerplate."""
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    assert "blame" not in {tool["name"] for tool in _list()}
    _stub_handler(monkeypatch, "blame")

    assert "broker_note" not in _broker({"action": "call", "name": "blame", "arguments": {}})


# --------------------------------------------------------------------------- #
# PRD-739 FR4 -- the broker reaches read-only tools only                       #
# --------------------------------------------------------------------------- #

# Every registered tool the broker refuses (PLN-2027 PR 3). The broker reads only
# its allow-list; these sets exist so that a new tool fails
# test_every_registered_tool_is_classified until someone decides where it goes.
# Each one executes commands, writes files or LemonCrow state (index, cache,
# memory, session, review data), reaches the network, or calls other tools.
_BROKER_DENIED = frozenset(
    {
        "agent",
        "bash",
        "cache",
        "codemod",
        "compact",
        "edit",
        "index",
        "mcp",
        "memory",
        "rescue",
        "review_evidence",
        "review_feedback_addressed",
        "review_rationale",
        "sql",
        "tool",
        "trace",
        "verify",
        "web_fetch",
        "workflow",
    }
)
# Denied until shown read-only, and the code shows otherwise: `scan` runs the
# ast-grep binary, `context` records the task on the session ledger.
_BROKER_DENIED_UNTIL_SHOWN_READ_ONLY = frozenset({"context", "scan"})
_GRAPH_KINDS_REFUSED = frozenset({"index_docs", "pr_risk", "recall_docs"})


def test_every_registered_tool_is_classified() -> None:
    """Each registered tool sits in exactly one class, so a new one fails here until classified."""
    from lemoncrow.gateway.adapters import mcp_server
    from lemoncrow.gateway.adapters.mcp.broker_policy import GRAPH_READ_ONLY_KINDS

    classes = (mcp_server._BROKER_READ_ONLY, _BROKER_DENIED, _BROKER_DENIED_UNTIL_SHOWN_READ_ONLY)
    registered = set(mcp_server.TOOLS) | {"tool"}  # `tool` is the broker itself, not a TOOLS key
    assert sorted(name for name in registered if sum(name in cls for cls in classes) != 1) == []
    assert set().union(*classes) == registered
    # `graph` is allowed kind by kind, and every kind is classified too.
    assert GRAPH_READ_ONLY_KINDS | _GRAPH_KINDS_REFUSED == mcp_server._GRAPH_KINDS
    assert not GRAPH_READ_ONLY_KINDS & _GRAPH_KINDS_REFUSED


@pytest.mark.parametrize("name", sorted(_BROKER_DENIED | _BROKER_DENIED_UNTIL_SHOWN_READ_ONLY))
def test_broker_refuses_execution_write_and_network_tools(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Refused by `call` AND absent from `search`, advertised or not.

    search must never surface something call would then refuse. The handler is
    stubbed, so a broken guard shows up as a call that returned rather than as a
    shell that ran.
    """
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    if name in mcp_server.TOOLS:
        _stub_handler(monkeypatch, name)
    with pytest.raises(mcp_server._ToolArgumentError, match="not reachable through the broker"):
        _broker({"action": "call", "name": name, "arguments": {}})

    found = {match["name"] for match in _broker({"action": "search", "query": name})["matches"]}
    assert name not in found


@pytest.mark.parametrize(
    "arguments",
    [
        {"kind": "index_docs"},
        {"kind": "recall_docs", "query": "design"},
        {"kind": "pr_risk", "paths": ["a.py"]},
        {"kind": "dead_code", "enable": True},
    ],
    ids=["index_docs", "recall_docs", "pr_risk", "enable"],
)
def test_broker_refuses_graph_write_kinds(monkeypatch: pytest.MonkeyPatch, arguments: dict) -> None:
    """`graph` runs through the broker for the kinds that only read; the rest are refused by kind."""
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    _stub_handler(monkeypatch, "graph")
    with pytest.raises(mcp_server._ToolArgumentError, match="not reachable through the broker"):
        _broker({"action": "call", "name": "graph", "arguments": arguments})

    # Per kind, not per tool: a read-only kind still runs.
    assert _broker({"action": "call", "name": "graph", "arguments": {"kind": "dead_code"}})["called"] == "graph"


def test_broker_refusal_names_read_only_alternatives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over JSON-RPC, a refusal is an argument error that says where to go instead; the session carries on."""
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    _stub_handler(monkeypatch, "bash")

    def call_broker(request_id: int, arguments: dict) -> dict:
        response = mcp_server._handle(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "tool", "arguments": arguments},
            }
        )
        assert isinstance(response, dict)
        return response

    refused = call_broker(1, {"action": "call", "name": "bash", "arguments": {"command": "true"}})
    assert "Read-only alternatives: read, code_search, relations, code_query." in refused["error"]["message"]

    followup = call_broker(2, {"action": "search", "query": "blame"})
    assert "error" not in followup
    assert "blame" in str(followup["result"])


def test_broker_rejects_an_unregistered_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    with pytest.raises(mcp_server._ToolArgumentError, match="unknown tool"):
        _broker({"action": "call", "name": "definitely_not_a_tool", "arguments": {}})
