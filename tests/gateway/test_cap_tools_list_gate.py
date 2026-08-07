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


@pytest.mark.parametrize("name", ["relations", "graph"])
def test_broker_calls_tools_that_are_hidden_under_the_core_profile(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Regression: `relations` is in BOTH _CORE_MCP_TOOLS and HIDDEN_LLM_TOOLS.

    The old guard refused it as "already exposed" because it is in the core
    profile, while HIDDEN_LLM_TOOLS meant nothing ever advertised it -- so it
    was unreachable by every route. That left `statusline_segment` as the only
    tool the broker could reach.
    """
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    assert name not in {tool["name"] for tool in _list()}
    _stub_handler(monkeypatch, name)

    assert _broker({"action": "call", "name": name, "arguments": {"op": "callers"}}) == {
        "called": name,
        "args": {"op": "callers"},
    }


def test_broker_search_returns_hidden_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """`search` used to filter to *visible* tools, so it could never match."""
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    found = {match["name"] for match in _broker({"action": "search", "query": ""})["matches"]}
    assert found
    assert "blame" in {match["name"] for match in _broker({"action": "search", "query": "blame"})["matches"]}


def test_broker_search_never_returns_an_advertised_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    advertised = {tool["name"] for tool in _list()}
    found = {match["name"] for match in _broker({"action": "search", "query": ""})["matches"]}
    assert not (found & advertised)


def test_broker_refuses_an_already_advertised_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    assert "read" in {tool["name"] for tool in _list()}
    with pytest.raises(mcp_server._ToolArgumentError, match="already exposed"):
        _broker({"action": "call", "name": "read", "arguments": {}})


@pytest.mark.parametrize("name", sorted({"agent", "codemod", "mcp", "sql", "tool", "workflow"}))
def test_broker_deny_list_holds(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Denied tools are refused by `call` AND absent from `search`.

    search must never surface something call would then refuse.
    """
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    with pytest.raises(mcp_server._ToolArgumentError, match="not reachable through the broker"):
        _broker({"action": "call", "name": name, "arguments": {}})

    found = {match["name"] for match in _broker({"action": "search", "query": name})["matches"]}
    assert name not in found


def test_broker_rejects_an_unregistered_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters import mcp_server

    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "core")
    with pytest.raises(mcp_server._ToolArgumentError, match="unknown tool"):
        _broker({"action": "call", "name": "definitely_not_a_tool", "arguments": {}})
