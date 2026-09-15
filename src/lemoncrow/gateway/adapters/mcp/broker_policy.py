"""What the MCP ``tool`` broker may run (PRD-739 FR4). Fork-only.

The broker calls a registered tool by exact name, and the review agents that
hold it read diffs written by PR authors. So it runs only tools that read:
anything not on :data:`BROKER_READ_ONLY` is refused, and a tool registered later
stays unreachable until it is classified (``test_cap_tools_list_gate`` fails
until it is).

Refused because the code shows a write, execute or network path:

* ``scan`` runs the ast-grep binary in a subprocess.
* ``context`` records the task on the session ledger.
* ``graph kind=index_docs`` writes the design-doc store; ``recall_docs`` embeds
  its query through the configured embedder (the OpenAI one posts to the
  network) and creates the store schema on connect; ``pr_risk`` folds each
  changed file into the machine-wide semantic file index. ``enable`` only
  switches ``index_docs`` indexing on, and is refused outright.

``statusline_segment`` is allowed although it refreshes its own statusline
sidecar file: no argument reaches that path or its content.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BROKER_READ_ONLY: frozenset[str] = frozenset(
    {
        "blame",
        "code_changes",
        "code_coverage_check",
        "code_query",
        "code_search",
        "graph",
        "grep",
        "orient",
        "read",
        "relations",
        "search",
        "statusline_segment",
    }
)

# `graph` runs only these kinds, which read the index or git history.
GRAPH_READ_ONLY_KINDS: frozenset[str] = frozenset(
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

_ALTERNATIVES = "Read-only alternatives: read, code_search, relations, code_query."


def broker_refusal(name: str, arguments: Mapping[str, Any]) -> str | None:
    """Why the broker must not run *name* with *arguments*; ``None`` when it may."""
    if name not in BROKER_READ_ONLY:
        return f"{name!r} is not reachable through the broker, which runs read-only tools only. {_ALTERNATIVES}"
    if name == "graph":
        if "enable" in arguments:
            return f"graph `enable` is not reachable through the broker: it switches on indexing. {_ALTERNATIVES}"
        kind = arguments.get("kind", "blast_radius")
        if not isinstance(kind, str) or kind not in GRAPH_READ_ONLY_KINDS:
            return (
                f"graph kind={kind!r} is not reachable through the broker, which runs only the kinds "
                f"{', '.join(sorted(GRAPH_READ_ONLY_KINDS))}. {_ALTERNATIVES}"
            )
    return None
