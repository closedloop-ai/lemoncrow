"""Typed call-graph helpers for routed code-intel traversal."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable
from typing import Any

from lemoncrow.core.capabilities.code_context_contract import (
    CallGraphDataStatus,
    CallGraphDirection,
    CallGraphEdge,
    CallGraphNode,
    CallGraphTraversalResult,
)


def summarize_symbol(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the cheapest useful target symbol fields for call-graph responses."""

    keys = [
        "symbol_id",
        "symbol_name",
        "qualified_name",
        "file_path",
        "kind",
        "signature",
        "start_line",
        "end_line",
        "language",
        "provenance",
    ]
    return {key: payload[key] for key in keys if key in payload}


def traverse_call_graph(
    target: dict[str, Any],
    *,
    direction: CallGraphDirection,
    depth: int,
    limit: int,
    lookup_neighbors: Callable[[str], list[CallGraphNode] | None],
    snapshot: bool = False,
    neighbor_cap: int | None = None,
) -> CallGraphTraversalResult:
    """Traverse routed callers/callees with cycle-safe breadth-first expansion.

    Symbols past *limit* are still counted, though never kept or expanded, so the
    result's ``related_total`` is taken before the limit. At depth 1 every
    neighbour of the target is seen and the count is exact; past depth 1 a cut
    walk never expands what it dropped, so the count is a lower bound.

    *neighbor_cap* is the lookup's own row ceiling. A lookup that returns that
    many rows may have stopped early, so its neighbour set is treated as cut.
    """

    target_symbol_id = str(target["symbol_id"])
    queue: deque[tuple[str, int]] = deque([(target_symbol_id, 1)])
    visited: set[str] = {target_symbol_id}
    nodes_by_id: dict[str, CallGraphNode] = {}
    related_ids: set[str] = set()
    edge_keys: set[tuple[str, str, int]] = set()
    edges: list[CallGraphEdge] = []
    truncated = False
    lookup_capped = False

    while queue:
        current_symbol_id, current_depth = queue.popleft()
        neighbors = lookup_neighbors(current_symbol_id)
        if neighbors is None:
            return CallGraphTraversalResult(
                nodes=[],
                edges=[],
                # Nothing could be looked up, so zero is not a count.
                related_symbol_ids=[],
                related_total_exact=False,
                truncated=False,
                data_status="unavailable",
                message="routed call edge data is unavailable",
                snapshot=None,
            )
        if neighbor_cap is not None and len(neighbors) >= neighbor_cap:
            lookup_capped = True
            truncated = True
        for neighbor in neighbors:
            if direction == "callers":
                edge_key = (neighbor.symbol_id, current_symbol_id, current_depth)
                edge = CallGraphEdge(
                    caller_symbol_id=neighbor.symbol_id,
                    callee_symbol_id=current_symbol_id,
                    depth=current_depth,
                )
            else:
                edge_key = (current_symbol_id, neighbor.symbol_id, current_depth)
                edge = CallGraphEdge(
                    caller_symbol_id=current_symbol_id,
                    callee_symbol_id=neighbor.symbol_id,
                    depth=current_depth,
                )
            if edge_key not in edge_keys:
                edge_keys.add(edge_key)
                edges.append(edge)
            if neighbor.symbol_id == target_symbol_id:
                continue
            related_ids.add(neighbor.symbol_id)
            if neighbor.symbol_id not in nodes_by_id:
                if len(nodes_by_id) >= limit:
                    truncated = True
                    continue
                nodes_by_id[neighbor.symbol_id] = neighbor
            if current_depth < depth and neighbor.symbol_id not in visited:
                visited.add(neighbor.symbol_id)
                queue.append((neighbor.symbol_id, current_depth + 1))

    ordered_nodes = sorted(nodes_by_id.values(), key=lambda item: (item.file_path, item.start_line, item.symbol_id))
    ordered_edges = sorted(edges, key=lambda item: (item.depth, item.caller_symbol_id, item.callee_symbol_id))
    data_status: CallGraphDataStatus = "empty" if not ordered_edges else "available"
    snapshot_payload = (
        build_call_graph_snapshot(
            target=target,
            direction=direction,
            depth=depth,
            nodes=ordered_nodes,
            edges=ordered_edges,
        )
        if snapshot
        else None
    )
    return CallGraphTraversalResult(
        nodes=ordered_nodes,
        edges=ordered_edges,
        related_symbol_ids=sorted(related_ids),
        related_total_exact=not lookup_capped and (depth <= 1 or not truncated),
        truncated=truncated,
        data_status=data_status,
        message=None if ordered_edges else "no related call edges were found",
        snapshot=snapshot_payload,
    )


def build_call_graph_payload(
    target: dict[str, Any],
    *,
    direction: CallGraphDirection,
    depth: int,
    result: CallGraphTraversalResult,
) -> dict[str, Any]:
    """Shape the public callers/callees response payload.

    ``related_count`` is the rows returned. ``related_total`` is the distinct
    related symbols found before ``limit``; ``related_total_exact`` says whether
    that count is exact or a lower bound.
    """

    return {
        "target": summarize_symbol(target),
        "direction": direction,
        "depth": depth,
        "related": [item.model_dump(mode="json") for item in result.nodes],
        "edges": [item.model_dump(mode="json") for item in result.edges],
        "related_count": len(result.nodes),
        "related_total": result.related_total,
        "related_total_exact": result.related_total_exact,
        "edge_count": len(result.edges),
        "truncated": result.truncated,
        "data_status": result.data_status,
        "message": result.message,
        "snapshot": result.snapshot,
    }


def build_call_graph_snapshot(
    *,
    target: dict[str, Any],
    direction: CallGraphDirection,
    depth: int,
    nodes: list[CallGraphNode],
    edges: list[CallGraphEdge],
) -> dict[str, Any]:
    """Build cheap, deterministic snapshot metadata without persistence side effects."""

    digest = hashlib.sha256(
        json.dumps(
            {
                "target_symbol_id": target["symbol_id"],
                "direction": direction,
                "depth": depth,
                "edges": [edge.model_dump(mode="json") for edge in edges],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "snapshot_id": digest,
        "direction": direction,
        "depth": depth,
        "target_symbol_id": target["symbol_id"],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


__all__ = [
    "CallGraphDirection",
    "CallGraphEdge",
    "CallGraphNode",
    "CallGraphTraversalResult",
    "build_call_graph_payload",
    "build_call_graph_snapshot",
    "summarize_symbol",
    "traverse_call_graph",
]
