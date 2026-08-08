"""One contract for saying how complete a code-intel answer is.

Two objectives live in this subsystem and they are not interchangeable.
``code_search`` **ranks**: given a question, return the smallest sufficient
context. ``relations``, ``code_changes``, ``code_query`` and the file-graph
analytics **enumerate**: given a subject, return everything that matches.

A ranked result is a good answer to "where do I start" and a wrong answer to
"did I get them all" -- and once the tool name is out of view, the two are
indistinguishable. That ambiguity is not hypothetical: a reviewer reading a
ranked list as complete files a finding that says a symbol has one caller when
it has seven.

So every retrieval response states three things:

``objective``
    :data:`OBJECTIVE_RANKED` or :data:`OBJECTIVE_EXHAUSTIVE`.
``<subject>_count``
    What was found *before* any limit was applied.
``truncated``
    Whether the returned list is shorter than that count.

A consumer then evaluates one predicate -- exhaustive **and** not truncated --
instead of inferring completeness from which tool it happened to call. That is
exactly the difference between a reviewer being able to skip a grep replay and
having to guess.

**One field stays authoritative.** A surface that answers from stored results
rather than live ones can be complete about a subject it only partly examined,
and ``truncated`` cannot express that -- it speaks to the returned list, not to
what was looked at. Rather than adding a second thing every consumer must
remember to check, such a response reports :data:`OBJECTIVE_PARTIAL` and fails
the existing predicate on its own. ``coverage`` then says *how* partial.

An op absent from :data:`CODE_OP_OBJECTIVES` carries no ``objective`` field.
That is deliberate: the closed engine owns those code paths, and asserting an
objective we have not verified would manufacture the false confidence this
module exists to remove. Absent means unclassified, never "exhaustive".

Exhaustive is necessary, not sufficient
---------------------------------------

Complete is not the same as correct, and for anything that shows a callsite to
a human the difference decides whether the output is trustworthy. Both edge
stores behind these ops are keyed by *name*: ``call_edges`` records the callee
as raw dotted call text with no ``symbol_id``, and ``"references"`` records a
bare ``symbol_name``. So a change to a method called ``open`` enumerates every
``open()`` in the repository -- exhaustively, untruncated, and wrong. The rows
are real lines, which is exactly why a grep replay does not catch them.

The error is one-directional: name matching over-reports and never
under-reports. That is the safe direction for an exemption (no false
negatives) and the unsafe one for review output (confident false findings). So
every enumerative response also states :data:`CODE_OP_MATCH_KINDS` as
``match_kind``, and a consumer that surfaces rows to a human gates on
``objective == "exhaustive" and not truncated and match_kind == "resolved"``,
validating the rows itself whenever the answer is ``"name"``.

Today that value is ``"name"`` everywhere, guaranteed by the schema rather than
by convention -- there is no column any row could be resolved *by*. It is
stamped at the top level for that reason: it describes every row in the payload
because no two rows can currently disagree. When F9's resolution sidecar lands
and rows can differ, ``match_kind`` moves onto the row and the top-level value
becomes ``"mixed"``; a consumer already gating on ``== "resolved"`` stays
correct through that change without an edit.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CODE_OP_MATCH_KINDS",
    "CODE_OP_OBJECTIVES",
    "MATCH_NAME",
    "MATCH_RESOLVED",
    "OBJECTIVE_EXHAUSTIVE",
    "OBJECTIVE_PARTIAL",
    "OBJECTIVE_RANKED",
    "objective_for_coverage",
    "with_match_kind",
    "with_objective",
]

#: Scored best-effort. The ordering *is* the answer; the list is not the set.
OBJECTIVE_RANKED = "ranked"

#: Everything matching, up to a stated limit, with an exact pre-limit count.
OBJECTIVE_EXHAUSTIVE = "exhaustive"

#: Exhaustive over what was examined -- but what was examined is not everything.
#:
#: Only derived surfaces can be in this state. They answer from stored results,
#: so the question "was the whole subject looked at" is separate from "was the
#: whole answer returned", and ``truncated`` only ever spoke to the second.
#:
#: This exists because the gap was reachable. A clone query against a table
#: whose every row had been superseded returned ``count: 0`` with
#: ``objective: "exhaustive"`` and ``truncated: false`` -- passing the
#: completeness predicate while reporting that code has no duplicates, on the
#: strength of an answer where nothing current had been examined. The coverage
#: number said so, but the contract does not require reading it, and a
#: consumer honouring the documented predicate got a false negative. So the
#: field the contract *does* make authoritative carries it: below full
#: coverage the objective is no longer exhaustive.
OBJECTIVE_PARTIAL = "partial"

#: Engine-backed ``code`` ops we have evidence for. ``pattern`` and ``node`` are
#: deliberately absent -- their extraction lives in the closed engine and has
#: not been checked against an oracle, so they stay unclassified rather than
#: guessed.
CODE_OP_OBJECTIVES: dict[str, str] = {
    "search": OBJECTIVE_RANKED,
    "explore": OBJECTIVE_RANKED,
    "context": OBJECTIVE_RANKED,
    "centrality": OBJECTIVE_RANKED,
    "callers": OBJECTIVE_EXHAUSTIVE,
    "callees": OBJECTIVE_EXHAUSTIVE,
    "usages": OBJECTIVE_EXHAUSTIVE,
}


#: The edge text equalled the symbol's name. Over-reports, never under-reports.
MATCH_NAME = "name"

#: The edge was resolved to a symbol id. Nothing produces this until F9's
#: resolution sidecar lands; it exists so consumers can write the predicate
#: they will still want then.
MATCH_RESOLVED = "resolved"

#: How each enumerative op's edges were matched. Ranked ops are absent: their
#: answer is an ordering, not an edge set, so the question does not apply.
#: ``pattern`` and ``node`` are absent for the same reason they carry no
#: objective -- unclassified, never guessed.
CODE_OP_MATCH_KINDS: dict[str, str] = {
    "callers": MATCH_NAME,
    "callees": MATCH_NAME,
    "usages": MATCH_NAME,
}


def objective_for_coverage(coverage: float | None, superseded: int | None = None) -> str:
    """The objective a stored-result surface may claim about this answer.

    ``coverage is None`` means the surface reads live data, where the question
    does not arise. Otherwise the answer is :data:`OBJECTIVE_PARTIAL` when
    either the subject was not fully examined (*coverage* below 1.0) or some
    stored rows were dropped as superseded.

    Both conditions are checked rather than one inferred from the other. In
    practice they move together -- a row is superseded because its symbol's
    content changed, which is the same fact that lowers coverage -- but
    "rows were discarded" and "the subject was under-examined" are separate
    claims, and an answer that quietly lost rows is not exhaustive whatever the
    coverage figure says.
    """
    if coverage is None:
        return OBJECTIVE_EXHAUSTIVE
    if coverage < 1.0 or (superseded or 0) > 0:
        return OBJECTIVE_PARTIAL
    return OBJECTIVE_EXHAUSTIVE


def with_objective(payload: dict[str, Any], objective: str) -> dict[str, Any]:
    """Stamp *payload* in place and return it."""
    payload["objective"] = objective
    return payload


def with_match_kind(payload: dict[str, Any], match_kind: str) -> dict[str, Any]:
    """Stamp *payload* in place and return it."""
    payload["match_kind"] = match_kind
    return payload
