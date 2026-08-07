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

An op absent from :data:`CODE_OP_OBJECTIVES` carries no ``objective`` field.
That is deliberate: the closed engine owns those code paths, and asserting an
objective we have not verified would manufacture the false confidence this
module exists to remove. Absent means unclassified, never "exhaustive".
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CODE_OP_OBJECTIVES",
    "OBJECTIVE_EXHAUSTIVE",
    "OBJECTIVE_RANKED",
    "with_objective",
]

#: Scored best-effort. The ordering *is* the answer; the list is not the set.
OBJECTIVE_RANKED = "ranked"

#: Everything matching, up to a stated limit, with an exact pre-limit count.
OBJECTIVE_EXHAUSTIVE = "exhaustive"

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


def with_objective(payload: dict[str, Any], objective: str) -> dict[str, Any]:
    """Stamp *payload* in place and return it."""
    payload["objective"] = objective
    return payload
