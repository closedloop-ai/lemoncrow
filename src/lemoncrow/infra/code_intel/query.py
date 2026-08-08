"""F5 -- a constrained predicate API over the engine's tables.

Deliberately not Cypher, and deliberately not raw SQL. The questions this has to
answer are narrow and repetitive -- "functions matching X with no callers",
"files importing Y ordered by centrality" -- and a query language would be a
parser to maintain plus an injection surface to defend, for the same answers.

So: a whitelist. ``select`` names one of six row sources, ``where`` is a flat
mapping of ``field`` or ``field_<operator>`` to a value, and both sides are
checked against a table declared in this module before any SQL is built.

**Every identifier comes from the whitelist and every value is bound.** No
caller-supplied string is ever concatenated into a statement. That matters more
here than in most modules: this is a code-search surface reachable by
model-generated input, so injection is an expected input rather than a
theoretical one. An unknown field is an error naming the allowed set, never a
silently-ignored filter -- a dropped predicate returns *more* rows than asked
for, and a caller reading a filtered list has no way to tell.

The call-graph sources (``callers``, ``callees``, ``references``) are name-keyed
in the engine's schema, so results there carry ``match_kind: "name"`` for the
same reason :mod:`lemoncrow.infra.code_intel.change_impact` does.

``clones`` is the one source backed by the sidecar rather than the engine, and
therefore the one that can be *absent* rather than merely empty: nothing builds
it until ``lc code clones`` runs. It raises
:class:`~lemoncrow.infra.code_intel.clones.ClonesStale` in that case, and when
the engine has reindexed past it, instead of returning the zero rows that would
read as "this code has no duplicates".
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.clones import (
    ClonesStale,
    current_symbol_hashes,
    open_clone_table,
    signature_coverage,
)
from lemoncrow.infra.code_intel.completeness import MATCH_NAME, OBJECTIVE_EXHAUSTIVE
from lemoncrow.infra.code_intel.store import CodeIntelStore

__all__ = [
    "MAX_LIMIT",
    "MAX_SCAN",
    "ORDER_BY",
    "SELECTS",
    "QueryError",
    "QueryResult",
    "code_query",
    "describe_schema",
]

#: Hard ceiling on returned rows, whatever the caller asks for.
MAX_LIMIT = 1_000

#: Hard ceiling on rows read before ordering. Ordering by centrality or caller
#: count cannot be pushed into SQL (the scores live in a different database), so
#: it happens in Python over the scanned set. The cap keeps that bounded, and
#: ``scan_capped`` on the result says when it bit.
MAX_SCAN = 10_000

#: Longest-first, because field names contain underscores too: ``file_path_like``
#: must resolve as ``file_path`` + ``like``, never ``file_path_like`` + nothing.
_OPERATORS: tuple[tuple[str, str], ...] = (
    ("regex", "REGEXP"),
    ("like", "LIKE"),
    ("not", "!="),
    ("gte", ">="),
    ("lte", "<="),
    ("gt", ">"),
    ("lt", "<"),
    ("in", "IN"),
)

_NUMERIC_OPERATORS = frozenset({">=", "<=", ">", "<"})
_TEXT_OPERATORS = frozenset({"LIKE", "REGEXP"})

#: A pattern longer than this is rejected rather than compiled. Regex runs once
#: per scanned row, and an unbounded pattern from model-generated input is a
#: denial-of-service waiting to happen.
_MAX_PATTERN_CHARS = 512

ORDER_BY: tuple[str, ...] = ("name", "centrality", "callers")


class QueryError(ValueError):
    """The query was rejected before any SQL ran."""


@dataclass(frozen=True)
class _Select:
    """One queryable row source and everything a caller is allowed to say about it."""

    name: str
    database: str
    table: str
    columns: tuple[str, ...]
    fields: Mapping[str, str]
    numeric: frozenset[str] = field(default_factory=frozenset)
    name_column: str = ""
    name_keyed: bool = False
    #: SQL fragment used to order the scan when the caller names no `order_by`.
    #: Defaults to `name_column`, which is right for row sources whose interest
    #: is uniform. It is wrong for any source carrying a score: the scan is
    #: capped at MAX_SCAN and the result at `limit`, so an alphabetical default
    #: makes the highest-scoring rows unreachable. Whitelist-owned text, never
    #: caller input.
    default_order: str = ""
    #: Selected for internal verification and stripped before the rows are
    #: returned. Sidecar rows carry the content hashes they were derived from so
    #: each one can be re-checked against the live index; the caller wants the
    #: verdict, not the bookkeeping.
    internal_columns: tuple[str, ...] = ()

    @property
    def quoted_table(self) -> str:
        # "references" is a SQL keyword; the engine quotes it and so must we.
        return f'"{self.table}"' if self.table == "references" else self.table


_SYMBOLS = _Select(
    name="symbols",
    database="code",
    table="symbols",
    columns=(
        "symbol_id",
        "file_path",
        "language",
        "symbol_name",
        "qualified_name",
        "kind",
        "signature",
        "start_line",
        "end_line",
        "parent_symbol",
    ),
    fields={
        "name": "symbol_name",
        "symbol_name": "symbol_name",
        "qualified_name": "qualified_name",
        "kind": "kind",
        "language": "language",
        "file_path": "file_path",
        "parent_symbol": "parent_symbol",
        "start_line": "start_line",
        "end_line": "end_line",
    },
    numeric=frozenset({"start_line", "end_line"}),
    name_column="symbol_name",
)

_CALL_EDGE_COLUMNS = (
    "caller_symbol_name",
    "caller_qualified_name",
    "caller_file_path",
    "caller_start_line",
    "caller_end_line",
    "callee_name",
    "callee_short_name",
    "call_line",
    "call_column",
)

_CALL_EDGE_FIELDS = {
    "caller": "caller_symbol_name",
    "caller_symbol_name": "caller_symbol_name",
    "caller_qualified_name": "caller_qualified_name",
    "caller_file_path": "caller_file_path",
    "callee": "callee_name",
    "callee_name": "callee_name",
    "callee_short_name": "callee_short_name",
    "call_line": "call_line",
}

_CALLERS = _Select(
    name="callers",
    database="intel",
    table="call_edges",
    columns=_CALL_EDGE_COLUMNS,
    fields=_CALL_EDGE_FIELDS,
    numeric=frozenset({"call_line"}),
    name_column="caller_symbol_name",
    name_keyed=True,
)

_CALLEES = _Select(
    name="callees",
    database="intel",
    table="call_edges",
    columns=_CALL_EDGE_COLUMNS,
    fields=_CALL_EDGE_FIELDS,
    numeric=frozenset({"call_line"}),
    name_column="callee_name",
    name_keyed=True,
)

_IMPORTERS = _Select(
    name="importers",
    database="code",
    table="imports",
    columns=("source_file", "raw_import", "target_file"),
    fields={
        "source_file": "source_file",
        "raw_import": "raw_import",
        "target_file": "target_file",
    },
    name_column="source_file",
)

_REFERENCES = _Select(
    name="references",
    database="intel",
    table="references",
    columns=(
        "symbol_name",
        "file_path",
        "line",
        "column",
        "end_column",
        "enclosing_symbol_name",
        "enclosing_qualified_name",
        "snippet",
    ),
    fields={
        "name": "symbol_name",
        "symbol_name": "symbol_name",
        "file_path": "file_path",
        "line": "line",
        "enclosing": "enclosing_symbol_name",
        "enclosing_symbol_name": "enclosing_symbol_name",
        "enclosing_qualified_name": "enclosing_qualified_name",
    },
    numeric=frozenset({"line"}),
    name_column="symbol_name",
    name_keyed=True,
)

# F6. The only select backed by the sidecar rather than the engine, and the only
# one whose rows we produced ourselves -- so it is the only one that can be
# *absent* rather than merely empty. `code_query` raises for that case instead of
# returning zero rows, because "no clones recorded" and "nobody has looked" are
# different answers and only one of them is about the code.
_CLONES = _Select(
    name="clones",
    database="sidecar",
    table="symbol_clones",
    columns=(
        "qualified_name_a",
        "qualified_name_b",
        "file_path_a",
        "file_path_b",
        "token_count_a",
        "token_count_b",
        "jaccard",
    ),
    fields={
        "name": "qualified_name_a",
        "qualified_name_a": "qualified_name_a",
        "qualified_name_b": "qualified_name_b",
        "file_path_a": "file_path_a",
        "file_path_b": "file_path_b",
        "jaccard": "jaccard",
        "token_count_a": "token_count_a",
        "token_count_b": "token_count_b",
    },
    numeric=frozenset({"jaccard", "token_count_a", "token_count_b"}),
    name_column="qualified_name_a",
    # The score is the reason this table exists. Ordering by name instead sent
    # the default `limit=50` at the alphabetically-first of ~1,900 pairs and
    # reported `truncated: true`, so the strongest clones were unreachable
    # unless a caller already knew to pass `jaccard_gte` -- and `idx_symbol_
    # clones_score` was an index built for an ordering nothing ever issued.
    default_order="jaccard DESC, qualified_name_a, qualified_name_b",
    internal_columns=("symbol_a", "symbol_b", "content_hash_a", "content_hash_b"),
)

SELECTS: dict[str, _Select] = {
    select.name: select for select in (_SYMBOLS, _CALLERS, _CALLEES, _IMPORTERS, _REFERENCES, _CLONES)
}


@dataclass(frozen=True)
class QueryResult:
    """Rows plus everything needed to judge them."""

    select: str
    where: dict[str, Any]
    order_by: str | None
    limit: int
    rows: tuple[dict[str, Any], ...]
    scanned: int
    truncated: bool
    scan_capped: bool
    match_kind: str | None
    engine_index_version: int
    #: Sidecar-backed selects only. What fraction of the repo's symbols the
    #: stored answer actually examined, and how many rows were dropped because
    #: their source changed since. Absent for engine-backed selects, which read
    #: the index live and are current by definition.
    coverage: float | None = None
    stale_symbols: int | None = None
    superseded_rows: int | None = None
    built_from_index_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            # A filter, not a ranker. `order_by=centrality` sorts what matched;
            # it does not decide what matched, and `scanned` / `scan_capped`
            # bound the claim.
            "objective": OBJECTIVE_EXHAUSTIVE,
            "select": self.select,
            "where": self.where,
            "order_by": self.order_by,
            "limit": self.limit,
            "rows": list(self.rows),
            "count": len(self.rows),
            "scanned": self.scanned,
            "truncated": self.truncated,
            "scan_capped": self.scan_capped,
            "engine_index_version": self.engine_index_version,
        }
        if self.match_kind is not None:
            payload["match_kind"] = self.match_kind
        if self.coverage is not None:
            # Reading the *absence* of a row correctly needs this: at coverage
            # 1.0 "not reported" means measured-and-none; below it, some symbols
            # changed after the last build and were never examined.
            payload["coverage"] = round(self.coverage, 4)
            payload["stale_symbols"] = self.stale_symbols
            payload["superseded_rows"] = self.superseded_rows
            payload["built_from_index_version"] = self.built_from_index_version
        return payload


def describe_schema() -> dict[str, Any]:
    """The whitelist, as data -- so a caller can discover it without guessing."""
    return {
        "selects": {
            name: {
                "columns": list(select.columns),
                "fields": sorted(select.fields),
                "numeric_fields": sorted(select.numeric),
                "name_keyed": select.name_keyed,
            }
            for name, select in SELECTS.items()
        },
        "operators": [suffix for suffix, _ in _OPERATORS],
        "order_by": list(ORDER_BY),
        "max_limit": MAX_LIMIT,
    }


@lru_cache(maxsize=256)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _regexp(pattern: str, value: object) -> bool:
    if value is None:
        return False
    return _compiled(pattern).search(str(value)) is not None


@dataclass(frozen=True)
class _Predicate:
    column: str
    operator: str
    value: Any
    source_key: str


def _split_key(select: _Select, key: str) -> tuple[str, str]:
    """``"file_path_like"`` -> ``("file_path", "LIKE")``.

    Exact field names win over suffix parsing, so a column that happens to end
    in an operator name is still addressable directly.
    """
    if key in select.fields:
        return select.fields[key], "="
    for suffix, operator in _OPERATORS:
        tail = f"_{suffix}"
        if key.endswith(tail):
            head = key[: -len(tail)]
            if head in select.fields:
                return select.fields[head], operator
    allowed = ", ".join(sorted(select.fields))
    raise QueryError(f"unknown field {key!r} for select={select.name!r}; allowed fields: {allowed}")


def _build_predicates(select: _Select, where: Mapping[str, Any]) -> list[_Predicate]:
    predicates: list[_Predicate] = []
    for key in sorted(where):
        if not isinstance(key, str):  # pragma: no cover - defensive; dict keys arrive as str
            raise QueryError(f"where keys must be strings, got {type(key).__name__}")
        column, operator = _split_key(select, key)
        value = where[key]
        field_name = key
        for suffix, candidate in _OPERATORS:
            if candidate == operator and key.endswith(f"_{suffix}"):
                field_name = key[: -len(suffix) - 1]
                break

        if operator == "IN":
            if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
                raise QueryError(f"{key!r} expects a list of values")
            items = list(value)
            if not items:
                raise QueryError(f"{key!r} was given an empty list; that matches nothing by construction")
            for item in items:
                _reject_container(key, item)
            predicates.append(_Predicate(column, operator, items, key))
            continue

        _reject_container(key, value)

        if operator in _NUMERIC_OPERATORS and field_name not in select.numeric:
            numeric = ", ".join(sorted(select.numeric)) or "(none)"
            raise QueryError(f"{key!r} compares a non-numeric field; numeric fields: {numeric}")
        if operator in _TEXT_OPERATORS:
            if not isinstance(value, str):
                raise QueryError(f"{key!r} expects a string pattern")
            if len(value) > _MAX_PATTERN_CHARS:
                raise QueryError(f"{key!r} pattern exceeds {_MAX_PATTERN_CHARS} characters")
            if operator == "REGEXP":
                try:
                    _compiled(value)
                except re.error as exc:
                    raise QueryError(f"{key!r} is not a valid regular expression: {exc}") from exc
        predicates.append(_Predicate(column, operator, value, key))
    return predicates


def _reject_container(key: str, value: object) -> None:
    """A dict or nested list as a bound value is never a legitimate predicate.

    It is, however, exactly the shape a caller reaches for when trying to smuggle
    structure past a flat whitelist, so it is refused by name rather than
    stringified into something that would bind cleanly and match nothing.
    """
    if isinstance(value, (dict, list, tuple, set)):
        raise QueryError(f"{key!r} expects a scalar value, got {type(value).__name__}")


def _where_clause(predicates: Sequence[_Predicate]) -> tuple[str, list[Any]]:
    """Compile predicates to SQL. Columns come from the whitelist; values bind."""
    fragments: list[str] = []
    params: list[Any] = []
    for predicate in predicates:
        if predicate.operator == "IN":
            placeholders = ",".join("?" * len(predicate.value))
            fragments.append(f"{predicate.column} IN ({placeholders})")
            params.extend(predicate.value)
        elif predicate.operator == "REGEXP":
            fragments.append(f"{predicate.column} REGEXP ?")
            params.append(predicate.value)
        else:
            fragments.append(f"{predicate.column} {predicate.operator} ?")
            params.append(predicate.value)
    return " AND ".join(fragments), params


def _centrality_scores(store: CodeIntelStore) -> dict[str, float]:
    return {row.name_key: row.score for row in store.centrality()}


def _caller_counts(store: CodeIntelStore) -> dict[str, int]:
    """``callee name -> call sites``, counting both the dotted and short forms."""
    conn = store.intel
    repo_id = store.repo_id_or_none()
    if conn is None or repo_id is None:
        return {}
    counts: dict[str, int] = {}
    for column in ("callee_name", "callee_short_name"):
        rows = conn.execute(
            f"SELECT {column} AS key, COUNT(*) AS n FROM call_edges WHERE repo_id = ? GROUP BY {column}",
            (repo_id,),
        )
        for row in rows:
            key = str(row["key"] or "")
            if key:
                counts[key] = max(counts.get(key, 0), int(row["n"]))
    return counts


def code_query(
    select: str = "symbols",
    where: Mapping[str, Any] | None = None,
    order_by: str | None = None,
    limit: int = 50,
    repo_root: Path | str = ".",
) -> QueryResult:
    """Run a whitelisted predicate query over the engine's tables.

    Raises :class:`QueryError` for anything not on the whitelist -- an unknown
    select, an unknown field, an operator applied to the wrong column type, a
    malformed regex, or a container where a scalar belongs. Rejecting is the
    point: a silently-dropped predicate widens the result set, and a caller
    reading a list it believes was filtered cannot detect that.
    """
    spec = SELECTS.get(select)
    if spec is None:
        raise QueryError(f"unknown select {select!r}; allowed: {', '.join(sorted(SELECTS))}")
    if order_by is not None and order_by not in ORDER_BY:
        raise QueryError(f"unknown order_by {order_by!r}; allowed: {', '.join(ORDER_BY)}")
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise QueryError(f"limit must be an integer, got {limit!r}") from exc
    limit = max(1, min(limit, MAX_LIMIT))

    predicates = _build_predicates(spec, where or {})

    with CodeIntelStore(repo_root) as store:
        index_version = store.engine_state("index_version")
        sidecar_conn: sqlite3.Connection | None = None
        conn: sqlite3.Connection | None
        built_from: int | None = None
        coverage: float | None = None
        stale_symbols: int | None = None
        superseded_rows: int | None = None
        if spec.database == "sidecar":
            # Raises only when nothing ever built the table -- the one case where
            # zero rows would answer "does this code have duplicates" with "no"
            # on the strength of never having looked. A reindex is *not* an
            # error: each row is re-checked below against the content it was
            # measured on, so staleness costs coverage rather than the answer.
            sidecar_conn, built_from = open_clone_table(repo_root)
            conn = sidecar_conn
        elif spec.database == "code":
            conn = store.code
        else:
            conn = store.intel
        repo_id = store.repo_id_or_none()
        if conn is None or repo_id is None:
            if sidecar_conn is not None:
                sidecar_conn.close()
            return QueryResult(
                select=select,
                where=dict(where or {}),
                order_by=order_by,
                limit=limit,
                rows=(),
                scanned=0,
                truncated=False,
                scan_capped=False,
                match_kind=MATCH_NAME if spec.name_keyed else None,
                engine_index_version=index_version,
            )

        try:
            conn.create_function("regexp", 2, _regexp, deterministic=True)
            clause, params = _where_clause(predicates)
            selected = spec.columns + spec.internal_columns
            sql = f"SELECT {', '.join(selected)} FROM {spec.quoted_table} WHERE repo_id = ?"
            bound: list[Any] = [repo_id]
            if clause:
                sql += f" AND {clause}"
                bound.extend(params)
            sql += f" ORDER BY {spec.default_order or spec.name_column} LIMIT ?"
            bound.append(MAX_SCAN)

            try:
                rows = [dict(row) for row in conn.execute(sql, bound)]
            except sqlite3.OperationalError as exc:
                if sidecar_conn is None:
                    raise
                # The sidecar is ours, so an unreadable one is our state to
                # report, not a transport failure for the caller to decode.
                # `load_clones` already converted this; the query path reaching
                # the same table had to as well.
                raise ClonesStale(f"the clone table is unreadable ({exc})") from exc

            if sidecar_conn is not None:
                # Every row is re-checked against the content it was measured
                # on, so what comes back is current by construction rather than
                # by assumption. Verification happens before `limit` is applied
                # -- filtering after would quietly return fewer rows than asked
                # for whenever a top-scoring pair had gone stale.
                recorded_rows = len(rows)
                hashes = current_symbol_hashes(repo_root)
                rows = [
                    row
                    for row in rows
                    if hashes.get(str(row.get("symbol_a"))) == row.get("content_hash_a")
                    and hashes.get(str(row.get("symbol_b"))) == row.get("content_hash_b")
                ]
                superseded_rows = recorded_rows - len(rows)
                examined, total = signature_coverage(repo_root)
                coverage = 1.0 if total == 0 else examined / total
                stale_symbols = total - examined
                rows = [{k: v for k, v in row.items() if k not in spec.internal_columns} for row in rows]

            scanned = len(rows)

            if order_by == "centrality":
                scores = _centrality_scores(store)
                rows.sort(
                    key=lambda row: (
                        -scores.get(str(row.get(spec.name_column) or ""), 0.0),
                        str(row.get(spec.name_column) or ""),
                    )
                )
            elif order_by == "callers":
                counts = _caller_counts(store)
                rows.sort(
                    key=lambda row: (
                        -counts.get(str(row.get(spec.name_column) or ""), 0),
                        str(row.get(spec.name_column) or ""),
                    )
                )
        finally:
            if sidecar_conn is not None:
                sidecar_conn.close()

    return QueryResult(
        select=select,
        where=dict(where or {}),
        order_by=order_by,
        limit=limit,
        rows=tuple(rows[:limit]),
        scanned=scanned,
        truncated=scanned > limit,
        scan_capped=scanned >= MAX_SCAN,
        match_kind=MATCH_NAME if spec.name_keyed else None,
        engine_index_version=index_version,
        coverage=coverage,
        stale_symbols=stale_symbols,
        superseded_rows=superseded_rows,
        built_from_index_version=built_from,
    )
