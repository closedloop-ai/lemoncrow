"""The constrained predicate API, and the injection attempts it must refuse (F5).

This is a code-search surface reachable by model-generated input, so the
injection cases below are not hypothetical hardening -- they are the expected
traffic. Two properties are asserted throughout: a hostile *value* binds and
matches nothing, and a hostile *key* is rejected outright rather than dropped.
A dropped predicate is the dangerous one: it widens the result set, and a caller
reading a list it believes was filtered has no way to notice.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.query import (
    MAX_LIMIT,
    ORDER_BY,
    SELECTS,
    QueryError,
    code_query,
    describe_schema,
)
from lemoncrow.infra.code_intel.store import CodeIntelStore

WorkspaceFactory = Callable[..., Path]

_FILES = [{"file_path": "src/a.py"}, {"file_path": "src/b.py"}, {"file_path": "tests/test_a.py"}]

_SYMBOLS = [
    {"file_path": "src/a.py", "symbol_name": "alpha", "kind": "function", "start_line": 10, "end_line": 20},
    {"file_path": "src/a.py", "symbol_name": "_private", "kind": "function", "start_line": 30, "end_line": 40},
    {"file_path": "src/b.py", "symbol_name": "Beta", "kind": "class", "start_line": 1, "end_line": 50},
    {"file_path": "tests/test_a.py", "symbol_name": "test_alpha", "kind": "function", "start_line": 5, "end_line": 8},
]

_EDGES = [
    {
        "caller_symbol_name": "test_alpha",
        "caller_file_path": "tests/test_a.py",
        "callee_name": "alpha",
        "call_line": 6,
    },
    {
        "caller_symbol_name": "Beta",
        "caller_file_path": "src/b.py",
        "callee_name": "alpha",
        "call_line": 12,
        "call_column": 8,
    },
    {
        "caller_symbol_name": "alpha",
        "caller_file_path": "src/a.py",
        "callee_name": "_private",
        "call_line": 15,
    },
]

_IMPORTS = [
    {"source_file": "src/b.py", "raw_import": "a", "target_file": "src/a.py"},
    {"source_file": "tests/test_a.py", "raw_import": "a", "target_file": "src/a.py"},
    {"source_file": "src/a.py", "raw_import": "os", "target_file": None},
]

_REFERENCES = [
    {"symbol_name": "alpha", "file_path": "src/b.py", "line": 12},
    {"symbol_name": "alpha", "file_path": "tests/test_a.py", "line": 6},
    {"symbol_name": "Beta", "file_path": "src/a.py", "line": 3},
]

_CENTRALITY = [
    {"name_key": "alpha", "score": 9.0},
    {"name_key": "Beta", "score": 4.0},
    {"name_key": "_private", "score": 1.0},
]


@pytest.fixture
def repo(make_workspace: WorkspaceFactory) -> Path:
    return make_workspace(
        files=_FILES,
        symbols=_SYMBOLS,
        imports=_IMPORTS,
        call_edges=_EDGES,
        references=_REFERENCES,
        centrality=_CENTRALITY,
        index_version=17,
    )


def _names(result: object, column: str = "symbol_name") -> set[str]:
    return {str(row[column]) for row in result.rows}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# selects
# --------------------------------------------------------------------------- #


def test_every_declared_select_is_queryable(repo: Path) -> None:
    for name in SELECTS:
        result = code_query(select=name, repo_root=repo)
        assert result.select == name
        assert result.engine_index_version == 17


def test_symbols_select_returns_the_whitelisted_columns(repo: Path) -> None:
    result = code_query(select="symbols", repo_root=repo)
    assert _names(result) == {"alpha", "_private", "Beta", "test_alpha"}
    assert set(result.rows[0]) == set(SELECTS["symbols"].columns)
    assert result.match_kind is None, "symbols are keyed by symbol_id, not by name"


def test_name_keyed_selects_say_so(repo: Path) -> None:
    for name in ("callers", "callees", "references"):
        assert code_query(select=name, repo_root=repo).match_kind == "name"


def test_importers_select_reads_the_imports_table(repo: Path) -> None:
    result = code_query(select="importers", where={"target_file": "src/a.py"}, repo_root=repo)
    assert _names(result, "source_file") == {"src/b.py", "tests/test_a.py"}


def test_references_select_filters_by_symbol(repo: Path) -> None:
    result = code_query(select="references", where={"name": "alpha"}, repo_root=repo)
    assert _names(result, "file_path") == {"src/b.py", "tests/test_a.py"}


def test_unknown_select_is_rejected(repo: Path) -> None:
    with pytest.raises(QueryError, match="unknown select"):
        code_query(select="symbols; DROP TABLE symbols", repo_root=repo)


# --------------------------------------------------------------------------- #
# predicates
# --------------------------------------------------------------------------- #


def test_equality_predicate(repo: Path) -> None:
    assert _names(code_query(select="symbols", where={"kind": "class"}, repo_root=repo)) == {"Beta"}


def test_field_aliases_resolve(repo: Path) -> None:
    by_alias = code_query(select="symbols", where={"name": "alpha"}, repo_root=repo)
    by_column = code_query(select="symbols", where={"symbol_name": "alpha"}, repo_root=repo)
    assert _names(by_alias) == _names(by_column) == {"alpha"}


def test_like_predicate(repo: Path) -> None:
    result = code_query(select="symbols", where={"file_path_like": "src/%"}, repo_root=repo)
    assert _names(result) == {"alpha", "_private", "Beta"}


def test_regex_predicate(repo: Path) -> None:
    result = code_query(select="symbols", where={"name_regex": "^_"}, repo_root=repo)
    assert _names(result) == {"_private"}


def test_not_predicate(repo: Path) -> None:
    result = code_query(select="symbols", where={"kind_not": "function"}, repo_root=repo)
    assert _names(result) == {"Beta"}


def test_in_predicate(repo: Path) -> None:
    result = code_query(select="symbols", where={"name_in": ["alpha", "Beta"]}, repo_root=repo)
    assert _names(result) == {"alpha", "Beta"}


def test_numeric_comparison_predicates(repo: Path) -> None:
    assert _names(code_query(select="symbols", where={"start_line_gt": 9}, repo_root=repo)) == {"alpha", "_private"}
    assert _names(code_query(select="symbols", where={"start_line_gte": 10}, repo_root=repo)) == {"alpha", "_private"}
    assert _names(code_query(select="symbols", where={"start_line_lt": 5}, repo_root=repo)) == {"Beta"}
    assert _names(code_query(select="symbols", where={"start_line_lte": 5}, repo_root=repo)) == {"Beta", "test_alpha"}


def test_predicates_combine_with_and(repo: Path) -> None:
    result = code_query(
        select="symbols",
        where={"kind": "function", "file_path_like": "src/%", "name_regex": "^_"},
        repo_root=repo,
    )
    assert _names(result) == {"_private"}


def test_callers_of_a_symbol(repo: Path) -> None:
    result = code_query(select="callers", where={"callee": "alpha"}, repo_root=repo)
    assert _names(result, "caller_symbol_name") == {"test_alpha", "Beta"}


def test_callees_of_a_caller(repo: Path) -> None:
    result = code_query(select="callees", where={"caller": "alpha"}, repo_root=repo)
    assert _names(result, "callee_name") == {"_private"}


# --------------------------------------------------------------------------- #
# rejection
# --------------------------------------------------------------------------- #


def test_unknown_field_is_rejected_and_names_the_allowed_set(repo: Path) -> None:
    with pytest.raises(QueryError) as excinfo:
        code_query(select="symbols", where={"nonexistent": "x"}, repo_root=repo)
    message = str(excinfo.value)
    assert "nonexistent" in message
    assert "qualified_name" in message, "the error must show what IS allowed"


def test_numeric_operator_on_a_text_field_is_rejected(repo: Path) -> None:
    with pytest.raises(QueryError, match="non-numeric"):
        code_query(select="symbols", where={"kind_gt": "function"}, repo_root=repo)


def test_in_requires_a_non_empty_list(repo: Path) -> None:
    with pytest.raises(QueryError, match="list of values"):
        code_query(select="symbols", where={"name_in": "alpha"}, repo_root=repo)
    with pytest.raises(QueryError, match="matches nothing"):
        code_query(select="symbols", where={"name_in": []}, repo_root=repo)


def test_invalid_regex_is_rejected_before_it_runs(repo: Path) -> None:
    with pytest.raises(QueryError, match="not a valid regular expression"):
        code_query(select="symbols", where={"name_regex": "([unclosed"}, repo_root=repo)


def test_oversized_regex_is_rejected(repo: Path) -> None:
    with pytest.raises(QueryError, match="exceeds"):
        code_query(select="symbols", where={"name_regex": "a" * 1000}, repo_root=repo)


def test_unknown_order_by_is_rejected(repo: Path) -> None:
    with pytest.raises(QueryError, match="unknown order_by"):
        code_query(select="symbols", order_by="name; DROP TABLE symbols", repo_root=repo)


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #


def _symbols_intact(repo: Path) -> int:
    with CodeIntelStore(repo) as store:
        return len(store.symbols())


@pytest.mark.parametrize(
    "hostile",
    [
        "'; DROP TABLE symbols; --",
        '" OR "1"="1',
        "' OR 1=1 --",
        "’ OR 1=1 --",  # noqa: RUF001  curly apostrophe -- ambiguity IS the test
        "＇; DROP TABLE symbols; --",  # noqa: RUF001  fullwidth apostrophe
        "alpha'/**/UNION/**/SELECT/**/1,2,3,4,5,6,7,8,9,10--",
        "\x00; DROP TABLE symbols",
    ],
)
def test_hostile_values_bind_and_match_nothing(repo: Path, hostile: str) -> None:
    before = _symbols_intact(repo)
    result = code_query(select="symbols", where={"name": hostile}, repo_root=repo)
    assert result.rows == ()
    assert _symbols_intact(repo) == before


@pytest.mark.parametrize(
    "hostile",
    [
        "%'; DROP TABLE symbols; --",
        "%’%",  # noqa: RUF001  curly apostrophe -- ambiguity IS the test
    ],
)
def test_hostile_like_patterns_bind_and_match_nothing(repo: Path, hostile: str) -> None:
    before = _symbols_intact(repo)
    result = code_query(select="symbols", where={"name_like": hostile}, repo_root=repo)
    assert result.rows == ()
    assert _symbols_intact(repo) == before


def test_hostile_in_values_bind_individually(repo: Path) -> None:
    before = _symbols_intact(repo)
    result = code_query(
        select="symbols",
        where={"name_in": ["alpha", "'); DROP TABLE symbols; --"]},
        repo_root=repo,
    )
    assert _names(result) == {"alpha"}, "the legitimate value must still match"
    assert _symbols_intact(repo) == before


@pytest.mark.parametrize(
    "hostile_key",
    [
        "'; DROP TABLE symbols; --",
        "symbol_name = 'x' OR 1=1",
        "1=1",
        "name_like; DROP TABLE symbols",
    ],
)
def test_hostile_keys_are_rejected_not_ignored(repo: Path, hostile_key: str) -> None:
    """Silently dropping an unrecognised key would widen the result set."""
    before = _symbols_intact(repo)
    with pytest.raises(QueryError):
        code_query(select="symbols", where={hostile_key: "x"}, repo_root=repo)
    assert _symbols_intact(repo) == before


@pytest.mark.parametrize(
    "nested",
    [
        {"$ne": None},
        {"or": [{"name": "alpha"}, {"name": "Beta"}]},
        ["alpha", "Beta"],
        ("alpha",),
        {"alpha"},
    ],
)
def test_container_values_are_rejected(repo: Path, nested: object) -> None:
    with pytest.raises(QueryError, match="scalar value"):
        code_query(select="symbols", where={"name": nested}, repo_root=repo)


def test_nested_containers_inside_in_are_rejected(repo: Path) -> None:
    with pytest.raises(QueryError, match="scalar value"):
        code_query(select="symbols", where={"name_in": ["alpha", {"$ne": None}]}, repo_root=repo)


# --------------------------------------------------------------------------- #
# ordering, limits, provenance
# --------------------------------------------------------------------------- #


def test_order_by_centrality(repo: Path) -> None:
    result = code_query(select="symbols", where={"kind_not": "nothing"}, order_by="centrality", repo_root=repo)
    ordered = [str(row["symbol_name"]) for row in result.rows]
    assert ordered[:2] == ["alpha", "Beta"], ordered


def test_order_by_callers(repo: Path) -> None:
    result = code_query(select="symbols", order_by="callers", repo_root=repo)
    ordered = [str(row["symbol_name"]) for row in result.rows]
    assert ordered[0] == "alpha", ordered


def test_order_by_name_is_stable(repo: Path) -> None:
    result = code_query(select="symbols", order_by="name", repo_root=repo)
    ordered = [str(row["symbol_name"]) for row in result.rows]
    assert ordered == sorted(ordered)


def test_limit_truncates_and_reports_the_scan(repo: Path) -> None:
    result = code_query(select="symbols", limit=2, repo_root=repo)
    assert len(result.rows) == 2
    assert result.scanned == 4
    assert result.truncated is True
    assert result.scan_capped is False


def test_limit_is_clamped_to_the_ceiling(repo: Path) -> None:
    assert code_query(select="symbols", limit=10**9, repo_root=repo).limit == MAX_LIMIT
    assert code_query(select="symbols", limit=0, repo_root=repo).limit == 1


def test_non_integer_limit_is_rejected(repo: Path) -> None:
    with pytest.raises(QueryError, match="limit must be an integer"):
        code_query(select="symbols", limit="; DROP TABLE symbols", repo_root=repo)  # type: ignore[arg-type]


def test_missing_intel_database_degrades_to_empty_not_an_error(
    make_workspace: WorkspaceFactory,
) -> None:
    root = make_workspace(files=_FILES, symbols=_SYMBOLS, with_intel=False)
    result = code_query(select="callers", repo_root=root)
    assert result.rows == ()
    assert result.match_kind == "name"


def test_empty_index_degrades_to_empty(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(index_version=2)
    result = code_query(select="symbols", repo_root=root)
    assert result.rows == ()
    assert result.engine_index_version == 2


def test_result_serializes_to_plain_json_types(repo: Path) -> None:
    import json

    payload = code_query(select="symbols", where={"kind": "class"}, order_by="name", repo_root=repo).to_dict()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["count"] == 1
    assert round_tripped["where"] == {"kind": "class"}
    assert round_tripped["order_by"] == "name"


def test_schema_describes_the_whitelist() -> None:
    schema = describe_schema()
    assert set(schema["selects"]) == set(SELECTS)
    assert "file_path" in schema["selects"]["symbols"]["fields"]
    assert schema["order_by"] == list(ORDER_BY)
    assert schema["max_limit"] == MAX_LIMIT
    assert "regex" in schema["operators"]
