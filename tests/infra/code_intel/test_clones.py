"""F6 -- clone detection, and the ways it is allowed to be wrong.

The detector is approximate by construction: MinHash estimates Jaccard, and LSH
banding discards candidate pairs before they are ever scored. Both errors are
acceptable only in one direction. Banding may miss a real clone; it must never
invent one, because every reported pair is scored individually after banding.
And a stale or unbuilt table must never surface as an empty list, since "no
duplicates found" is a claim about the code and "nobody has looked" is not.

These tests pin the score's behaviour at the three points that matter --
identical, renamed, unrelated -- then check LSH recall against a brute-force
baseline rather than against a number someone once observed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.clones import (
    DEFAULT_THRESHOLD,
    MIN_TOKENS,
    NUM_PERM,
    SHINGLE_K,
    ClonesStale,
    build_clones,
    load_clones,
    normalise_tokens,
    shingle,
    signature,
)
from lemoncrow.infra.code_intel.query import QueryError, code_query
from lemoncrow.infra.code_intel.sidecar import open_sidecar, stamp_of

WorkspaceFactory = Callable[..., Path]


# A body long enough to clear MIN_TOKENS, built so the variants below differ in
# exactly one dimension each.
_BODY = """
    total = 0
    for index, item in enumerate(values):
        if item is None:
            continue
        scaled = item * factor + offset
        if scaled > ceiling:
            scaled = ceiling
        elif scaled < floor:
            scaled = floor
        total += scaled * weights[index % len(weights)]
    average = total / max(1, len(values))
    return round(average, precision)
"""

_RENAMED_BODY = """
    accumulator = 0
    for position, element in enumerate(numbers):
        if element is None:
            continue
        adjusted = element * multiplier + shift
        if adjusted > upper:
            adjusted = upper
        elif adjusted < lower:
            adjusted = lower
        accumulator += adjusted * coefficients[position % len(coefficients)]
        mean = accumulator / max(1, len(numbers))
    return round(mean, digits)
"""

_UNRELATED_BODY = """
    parsed = urlsplit(location)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported scheme: {parsed.scheme}")
    headers = dict(default_headers)
    headers.setdefault("accept", "application/json")
    session = build_session(retries=retries, backoff=backoff)
    response = session.get(parsed.geturl(), headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return {key: payload[key] for key in wanted if key in payload}
"""


def _write_module(root: Path, rel_path: str, functions: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Write a real source file and return symbol rows with true byte offsets.

    The detector slices source by ``start_byte``/``end_byte`` from the engine, so
    a fixture that guesses those offsets would test a different program.
    """
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    text = ""
    for name, body in functions:
        source = f"def {name}(values, factor, offset, ceiling, floor, weights, precision):{body}\n"
        start = len(text.encode("utf-8"))
        text += source
        rows.append(
            {
                "symbol_id": f"{rel_path}::{name}",
                "file_path": rel_path,
                "symbol_name": name,
                "qualified_name": name,
                "kind": "function",
                "start_byte": start,
                "end_byte": len(text.encode("utf-8")),
            }
        )
    path.write_text(text, encoding="utf-8")
    return rows


@pytest.fixture
def clone_repo(make_workspace: WorkspaceFactory, tmp_path: Path) -> Path:
    root = make_workspace(files=[{"file_path": "src/a.py"}, {"file_path": "src/b.py"}])
    rows_a = _write_module(root, "src/a.py", [("compute_average", _BODY)])
    rows_b = _write_module(
        root,
        "src/b.py",
        [
            ("compute_average_copy", _BODY),
            ("compute_mean", _RENAMED_BODY),
            ("fetch_payload", _UNRELATED_BODY),
        ],
    )
    _insert_symbols(root, rows_a + rows_b)
    return root


def _insert_symbols(root: Path, rows: list[dict[str, object]]) -> None:
    from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir

    conn = sqlite3.connect(workspace_dir(root) / CODE_CONTEXT_DB)
    try:
        repo_id = str(conn.execute("SELECT repo_id FROM files LIMIT 1").fetchone()[0])
        conn.executemany(
            "INSERT OR REPLACE INTO symbols (symbol_id, repo_id, file_path, language, symbol_name, "
            "qualified_name, kind, signature, start_byte, end_byte, start_line, end_line, "
            "parent_symbol, doc_summary, content_hash) "
            "VALUES (?, ?, ?, 'python', ?, ?, ?, '', ?, ?, 1, 2, NULL, NULL, 'h')",
            [
                (
                    row["symbol_id"],
                    repo_id,
                    row["file_path"],
                    row["symbol_name"],
                    row["qualified_name"],
                    row["kind"],
                    row["start_byte"],
                    row["end_byte"],
                )
                for row in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Normalisation                                                               #
# --------------------------------------------------------------------------- #


def test_reformatting_alone_cannot_change_the_token_stream() -> None:
    """Whitespace is a separator, never a token."""
    assert normalise_tokens("a  =   b+1") == normalise_tokens("a = b + 1")


def test_identifiers_are_placeholdered_but_keywords_survive() -> None:
    """Rename-blindness is the whole reason this detector finds anything.

    Keywords have to survive, or every `if` block would look like every `for`
    block and control-flow shape would leave the signature entirely.
    """
    assert normalise_tokens("if total > 3: return total") == normalise_tokens("if count > 9: return count")
    assert normalise_tokens("if x: pass") != normalise_tokens("while x: pass")


def test_verbatim_mode_still_distinguishes_renamed_code() -> None:
    """The escape hatch keeps literal-copy detection available."""
    assert normalise_tokens("total = 1", rename_blind=False) != normalise_tokens("count = 1", rename_blind=False)


def test_comments_are_stripped_before_tokenising() -> None:
    """A shared comment is not evidence of a shared implementation."""
    assert normalise_tokens("x = 1  # explain the thing") == normalise_tokens("x = 1")
    assert normalise_tokens("x = 1  // explain") == normalise_tokens("x = 1")
    assert normalise_tokens("x = 1  /* explain\nover lines */") == normalise_tokens("x = 1")


def test_docstrings_are_stripped() -> None:
    """Two functions sharing a docstring are not thereby clones."""
    assert normalise_tokens('def f():\n    """Doc."""\n    return 1') == normalise_tokens("def f():\n    return 1")


def test_string_literals_are_kept_as_tokens() -> None:
    """Regression: stripping every string made all boilerplate identical.

    Measured on this repository, dropping string literals produced 2,968 pairs
    whose top scorers were every ``to_dict`` matching every other ``to_dict`` at
    1.000 -- because once identifiers are placeholdered, a ``to_dict`` method's
    dict keys are the only thing left that distinguishes it. Keeping literals
    cut that to 1,909 and put real duplication at the top.
    """
    a = normalise_tokens('return {"model": self.model, "tokens": self.tokens}')
    b = normalise_tokens('return {"rule": self.rule, "severity": self.severity}')
    assert a != b
    assert '"model"' in a


def test_shingles_encode_order_not_just_vocabulary() -> None:
    """The property that makes this a clone detector rather than a bag of words."""
    forward = shingle(["a", "b", "c", "d", "e", "f"])
    reversed_ = shingle(["f", "e", "d", "c", "b", "a"])
    assert not forward & reversed_


def test_short_token_runs_yield_one_shingle_rather_than_none() -> None:
    """Empty sets would make every short symbol identical to every other."""
    assert len(shingle(["a", "b"], k=SHINGLE_K)) == 1
    assert shingle([]) == set()


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #


def test_identical_source_scores_one() -> None:
    tokens = normalise_tokens(_BODY)
    assert signature(tokens).jaccard(signature(tokens)) == 1.0


def test_rename_only_clone_scores_high() -> None:
    """Identifier renaming is the case a text diff misses and this must not.

    Regression: with verbatim tokens this scored 0.039, because at k=5 nearly
    every shingle contains at least one renamed identifier. Renaming is the
    first thing that happens to copied code.
    """
    score = signature(normalise_tokens(_BODY)).jaccard(signature(normalise_tokens(_RENAMED_BODY)))
    assert score >= DEFAULT_THRESHOLD, score


def test_renaming_defeats_a_verbatim_token_comparison() -> None:
    """Pins the defect the placeholder pass exists to fix."""
    verbatim = signature(normalise_tokens(_BODY, rename_blind=False)).jaccard(
        signature(normalise_tokens(_RENAMED_BODY, rename_blind=False))
    )
    assert verbatim < 0.1, verbatim


def test_unrelated_functions_score_low() -> None:
    score = signature(normalise_tokens(_BODY)).jaccard(signature(normalise_tokens(_UNRELATED_BODY)))
    assert score < 0.1, score


def test_signature_width_is_the_declared_constant() -> None:
    assert signature(normalise_tokens(_BODY)).num_perm == NUM_PERM


# --------------------------------------------------------------------------- #
# LSH recall against brute force                                              #
# --------------------------------------------------------------------------- #


def test_banding_finds_every_pair_brute_force_would_report(clone_repo: Path) -> None:
    """Banding may cost recall; here it must cost none.

    The baseline is computed the expensive way -- every pair, scored -- so this
    asserts against the algorithm's own definition rather than a recorded number.
    """
    from lemoncrow.infra.code_intel.clones import _read_symbol_sources
    from lemoncrow.infra.code_intel.store import CodeIntelStore

    with CodeIntelStore(clone_repo) as store:
        symbols = store.symbols()
    tokens_by_symbol, _ = _read_symbol_sources(symbols, clone_repo)
    signatures = {
        symbol_id: signature(tokens)
        for symbol_id, tokens in tokens_by_symbol.items()
        if len(tokens) >= MIN_TOKENS
    }

    brute: set[tuple[str, str]] = set()
    ids = sorted(signatures)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if signatures[left].jaccard(signatures[right]) >= DEFAULT_THRESHOLD:
                brute.add((left, right))

    found = {(pair.symbol_a, pair.symbol_b) for pair in build_clones(clone_repo).pairs}
    assert brute, "fixture produced no clones for the baseline to compare against"
    assert found == brute


def test_every_reported_pair_was_actually_scored(clone_repo: Path) -> None:
    """Banding proposes; scoring disposes. No pair is reported on banding alone."""
    report = build_clones(clone_repo)
    assert report.candidate_pairs >= len(report.pairs)
    for pair in report.pairs:
        assert pair.jaccard >= report.threshold


# --------------------------------------------------------------------------- #
# Build behaviour                                                             #
# --------------------------------------------------------------------------- #


def test_the_copied_function_is_found_and_the_unrelated_one_is_not(clone_repo: Path) -> None:
    names = {
        (pair.qualified_name_a, pair.qualified_name_b) for pair in build_clones(clone_repo).pairs
    }
    assert ("compute_average", "compute_average_copy") in names
    assert not any("fetch_payload" in pair for pair in names)


def test_short_symbols_are_skipped_rather_than_reported(make_workspace: WorkspaceFactory) -> None:
    """Every three-line getter matches every other; reporting those buries the rest."""
    root = make_workspace(files=[{"file_path": "src/tiny.py"}])
    rows = _write_module(root, "src/tiny.py", [("one", "\n    return 1\n"), ("two", "\n    return 1\n")])
    _insert_symbols(root, rows)

    report = build_clones(root)
    assert report.pairs == ()
    assert report.symbols_skipped_short == 2
    assert report.symbols_considered == 0


def test_a_deleted_file_is_counted_not_raised(clone_repo: Path) -> None:
    """One file removed after indexing must not cost the whole pass."""
    (clone_repo / "src/a.py").unlink()
    report = build_clones(clone_repo)
    assert report.symbols_unreadable == 1
    assert report.symbols_considered == 3


def test_rebuild_replaces_rather_than_accumulates(clone_repo: Path) -> None:
    """A symbol that stopped being a clone has to stop being reported."""
    first = build_clones(clone_repo)
    assert first.pairs

    (clone_repo / "src/b.py").write_text(
        "def compute_average_copy(values, factor, offset, ceiling, floor, weights, precision):"
        + _UNRELATED_BODY
        + "\n",
        encoding="utf-8",
    )
    _insert_symbols(
        clone_repo,
        _write_module(clone_repo, "src/b.py", [("compute_average_copy", _UNRELATED_BODY)]),
    )
    build_clones(clone_repo)
    assert load_clones(clone_repo) == ()


def test_the_build_stamps_the_index_generation_it_read(clone_repo: Path) -> None:
    report = build_clones(clone_repo)
    conn = open_sidecar(clone_repo)
    try:
        recorded = stamp_of(conn, "symbol_clones")
    finally:
        conn.close()
    assert recorded is not None
    assert recorded.engine_index_version == report.engine_index_version


# --------------------------------------------------------------------------- #
# Staleness -- never an empty list                                            #
# --------------------------------------------------------------------------- #


def test_loading_an_unbuilt_table_raises_rather_than_returning_nothing(clone_repo: Path) -> None:
    """"No duplicates" is a claim about the code; "never built" is not."""
    with pytest.raises(ClonesStale, match="never been built"):
        load_clones(clone_repo)


def test_a_reindex_makes_the_stored_table_refuse_to_load(clone_repo: Path) -> None:
    build_clones(clone_repo)
    assert load_clones(clone_repo)

    _bump_index_version(clone_repo)

    with pytest.raises(ClonesStale, match="index is now at"):
        load_clones(clone_repo)


def _bump_index_version(root: Path) -> None:
    from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir

    conn = sqlite3.connect(workspace_dir(root) / CODE_CONTEXT_DB)
    try:
        current = int(conn.execute("SELECT value FROM engine_state WHERE key = 'index_version'").fetchone()[0])
        conn.execute("UPDATE engine_state SET value = ? WHERE key = 'index_version'", (str(current + 1),))
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# code_query integration                                                      #
# --------------------------------------------------------------------------- #


def test_code_query_reads_the_clone_table(clone_repo: Path) -> None:
    build_clones(clone_repo)
    result = code_query(select="clones", repo_root=clone_repo)

    assert result.rows
    assert result.rows[0]["qualified_name_a"] == "compute_average"
    assert result.rows[0]["jaccard"] >= DEFAULT_THRESHOLD


def test_code_query_filters_clones_by_score(clone_repo: Path) -> None:
    build_clones(clone_repo)
    assert code_query(select="clones", where={"jaccard_gte": 0.99}, repo_root=clone_repo).rows
    assert not code_query(select="clones", where={"jaccard_gt": 1.0}, repo_root=clone_repo).rows


def test_code_query_rejects_an_unknown_clone_field(clone_repo: Path) -> None:
    build_clones(clone_repo)
    with pytest.raises(QueryError):
        code_query(select="clones", where={"similarity": 0.9}, repo_root=clone_repo)


def test_code_query_refuses_a_clone_table_that_was_never_built(clone_repo: Path) -> None:
    """The zero-row answer this replaces would have read as "no duplicates"."""
    with pytest.raises(ClonesStale):
        code_query(select="clones", repo_root=clone_repo)


def test_clones_is_not_name_keyed(clone_repo: Path) -> None:
    """Rows are keyed by symbol_id, so no `match_kind: name` caveat applies."""
    build_clones(clone_repo)
    assert code_query(select="clones", repo_root=clone_repo).match_kind is None


def test_describe_schema_advertises_clones() -> None:
    from lemoncrow.infra.code_intel.query import describe_schema

    assert "clones" in describe_schema()["selects"]
