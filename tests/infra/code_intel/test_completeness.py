"""The completeness contract every code-intel response has to honour.

The predicate a reviewer wants to evaluate is ``objective == "exhaustive" and
not truncated``. That only works if every enumerating surface stamps both, and
if the pre-limit count is exact regardless of what got cut. These tests pin the
contract across all of them at once, so a new tool that forgets it fails here
rather than silently returning a list a consumer over-trusts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.change_impact import analyze_changes
from lemoncrow.infra.code_intel.completeness import (
    CODE_OP_OBJECTIVES,
    OBJECTIVE_EXHAUSTIVE,
    OBJECTIVE_RANKED,
    with_objective,
)
from lemoncrow.infra.code_intel.coverage import check_coverage
from lemoncrow.infra.code_intel.file_graph import open_file_graph
from lemoncrow.infra.code_intel.query import code_query

WorkspaceFactory = Callable[..., Path]


@pytest.fixture
def repo(make_workspace: WorkspaceFactory) -> Path:
    return make_workspace(
        files=[{"file_path": "src/a.py"}, {"file_path": "src/b.py"}],
        symbols=[{"file_path": "src/a.py", "symbol_name": "alpha"}],
        imports=[{"source_file": "src/b.py", "raw_import": "a", "target_file": "src/a.py"}],
    )


def test_ranked_and_exhaustive_are_distinct_values() -> None:
    assert OBJECTIVE_RANKED != OBJECTIVE_EXHAUSTIVE


def test_with_objective_stamps_in_place() -> None:
    payload: dict[str, object] = {"rows": []}
    assert with_objective(payload, OBJECTIVE_EXHAUSTIVE) is payload
    assert payload["objective"] == OBJECTIVE_EXHAUSTIVE


def test_engine_op_map_only_uses_known_objectives() -> None:
    assert set(CODE_OP_OBJECTIVES.values()) <= {OBJECTIVE_RANKED, OBJECTIVE_EXHAUSTIVE}


def test_search_is_ranked_and_relations_ops_are_exhaustive() -> None:
    """The split that the whole contract exists to express."""
    assert CODE_OP_OBJECTIVES["search"] == OBJECTIVE_RANKED
    for op in ("callers", "callees", "usages"):
        assert CODE_OP_OBJECTIVES[op] == OBJECTIVE_EXHAUSTIVE


def test_unverified_engine_ops_are_absent_rather_than_guessed() -> None:
    """Absent must read as unclassified, never as exhaustive."""
    for op in ("pattern", "node", "blame", "index"):
        assert op not in CODE_OP_OBJECTIVES


def test_every_open_module_stamps_an_objective(repo: Path, workspace_root: Path) -> None:
    payloads = [
        check_coverage(paths=["src/a.py"], repo_root=repo).to_dict(),
        code_query(select="symbols", repo_root=repo).to_dict(),
    ]
    with open_file_graph(repo) as graph:
        payloads.append(graph.blast_radius("src/a.py"))
        payloads.append(graph.dead_code())
        payloads.append(graph.cycles())
        payloads.append(graph.coupling())
        payloads.append(graph.topology())
    for payload in payloads:
        assert payload["objective"] == OBJECTIVE_EXHAUSTIVE


def test_change_impact_stamps_an_objective(make_workspace: WorkspaceFactory, workspace_root: Path) -> None:
    import subprocess

    workspace_root.mkdir(parents=True, exist_ok=True)
    for args in (("init", "-q", "-b", "main"),):
        subprocess.run(["git", *args], cwd=workspace_root, check=True, capture_output=True)
    (workspace_root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=workspace_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-q", "-m", "i"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )
    make_workspace(files=[{"file_path": "a.py"}], symbols=[{"file_path": "a.py", "symbol_name": "alpha"}])

    assert analyze_changes(repo_root=workspace_root).to_dict()["objective"] == OBJECTIVE_EXHAUSTIVE


# --------------------------------------------------------------------------- #
# blast_radius bounding
# --------------------------------------------------------------------------- #


@pytest.fixture
def wide_repo(make_workspace: WorkspaceFactory) -> Path:
    """One target imported directly by 30 files, 10 of them tests."""
    importers = [f"src/mod{n}.py" for n in range(20)] + [f"tests/test_{n}.py" for n in range(10)]
    return make_workspace(
        files=[{"file_path": "src/target.py"}] + [{"file_path": path} for path in importers],
        symbols=[{"file_path": "src/target.py", "symbol_name": "target"}],
        imports=[{"source_file": path, "raw_import": "target", "target_file": "src/target.py"} for path in importers],
    )


def test_blast_radius_bounds_its_lists(wide_repo: Path) -> None:
    with open_file_graph(wide_repo) as graph:
        result = graph.blast_radius("src/target.py", limit=5)
    assert len(result["direct_importers"]) == 5
    assert result["truncated"] is True


def test_blast_radius_counts_are_exact_despite_truncation(wide_repo: Path) -> None:
    """Truncation must cost detail, never the answer to "how big is this"."""
    with open_file_graph(wide_repo) as graph:
        clipped = graph.blast_radius("src/target.py", limit=5)
        whole = graph.blast_radius("src/target.py", limit=1000)

    assert clipped["direct_importer_count"] == whole["direct_importer_count"] == 30
    assert clipped["affected_test_count"] == whole["affected_test_count"] == 10
    assert clipped["risk_level"] == whole["risk_level"]


def test_blast_radius_is_not_truncated_when_it_fits(wide_repo: Path) -> None:
    with open_file_graph(wide_repo) as graph:
        result = graph.blast_radius("src/target.py", limit=100)
    assert result["truncated"] is False
    assert len(result["direct_importers"]) == 30


def test_blast_radius_reports_transitive_separately(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace(
        files=[{"file_path": p} for p in ("a.py", "b.py", "c.py")],
        imports=[
            {"source_file": "b.py", "raw_import": "a", "target_file": "a.py"},
            {"source_file": "c.py", "raw_import": "b", "target_file": "b.py"},
        ],
    )
    with open_file_graph(root) as graph:
        result = graph.blast_radius("a.py")
    assert result["direct_importers"] == ["b.py"]
    assert result["direct_importer_count"] == 1
    assert result["transitive_importers"] == ["c.py"]
    assert result["transitive_importer_count"] == 1
    assert result["truncated"] is False


def test_exhaustive_and_untruncated_is_a_usable_predicate(repo: Path) -> None:
    """The whole point: a consumer evaluates two fields, not a tool name."""
    with open_file_graph(repo) as graph:
        payload = graph.blast_radius("src/a.py")
    trustworthy = payload["objective"] == OBJECTIVE_EXHAUSTIVE and not payload["truncated"]
    assert trustworthy
