"""F1: file-graph analytics over the repo's own ``imports`` table.

Fixture graph (forward imports)::

    src/top.py      -> src/middle.py, src/base.py
    src/middle.py   -> src/base.py
    src/base.py     -> (nothing)
    src/orphan.py   -> (nothing; nobody imports it)      [dead]
    src/cyc_a.py    -> src/cyc_b.py
    src/cyc_b.py    -> src/cyc_a.py                      [2-cycle]
    src/cli.py      -> (nothing; a console script)       [entry point]
    src/__init__.py -> (nothing)                         [entry point]
    tests/test_top.py -> src/top.py                      [test, never dead]
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lemoncrow.infra.code_intel.file_graph import open_file_graph

WorkspaceFactory = Callable[..., Path]

_FILES = [
    "src/__init__.py",
    "src/base.py",
    "src/middle.py",
    "src/top.py",
    "src/orphan.py",
    "src/cyc_a.py",
    "src/cyc_b.py",
    "src/cli.py",
    "tests/test_top.py",
]

_EDGES = [
    ("src/middle.py", "src/base.py"),
    ("src/top.py", "src/middle.py"),
    ("src/top.py", "src/base.py"),
    ("src/cyc_a.py", "src/cyc_b.py"),
    ("src/cyc_b.py", "src/cyc_a.py"),
    ("tests/test_top.py", "src/top.py"),
]

#: Third-party and stdlib imports the engine could not resolve to a repo file.
_DANGLING = ["src/base.py", "src/top.py", "src/top.py"]


def _graph_workspace(make_workspace: WorkspaceFactory, root_files: bool = True) -> Path:
    imports = [{"source_file": source, "raw_import": target, "target_file": target} for source, target in _EDGES]
    imports += [{"source_file": source, "raw_import": "os", "target_file": None} for source in _DANGLING]
    return make_workspace(
        files=[{"file_path": path} for path in _FILES] if root_files else [],
        symbols=[
            {"file_path": "src/orphan.py", "symbol_name": "orphan_fn", "end_line": 12},
            {"file_path": "src/orphan.py", "symbol_name": "inner", "parent_symbol": "orphan_fn", "end_line": 9},
        ],
        imports=imports,
        index_version=8,
    )


def _write_pyproject(root: Path, target: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "fixture"\n\n[project.scripts]\nfixture = "{target}"\n',
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# The bug this feature exists to fix                                           #
# --------------------------------------------------------------------------- #


def test_blast_radius_returns_the_real_importers(make_workspace: WorkspaceFactory) -> None:
    """Regression: the old implementation returned empty importers, risk "low".

    It read a machine-global JSON cache instead of the repo's imports table, so
    a file with three importers looked like a file with none.
    """
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        result = graph.blast_radius("src/base.py")

    assert result["direct_importers"] == ["src/middle.py", "src/top.py"]
    assert result["transitive_importers"] == ["tests/test_top.py"]
    assert result["affected_tests"] == ["tests/test_top.py"]
    assert result["risk_level"] == "medium"  # 3 importers
    assert result["indexed"] is True


def test_blast_radius_of_a_leaf_is_genuinely_empty(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        result = graph.blast_radius("src/orphan.py")
    assert result["direct_importers"] == []
    assert result["risk_level"] == "low"
    # ...and the caller can tell that apart from an unresolved graph:
    assert result["unresolved_edges"] == len(_DANGLING)


def test_every_response_reports_resolved_and_unresolved_edges(make_workspace: WorkspaceFactory) -> None:
    """The caveat that makes a small answer interpretable."""
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        results = [
            graph.blast_radius("src/base.py"),
            graph.dead_code(),
            graph.cycles(),
            graph.coupling(),
            graph.topology(),
        ]
    for result in results:
        assert result["resolved_edges"] == len(_EDGES)
        assert result["unresolved_edges"] == len(_DANGLING)
        assert result["engine_index_version"] == 8
        assert result["analyzed_files"] == len(_FILES)


def test_absolute_paths_resolve_against_the_repo_root(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        result = graph.blast_radius(str(root / "src/base.py"))
    assert result["modified_file"] == "src/base.py"
    assert result["direct_importers"] == ["src/middle.py", "src/top.py"]


# --------------------------------------------------------------------------- #
# dead_code                                                                    #
# --------------------------------------------------------------------------- #


def test_dead_code_finds_the_orphan_and_excludes_entry_points(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    _write_pyproject(root, "src.cli:main")
    with open_file_graph(root) as graph:
        result = graph.dead_code()

    dead = {row["path"] for row in result["dead_files"]}
    assert "src/orphan.py" in dead
    # Entry points are reachable without an importer; calling them dead is noise.
    assert "src/cli.py" not in dead  # [project.scripts]
    assert "src/__init__.py" not in dead
    assert "tests/test_top.py" not in dead
    # Nothing with an inbound edge is dead.
    assert dead.isdisjoint({"src/base.py", "src/middle.py", "src/top.py", "src/cyc_a.py", "src/cyc_b.py"})


def test_dead_code_row_carries_top_level_exports(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        row = next(entry for entry in graph.dead_code()["dead_files"] if entry["path"] == "src/orphan.py")
    assert row["exports"] == ["orphan_fn"]  # `inner` is nested, not an export
    assert row["lines_total"] == 12
    assert row["language"] == "python"


def test_a_missing_pyproject_costs_entry_points_not_the_analysis(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    assert not (root / "pyproject.toml").exists()
    with open_file_graph(root) as graph:
        dead = {row["path"] for row in graph.dead_code()["dead_files"]}
    assert "src/orphan.py" in dead
    assert "src/cli.py" in dead  # no manifest to learn it from -- degraded, not broken


def test_dead_code_only_considers_languages_the_extractor_analysed(make_workspace: WorkspaceFactory) -> None:
    """A Markdown file has no imports because nothing parses its imports.

    Counting those as dead code buried the real hits 3:1 on this repo.
    """
    root = make_workspace(
        files=[
            {"file_path": "src/orphan.py", "language": "python"},
            {"file_path": "src/importer.py", "language": "python"},
            {"file_path": "README.md", "language": "markdown"},
            {"file_path": "deploy.sh", "language": "bash"},
        ],
        imports=[{"source_file": "src/importer.py", "raw_import": "os", "target_file": None}],
    )
    with open_file_graph(root) as graph:
        result = graph.dead_code()

    assert {row["path"] for row in result["dead_files"]} == {"src/orphan.py", "src/importer.py"}
    assert result["import_languages"] == ["python"]


def test_dead_code_truncates_and_says_so(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        result = graph.dead_code(limit=1)
    assert len(result["dead_files"]) == 1
    assert result["truncated"] is True
    assert result["dead_file_count"] > 1


# --------------------------------------------------------------------------- #
# cycles / coupling / topology                                                 #
# --------------------------------------------------------------------------- #


def test_cycles_finds_the_two_cycle_and_nothing_else(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        result = graph.cycles()
    assert result["cycle_count"] == 1
    assert result["cycles"] == [["src/cyc_a.py", "src/cyc_b.py"]]
    assert result["truncated"] is False


def test_instability_arithmetic(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        rows = {row["path"]: row for row in graph.coupling()["files"]}

    # base.py: imported by middle+top, imports nothing -> maximally stable.
    assert rows["src/base.py"]["afferent"] == 2
    assert rows["src/base.py"]["efferent"] == 0
    assert rows["src/base.py"]["instability"] == 0.0

    # top.py: imported only by the test, imports two -> Ce/(Ca+Ce) = 2/3.
    assert rows["src/top.py"]["instability"] == round(2 / 3, 4)

    # middle.py sits between them.
    assert rows["src/middle.py"]["instability"] == 0.5

    # A file with no edges at all is not "coupled" and is left out.
    assert "src/orphan.py" not in rows


def test_topology_groups_by_directory(make_workspace: WorkspaceFactory) -> None:
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        result = graph.topology()
    modules = {row["module"]: row for row in result["modules"]}
    assert set(modules) == {"src", "tests"}
    assert modules["tests"]["depends_on"] == ["src"]
    assert modules["src"]["afferent_modules"] == 1
    assert modules["src"]["files"] == 8
    assert result["hotspots"]


def test_paths_restrict_the_report_not_the_graph(make_workspace: WorkspaceFactory) -> None:
    """Narrowing the graph would change the answers; narrowing the view does not."""
    root = _graph_workspace(make_workspace)
    with open_file_graph(root) as graph:
        scoped = graph.coupling(paths=["tests"])
        full = graph.coupling()

    assert {row["path"] for row in scoped["files"]} == {"tests/test_top.py"}
    # The edge counts are still the whole repo's -- the graph was not truncated.
    assert scoped["resolved_edges"] == full["resolved_edges"]
    assert scoped["analyzed_files"] == full["analyzed_files"]


def test_an_empty_index_analyses_cleanly(make_workspace: WorkspaceFactory) -> None:
    root = make_workspace()
    with open_file_graph(root) as graph:
        result = graph.dead_code()
    assert result["dead_files"] == []
    assert result["analyzed_files"] == 0
    assert result["resolved_edges"] == 0
