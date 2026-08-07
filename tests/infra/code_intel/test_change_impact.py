"""Diff -> changed symbols -> impacted callers (F2).

Two halves. :func:`parse_diff` is pure and gets exercised directly against
git's own output shapes, because the hunk grammar is where the edge cases live.
The rest runs against a real ``git init`` repository with a synthetic index laid
over it, so the line-range arithmetic is checked against what git actually
emits rather than what it is assumed to emit.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.change_impact import (
    MATCH_NAME,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    STATUS_ADDED,
    STATUS_DELETED,
    STATUS_MODIFIED,
    STATUS_RENAMED,
    GitUnavailable,
    analyze_changes,
    parse_diff,
)

WorkspaceFactory = Callable[..., Path]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _init_repo(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    for path, body in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")


_ALPHA = "\n".join(
    [
        "def alpha():",  # 1
        "    return 1",  # 2
        "",  # 3
        "",  # 4
        "def beta():",  # 5
        "    return alpha()",  # 6
        "",  # 7
    ]
)


# --------------------------------------------------------------------------- #
# parse_diff
# --------------------------------------------------------------------------- #


def test_parses_a_single_modification_hunk() -> None:
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -10,3 +10,3 @@ def thing():\n-old\n+new\n"
    (change,) = parse_diff(diff)
    assert change.path == "a.py"
    assert change.status == STATUS_MODIFIED
    assert [(r.start, r.end) for r in change.ranges] == [(10, 12)]


def test_parses_multiple_hunks_in_one_file() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-x\n"
        "+y\n"
        "@@ -40,0 +41,2 @@\n"
        "+added\n"
        "+added\n"
    )
    (change,) = parse_diff(diff)
    assert [(r.start, r.end) for r in change.ranges] == [(2, 2), (41, 42)]


def test_pure_deletion_anchors_on_the_surviving_line() -> None:
    """``+22,0`` has no post-change span; the anchor is what still spans it."""
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -23,2 +22,0 @@\n-gone\n-gone\n"
    (change,) = parse_diff(diff)
    assert [(r.start, r.end) for r in change.ranges] == [(22, 22)]


def test_deletion_at_the_top_of_a_file_clamps_to_line_one() -> None:
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-gone\n-gone\n"
    (change,) = parse_diff(diff)
    assert [(r.start, r.end) for r in change.ranges] == [(1, 1)]


def test_hunk_without_an_explicit_count_is_one_line() -> None:
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -5 +5 @@\n-x\n+y\n"
    (change,) = parse_diff(diff)
    assert [(r.start, r.end) for r in change.ranges] == [(5, 5)]


def test_parses_added_and_deleted_files() -> None:
    diff = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+a\n"
        "+b\n"
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-a\n"
        "-b\n"
    )
    added, deleted = parse_diff(diff)
    assert (added.path, added.status) == ("new.py", STATUS_ADDED)
    assert [(r.start, r.end) for r in added.ranges] == [(1, 2)]
    assert (deleted.path, deleted.status) == ("gone.py", STATUS_DELETED)


def test_parses_a_rename_and_keeps_the_old_path() -> None:
    diff = (
        "diff --git a/old.py b/new.py\n"
        "similarity index 92%\n"
        "rename from old.py\n"
        "rename to new.py\n"
        "--- a/old.py\n"
        "+++ b/new.py\n"
        "@@ -3,1 +3,1 @@\n"
        "-x\n"
        "+y\n"
    )
    (change,) = parse_diff(diff)
    assert change.status == STATUS_RENAMED
    assert change.path == "new.py"
    assert change.old_path == "old.py"


def test_empty_diff_yields_nothing() -> None:
    assert parse_diff("") == []


# --------------------------------------------------------------------------- #
# analyze_changes
# --------------------------------------------------------------------------- #


def test_non_git_directory_is_an_error_not_an_empty_report(tmp_path: Path) -> None:
    with pytest.raises(GitUnavailable):
        analyze_changes(repo_root=tmp_path)


def test_edited_body_reports_its_symbol_and_callers(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}, {"file_path": "tests/test_a.py"}],
        symbols=[
            {"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2},
            {"file_path": "a.py", "symbol_name": "beta", "start_line": 5, "end_line": 6},
        ],
        call_edges=[
            {
                "caller_symbol_name": "beta",
                "caller_file_path": "a.py",
                "caller_start_line": 5,
                "caller_end_line": 6,
                "callee_name": "alpha",
                "call_line": 6,
            },
            {
                "caller_symbol_name": "test_alpha",
                "caller_file_path": "tests/test_a.py",
                "caller_start_line": 1,
                "caller_end_line": 3,
                "callee_name": "alpha",
                "call_line": 2,
            },
        ],
        index_version=11,
    )

    report = analyze_changes(repo_root=workspace_root)

    assert report.engine_index_version == 11
    assert report.match_kind == MATCH_NAME
    assert [s.symbol_name for s in report.changed_symbols] == ["alpha"]

    changed = report.changed_symbols[0]
    assert changed.status == STATUS_MODIFIED
    assert changed.exported is True
    assert changed.callers == 2
    assert changed.test_callers == 1
    assert changed.risk == RISK_MEDIUM

    sites = {(c.file_path, c.line) for c in report.impacted}
    assert sites == {("a.py", 6), ("tests/test_a.py", 2)}
    assert all(c.match_kind == MATCH_NAME for c in report.impacted)
    assert all(c.depth == 1 for c in report.impacted)
    assert {c.is_test for c in report.impacted} == {True, False}


def test_multi_hunk_edit_reports_every_symbol_it_touches(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    edited = _ALPHA.replace("return 1", "return 2").replace("return alpha()", "return alpha() + 1")
    (workspace_root / "a.py").write_text(edited, encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[
            {"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2},
            {"file_path": "a.py", "symbol_name": "beta", "start_line": 5, "end_line": 6},
        ],
    )

    report = analyze_changes(repo_root=workspace_root)

    assert {s.symbol_name for s in report.changed_symbols} == {"alpha", "beta"}
    assert len(report.files[0].ranges) == 2


def test_untouched_symbol_in_a_changed_file_is_left_out(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """File-level granularity would call this impacted; line ranges must not."""
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[
            {"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2},
            {"file_path": "a.py", "symbol_name": "beta", "start_line": 5, "end_line": 6},
        ],
    )

    report = analyze_changes(repo_root=workspace_root)
    assert [s.symbol_name for s in report.changed_symbols] == ["alpha"]


def test_deleted_file_surfaces_symbols_the_index_still_holds(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """The diff says gone, the index says present -- that gap is the finding."""
    _init_repo(workspace_root, {"a.py": _ALPHA, "b.py": "import a\n"})
    (workspace_root / "a.py").unlink()

    make_workspace(
        files=[{"file_path": "a.py"}, {"file_path": "b.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
        call_edges=[
            {
                "caller_symbol_name": "user",
                "caller_file_path": "b.py",
                "caller_start_line": 1,
                "caller_end_line": 1,
                "callee_name": "alpha",
                "call_line": 1,
            }
        ],
    )

    report = analyze_changes(repo_root=workspace_root)

    (changed,) = report.changed_symbols
    assert changed.symbol_name == "alpha"
    assert changed.status == STATUS_DELETED
    assert changed.callers == 1
    assert changed.risk == RISK_HIGH, "a deleted symbol with a live caller is broken, not merely risky"


def test_renamed_file_finds_symbols_under_the_old_path(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"old.py": _ALPHA})
    _git(workspace_root, "mv", "old.py", "new.py")

    make_workspace(
        files=[{"file_path": "old.py"}],
        symbols=[
            {"file_path": "old.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2},
            {"file_path": "old.py", "symbol_name": "beta", "start_line": 5, "end_line": 6},
        ],
    )

    report = analyze_changes(repo_root=workspace_root)

    (change,) = report.files
    assert change.status == STATUS_RENAMED
    assert change.old_path == "old.py"
    assert {s.symbol_name for s in report.changed_symbols} == {"alpha", "beta"}
    assert all(s.status == STATUS_RENAMED for s in report.changed_symbols)


def test_builtin_name_collision_over_reports_and_says_so(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """A method named ``open`` collects every ``open()`` in the repository.

    That is the documented cost of a name-keyed graph. The contract is not
    that it is precise -- it is that it never *misses* a caller and never
    claims to have resolved one.
    """
    body = "class Store:\n    def open(self):\n        return 1\n"
    _init_repo(workspace_root, {"store.py": body})
    (workspace_root / "store.py").write_text(body.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "store.py"}, {"file_path": "unrelated.py"}],
        symbols=[
            {
                "file_path": "store.py",
                "symbol_name": "open",
                "qualified_name": "Store.open",
                "kind": "method",
                "start_line": 2,
                "end_line": 3,
            }
        ],
        call_edges=[
            {
                "caller_symbol_name": "read_config",
                "caller_file_path": "unrelated.py",
                "caller_start_line": 1,
                "caller_end_line": 4,
                "callee_name": "open",
                "call_line": 2,
            }
        ],
    )

    report = analyze_changes(repo_root=workspace_root)

    (changed,) = report.changed_symbols
    assert changed.qualified_name == "Store.open"
    assert changed.callers == 1
    assert [c.file_path for c in report.impacted] == ["unrelated.py"]
    assert report.match_kind == MATCH_NAME
    assert all(c.match_kind == MATCH_NAME for c in report.impacted)


def test_private_symbol_with_no_callers_is_low_risk(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    body = "def _helper():\n    return 1\n"
    _init_repo(workspace_root, {"a.py": body})
    (workspace_root / "a.py").write_text(body.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "_helper", "start_line": 1, "end_line": 2}],
    )

    (changed,) = analyze_changes(repo_root=workspace_root).changed_symbols
    assert changed.exported is False
    assert changed.risk == RISK_LOW


def test_widely_called_export_is_high_risk(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
        call_edges=[
            {
                "caller_symbol_name": f"caller{n}",
                "caller_file_path": f"c{n}.py",
                "caller_start_line": 1,
                "caller_end_line": 2,
                "callee_name": "alpha",
                "call_line": 1,
            }
            for n in range(6)
        ],
    )

    (changed,) = analyze_changes(repo_root=workspace_root).changed_symbols
    assert changed.callers == 6
    assert changed.risk == RISK_HIGH


def test_depth_two_expands_through_the_caller_chain(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")

    edges = [
        {
            "caller_symbol_name": "beta",
            "caller_file_path": "b.py",
            "caller_start_line": 1,
            "caller_end_line": 2,
            "callee_name": "alpha",
            "call_line": 1,
        },
        {
            "caller_symbol_name": "gamma",
            "caller_file_path": "c.py",
            "caller_start_line": 1,
            "caller_end_line": 2,
            "callee_name": "beta",
            "call_line": 1,
        },
    ]
    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
        call_edges=edges,
    )

    shallow = analyze_changes(repo_root=workspace_root, depth=1)
    assert {c.file_path for c in shallow.impacted} == {"b.py"}

    deep = analyze_changes(repo_root=workspace_root, depth=2)
    assert {c.file_path for c in deep.impacted} == {"b.py", "c.py"}
    assert {c.file_path: c.depth for c in deep.impacted} == {"b.py": 1, "c.py": 2}
    # Direct-caller counts drive risk, so a deeper walk must not inflate them.
    assert deep.changed_symbols[0].callers == 1


def test_a_site_reaching_two_changed_symbols_is_attributed_to_both(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """A global site dedupe let whichever symbol arrived first own the anchor.

    ``alpha`` is processed before ``beta`` (pending is ordered by start_line).
    At depth 2 ``alpha`` reaches ``c.py:1`` via ``beta``, and a location-only
    dedupe then blocked ``beta`` -- whose *direct* caller that is -- from ever
    recording it. The report showed a direct caller as a remote one and dropped
    it from the symbol it actually calls.
    """
    _init_repo(workspace_root, {"a.py": _ALPHA})
    edited = _ALPHA.replace("return 1", "return 2").replace("return alpha()", "return alpha() + 1")
    (workspace_root / "a.py").write_text(edited, encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[
            {"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2},
            {"file_path": "a.py", "symbol_name": "beta", "start_line": 5, "end_line": 6},
        ],
        call_edges=[
            {
                "caller_symbol_name": "beta",
                "caller_file_path": "b.py",
                "caller_start_line": 1,
                "caller_end_line": 2,
                "callee_name": "alpha",
                "call_line": 1,
            },
            {
                "caller_symbol_name": "gamma",
                "caller_file_path": "c.py",
                "caller_start_line": 1,
                "caller_end_line": 2,
                "callee_name": "beta",
                "call_line": 1,
            },
        ],
    )

    report = analyze_changes(repo_root=workspace_root, depth=2)
    by_symbol = {(c.changed_symbol, c.file_path): c.depth for c in report.impacted}

    assert by_symbol[("beta", "c.py")] == 1, "c.py directly calls beta and must be reported as depth 1"
    assert by_symbol[("alpha", "c.py")] == 2, "c.py also reaches alpha, two hops out"


def test_a_caller_cycle_terminates(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
        call_edges=[
            {
                "caller_symbol_name": "beta",
                "caller_file_path": "b.py",
                "caller_start_line": 1,
                "caller_end_line": 2,
                "callee_name": "alpha",
                "call_line": 1,
            },
            {
                "caller_symbol_name": "alpha",
                "caller_file_path": "d.py",
                "caller_start_line": 1,
                "caller_end_line": 2,
                "callee_name": "beta",
                "call_line": 1,
            },
        ],
    )

    report = analyze_changes(repo_root=workspace_root, depth=6)
    assert {c.file_path for c in report.impacted} == {"b.py", "d.py"}


def test_limit_truncates_but_reports_the_real_total(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """A clipped list must never read as a complete one."""
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
        references=[{"symbol_name": "alpha", "file_path": f"r{n}.py", "line": n + 1} for n in range(12)],
    )

    report = analyze_changes(repo_root=workspace_root, limit=5)
    assert len(report.impacted) == 5
    assert report.impacted_total == 12
    assert report.truncated is True


def test_changed_file_absent_from_the_index_is_named(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """Silence here would read as "nothing depends on it"."""
    _init_repo(workspace_root, {"a.py": _ALPHA, "ghost.py": "x = 1\n"})
    (workspace_root / "ghost.py").write_text("x = 2\n", encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
    )

    report = analyze_changes(repo_root=workspace_root)
    assert report.unindexed_paths == ("ghost.py",)
    assert report.changed_symbols == ()


def test_paths_filter_narrows_the_diff(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA, "b.py": "def other():\n    return 1\n"})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")
    (workspace_root / "b.py").write_text("def other():\n    return 2\n", encoding="utf-8")

    make_workspace(
        files=[{"file_path": "a.py"}, {"file_path": "b.py"}],
        symbols=[
            {"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2},
            {"file_path": "b.py", "symbol_name": "other", "start_line": 1, "end_line": 2},
        ],
    )

    report = analyze_changes(repo_root=workspace_root, paths=["b.py"])
    assert [c.path for c in report.files] == ["b.py"]
    assert [s.symbol_name for s in report.changed_symbols] == ["other"]


def test_clean_tree_reports_no_changes(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
    )

    report = analyze_changes(repo_root=workspace_root)
    assert report.files == ()
    assert report.changed_symbols == ()
    assert report.truncated is False


def test_report_serializes_to_plain_json_types(
    workspace_root: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    _init_repo(workspace_root, {"a.py": _ALPHA})
    (workspace_root / "a.py").write_text(_ALPHA.replace("return 1", "return 2"), encoding="utf-8")
    make_workspace(
        files=[{"file_path": "a.py"}],
        symbols=[{"file_path": "a.py", "symbol_name": "alpha", "start_line": 1, "end_line": 2}],
    )

    import json

    payload = analyze_changes(repo_root=workspace_root).to_dict()
    assert json.loads(json.dumps(payload))["match_kind"] == MATCH_NAME
    assert payload["changed_symbols"][0]["name"] == "alpha"
