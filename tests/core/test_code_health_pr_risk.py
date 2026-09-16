"""WS10 G16 -- PR-risk profile + commit-provenance classification.

Verifies the risk score rises with blast-radius / churn / missing tests, and
that heuristic commit classification labels representative messages correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.freshness import IndexRebuilding
from lemoncrow.pro.capabilities.code_context.engine import CodeContextEngine
from lemoncrow.pro.capabilities.code_health.pr_risk import (
    _W_TESTGAP,
    classify_commit_message,
    commit_provenance,
    pr_risk,
)


def _write_graph(repo: Path) -> None:
    """base.py imported by many files; lonely.py imported by nobody."""
    src = repo / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "base.py").write_text(
        "def base_fn(x: int) -> int:\n"
        "    total = 0\n"
        "    for i in range(x):\n"
        "        if i % 2 == 0:\n"
        "            total += i\n"
        "        else:\n"
        "            total -= i\n"
        "    return total\n",
        encoding="utf-8",
    )
    for name in ("a", "b", "c", "d"):
        (src / f"{name}.py").write_text(
            f"from src.base import base_fn\n\ndef {name}_fn(x: int) -> int:\n    return base_fn(x)\n",
            encoding="utf-8",
        )
    (src / "lonely.py").write_text(
        "def lonely_fn() -> int:\n    return 1\n",
        encoding="utf-8",
    )


def _index_all(repo: Path, cache_root: Path) -> None:
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

    cap = SemanticFileMemoryCapability(cache_root)
    for py in sorted((repo / "src").glob("*.py")):
        cap.summarize_file(py)


def _index_repo(repo: Path) -> None:
    """Build the repo's own code index -- where the blast radius now comes from."""
    CodeContextEngine(repo).index_repo()


def test_pr_risk_rises_with_blast_radius_and_missing_tests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write_graph(repo)
    _index_all(repo, cache)
    _index_repo(repo)

    high = pr_risk(repo_root=repo, lemoncrow_root=cache, paths=["src/base.py"])
    low = pr_risk(repo_root=repo, lemoncrow_root=cache, paths=["src/lonely.py"])

    # base.py has 4 importers + no tests + branchy complexity; lonely.py has none.
    assert high["overall_score"] > low["overall_score"]
    assert high["file_count"] == 1
    base_file = high["files"][0]
    assert base_file["factors"]["blast_radius"]["impacted_files"] >= 4
    assert base_file["factors"]["test_gap"]["missing_tests"] is True
    assert base_file["factors"]["complexity"]["factor"] > 0.0
    assert high["overall_tier"] in {"low", "medium", "high", "critical"}
    assert 0.0 <= high["overall_score"] <= 1.0


def test_pr_risk_test_gap_lowers_score_when_tests_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write_graph(repo)
    # Add a linked test for base so the test-gap factor is removed.
    tests = repo / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test_base.py").write_text(
        "from src.base import base_fn\n\ndef test_base() -> None:\n    assert base_fn(2) == 0\n",
        encoding="utf-8",
    )
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

    cap = SemanticFileMemoryCapability(cache)
    for py in sorted((repo / "src").glob("*.py")):
        cap.summarize_file(py)
    cap.summarize_file(tests / "test_base.py")
    _index_repo(repo)

    result = pr_risk(repo_root=repo, lemoncrow_root=cache, paths=["src/base.py"])
    base_file = result["files"][0]
    # The linked test is discovered, so the test-gap penalty is gone.
    assert base_file["factors"]["test_gap"]["missing_tests"] is False
    assert base_file["factors"]["test_gap"]["factor"] == 0.0


def test_pr_risk_blast_uses_repo_import_graph(tmp_path: Path) -> None:
    """FR12: the blast radius reads this repository's index, not a machine-wide one.

    ``repo_b`` sits beside ``repo_a`` and imports ``repo_a.base``. The semantic
    file index both are summarised into is keyed by absolute path and resolves
    that import to repo_a's file, so the old implementation counted repo_b's
    module as an importer -- and repo_b's *test* as coverage, clearing a test
    gap repo_a genuinely has.
    """
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

    cache = tmp_path / "cache"
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "base.py").write_text("def base_fn(x: int) -> int:\n    return x\n", encoding="utf-8")
    (repo_a / "local.py").write_text(
        "from base import base_fn\n\ndef local_fn(x: int) -> int:\n    return base_fn(x)\n",
        encoding="utf-8",
    )
    (repo_b / "other.py").write_text(
        "from repo_a.base import base_fn\n\ndef other_fn(x: int) -> int:\n    return base_fn(x)\n",
        encoding="utf-8",
    )
    (repo_b / "test_other.py").write_text(
        "from repo_a.base import base_fn\n\ndef test_other() -> None:\n    assert base_fn(1) == 1\n",
        encoding="utf-8",
    )
    for repo in (repo_a, repo_b):
        _index_repo(repo)

    # One machine-wide index spanning both repositories -- the state the old
    # implementation read, and the reason this test needs two of them.
    cap = SemanticFileMemoryCapability(cache)
    for path in (repo_a / "base.py", repo_a / "local.py", repo_b / "other.py", repo_b / "test_other.py"):
        cap.summarize_file(path)
    leaked = cap.change_impact(str(repo_a / "base.py"))
    assert any(
        "repo_b" in importer for importer in leaked["direct_importers"]
    ), "fixture no longer reproduces the cross-repository leak this test guards"

    scored = pr_risk(repo_root=repo_a, lemoncrow_root=cache, paths=["base.py"])["files"][0]
    blast = scored["factors"]["blast_radius"]

    assert blast["impacted_files"] == 1  # repo_a/local.py, and nothing from repo_b
    assert not any("repo_b" in str(entry) for entry in blast["affected_tests"])
    assert scored["factors"]["test_gap"]["missing_tests"] is True
    assert scored["objective"] == "exhaustive"
    assert scored["truncated"] is False


def test_pr_risk_degrades_when_the_index_is_unavailable(tmp_path: Path) -> None:
    """An unindexed repo costs the blast factor, not the report.

    ``open_file_graph`` raises rather than analysing nothing, which is right for
    an enumerative tool. pr_risk's contract is fail-open per factor, so the
    raise must not reach the caller.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "base.py").write_text("def base_fn() -> int:\n    return 1\n", encoding="utf-8")

    scored = pr_risk(repo_root=repo, lemoncrow_root=tmp_path / "cache", paths=["base.py"])["files"][0]

    assert scored["objective"] == "partial"
    assert scored["factors"]["blast_radius"]["impacted_files"] == 0
    assert scored["factors"]["blast_radius"]["factor"] == 0.0
    assert scored["factors"]["blast_radius"]["reason"].startswith("code index unavailable:")
    # No data is not a known test gap: the 0.20 penalty must not be charged for
    # a gap nobody looked for, and "low" is a verdict no closure was walked for.
    assert scored["factors"]["test_gap"]["missing_tests"] is None
    assert scored["factors"]["test_gap"]["factor"] == 0.0
    assert scored["risk_level"] == "unknown"
    assert scored["score"] < _W_TESTGAP  # the penalty is not in there


def test_pr_risk_names_a_rebuilding_index_as_such(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuilding and absent are two conditions, and the reason has to say which.

    ``IndexRebuilding`` deliberately does not subclass ``CodeIntelUnavailable``
    so a transient rebuild cannot be swallowed as "no index here". Catching both
    into one reason string puts that distinction back out of the caller's reach:
    one resolves itself on the next call, the other never does.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "base.py").write_text("def base_fn() -> int:\n    return 1\n", encoding="utf-8")

    def _rebuilding(root: Path) -> object:
        raise IndexRebuilding(root, "index is mid-write")

    monkeypatch.setattr(sys.modules[pr_risk.__module__], "open_file_graph", _rebuilding)
    scored = pr_risk(repo_root=repo, lemoncrow_root=tmp_path / "cache", paths=["base.py"])["files"][0]

    assert scored["objective"] == "partial"
    assert scored["factors"]["blast_radius"]["reason"].startswith("code index is rebuilding:")


def test_pr_risk_does_not_claim_an_exhaustive_zero_for_an_unindexed_file(tmp_path: Path) -> None:
    """A file the index has never seen is pr_risk's common case, not an edge one.

    ``blast_radius`` answers for any path -- zero importers, zero tests -- and
    stamps the import table's ``exhaustive`` on it. Copied through unchanged that
    reads as "we looked at everything and nothing imports this file, and it has
    no tests", which is the completeness failure the contract exists to prevent,
    reintroduced at file granularity. A PR that adds a file is exactly the input
    this tool is built for, so the downgrade has to be per file.
    """
    repo = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write_graph(repo)
    _index_all(repo, cache)
    _index_repo(repo)
    # Added after the index was built -- what a PR under review looks like.
    (repo / "src" / "brand_new.py").write_text("def new_fn() -> int:\n    return 1\n", encoding="utf-8")

    result = pr_risk(repo_root=repo, lemoncrow_root=cache, paths=["src/brand_new.py", "src/lonely.py"])
    scored = {entry["path"]: entry for entry in result["files"]}
    unindexed = scored["src/brand_new.py"]
    indexed = scored["src/lonely.py"]

    assert unindexed["objective"] == "partial"
    assert unindexed["factors"]["blast_radius"]["reason"] == "file not in the code index"
    assert unindexed["factors"]["test_gap"]["missing_tests"] is None
    assert unindexed["factors"]["test_gap"]["factor"] == 0.0
    assert unindexed["risk_level"] == "unknown"

    # lonely.py IS in the index and genuinely has no importers and no tests --
    # the same zeroes, earned. One file degrading must not degrade the other.
    assert indexed["objective"] == "exhaustive"
    assert "reason" not in indexed["factors"]["blast_radius"]
    assert indexed["factors"]["test_gap"]["missing_tests"] is True

    # The envelope reports the mix honestly: one row it could not read makes the
    # whole answer partial, while a tier is still earned from the row it could.
    assert result["objective"] == "partial"
    assert result["overall_tier"] != "unknown"


def test_envelope_reports_unknown_when_nothing_could_be_read(tmp_path: Path) -> None:
    """A report made only of unreadable files must not headline a verdict.

    Each such file scores 0.0, so the maximum over them is 0.0 and the tier table
    calls that "low" -- the envelope announcing exactly what every row beneath it
    declines to say. The per-file downgrade made this quieter rather than louder:
    dropping the unearned test-gap penalty lowered the score the headline reads.
    """
    repo = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write_graph(repo)
    _index_all(repo, cache)
    _index_repo(repo)
    # Added after the index was built, and the only path asked about.
    (repo / "src" / "brand_new.py").write_text("def new_fn() -> int:\n    return 1\n", encoding="utf-8")

    result = pr_risk(repo_root=repo, lemoncrow_root=cache, paths=["src/brand_new.py"])

    assert result["objective"] == "partial"
    assert result["overall_tier"] == "unknown"
    assert result["files"][0]["objective"] == "partial"
    # The score is deliberately not asserted to be zero: complexity still reads
    # the file itself, so a small number is correct. What must not happen is that
    # number being dressed as a verdict by the tier table.
    assert result["overall_score"] > 0.0


def test_code_health_seam_resolves_a_relative_root_against_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR12 one layer up: a relative root must not mean the daemon's cwd.

    The blast radius now opens ``<repo_root>/.lemoncrow/workspace``, so the root
    this seam hands pr_risk decides which repository's import graph is read. The
    sibling file-graph kinds already resolve through ``_code_repo_root``; before
    this, ``graph(kind="pr_risk", repo_root=".")`` against a daemon standing
    somewhere else silently degraded every file instead.
    """
    from lemoncrow.gateway.adapters import mcp_server
    from lemoncrow.pro.capabilities import code_health

    workspace = tmp_path / "workspace"
    elsewhere = tmp_path / "elsewhere"
    workspace.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(mcp_server, "_workspace_root", lambda: workspace)
    monkeypatch.chdir(elsewhere)

    seen: dict[str, Path] = {}

    def _capture(**kwargs: object) -> dict[str, object]:
        seen["repo_root"] = kwargs["repo_root"]  # type: ignore[assignment]
        return {"kind": "pr_risk"}

    monkeypatch.setattr(code_health, "pr_risk", _capture)
    mcp_server._op_graph_code_health(
        kind="pr_risk", path=None, paths=["a.py"], limit=50, query=None, enable=None, repo_root="sub"
    )

    assert seen["repo_root"] == (workspace / "sub").resolve()


def test_pr_risk_empty_paths_yields_zero_fail_open(tmp_path: Path) -> None:
    # Direct call with no targets must not raise -- it returns a valid zero shape.
    # (The graph-tool seam separately rejects empty targets with a ValueError.)
    result = pr_risk(repo_root=tmp_path, lemoncrow_root=tmp_path / "cache", paths=[])
    assert result["kind"] == "pr_risk"
    assert result["overall_score"] == 0.0
    assert result["overall_tier"] == "low"
    assert result["file_count"] == 0


def test_classify_commit_message_samples() -> None:
    cases = {
        "fix: null deref in parser": "bugfix",
        "Fixed a crash when input is empty": "bugfix",
        "feat(api): add pagination support": "feature",
        "Implement retry logic for uploads": "feature",
        "refactor: extract helper from monolith": "refactor",
        "perf: optimize hot loop": "perf",
        "Rename FooService to BarService": "rename",
        'Revert "feat: add pagination"': "revert",
        "docs: update README": "docs",
        "test: add coverage for edge cases": "test",
        "chore: bump dependencies": "chore",
    }
    for message, expected in cases.items():
        verdict = classify_commit_message(message)
        assert verdict["category"] == expected, f"{message!r} -> {verdict}"
        assert 0.0 < verdict["confidence"] <= 1.0


def test_classify_commit_message_revert_body_and_conventional_priority() -> None:
    # Conventional prefix wins over free-text keywords in the body.
    conv = classify_commit_message("feat: add thing\n\nthis also fixes a bug")
    assert conv["category"] == "feature"
    assert conv["signal"] == "conventional_prefix"
    # Revert detected from the body signature even with a plain subject.
    rev = classify_commit_message("Roll back change\n\nThis reverts commit abc123.")
    assert rev["category"] == "revert"


def test_classify_commit_message_file_shape_fallback() -> None:
    docs = classify_commit_message("misc", ["docs/guide.md", "README.md"])
    assert docs["category"] == "docs"
    assert docs["signal"] == "file_shape"
    tests = classify_commit_message("misc", ["tests/test_a.py", "tests/test_b.py"])
    assert tests["category"] == "test"


def _git(args: list[str], repo: Path) -> None:
    import os

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@t.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@t.com",
    }
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def test_commit_provenance_classifies_real_repo(tmp_path: Path) -> None:
    try:
        import pygit2  # noqa: F401
    except ImportError:
        pytest.skip("pygit2 not available")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.name", "T"], repo)
    _git(["config", "user.email", "t@t.com"], repo)
    (repo / "README.md").write_text("init", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "initial"], repo)

    commits = [
        ("feat: add module", "mod.py", "x = 1\n"),
        ("fix: correct off-by-one", "mod.py", "x = 2\n"),
        ("docs: document module", "GUIDE.md", "guide\n"),
    ]
    for message, fname, content in commits:
        (repo / fname).write_text(content, encoding="utf-8")
        _git(["add", "-A"], repo)
        _git(["commit", "-m", message], repo)

    result = commit_provenance(repo_root=repo, path=None, limit=50)
    assert result["kind"] == "commit_provenance"
    cats = result["by_category"]
    assert cats.get("feature", 0) >= 1
    assert cats.get("bugfix", 0) >= 1
    assert cats.get("docs", 0) >= 1

    # Path-scoped: only commits touching mod.py.
    scoped = commit_provenance(repo_root=repo, path="mod.py", limit=50)
    scoped_cats = scoped["by_category"]
    assert scoped_cats.get("feature", 0) >= 1
    assert scoped_cats.get("bugfix", 0) >= 1
    assert "docs" not in scoped_cats  # GUIDE.md commit excluded


def test_commit_provenance_fail_open_non_git(tmp_path: Path) -> None:
    result = commit_provenance(repo_root=tmp_path / "not_a_repo", path=None, limit=10)
    assert result["kind"] == "commit_provenance"
    assert result["commit_count"] == 0
    assert result["commits"] == []
