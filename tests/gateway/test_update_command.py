"""Tests for ``lc update`` — install-method detection and the GitHub
release update channel.

LemonCrow ships only two ways (git checkout, GitHub-release install), so these
tests pin the two update paths and guard the release channel against drifting
away from ``scripts/install.sh``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from lemoncrow.gateway.cli import cli
from lemoncrow.gateway.cli.commands import update as update_mod

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke(tmp_path: Path, *args: str, stdin: str | None = None) -> Result:
    root = tmp_path / ".lemoncrow"
    root.mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    return runner.invoke(cli, ["--root", str(root), "update", *args], input=stdin)


# --------------------------------------------------------------------------- #
# detection                                                                   #
# --------------------------------------------------------------------------- #


def test_detect_release_when_no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_mod, "_git_project_root", lambda: None)
    assert update_mod._detect_method() == ("release", None)


def test_detect_git_when_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(update_mod, "_git_project_root", lambda: tmp_path)
    assert update_mod._detect_method() == ("git", str(tmp_path))


# --------------------------------------------------------------------------- #
# release channel (end-user install)                                          #
# --------------------------------------------------------------------------- #


def test_release_already_up_to_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(update_mod, "_detect_method", lambda: ("release", None))
    monkeypatch.setattr(update_mod, "current_version", "1.2.3")
    monkeypatch.setattr(update_mod, "_github_latest_version", lambda: "1.2.3")

    def _boom() -> bool:
        raise AssertionError("must not apply when already current")

    monkeypatch.setattr(update_mod, "_update_via_release", _boom)

    res = _invoke(tmp_path)
    assert res.exit_code == 0, res.output
    assert "Already up-to-date" in res.output


def test_release_check_only_reports_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(update_mod, "_detect_method", lambda: ("release", None))
    monkeypatch.setattr(update_mod, "current_version", "1.0.0")
    monkeypatch.setattr(update_mod, "_github_latest_version", lambda: "1.4.0")

    def _boom() -> bool:
        raise AssertionError("--check must not apply")

    monkeypatch.setattr(update_mod, "_update_via_release", _boom)

    res = _invoke(tmp_path, "--check")
    assert res.exit_code == 1, res.output  # --check: exit 1 when an update is available
    assert "Update available: 1.0.0 → 1.4.0" in res.output


def test_release_apply_runs_installer_and_records_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(update_mod, "_detect_method", lambda: ("release", None))
    monkeypatch.setattr(update_mod, "current_version", "1.0.0")
    monkeypatch.setattr(update_mod, "_github_latest_version", lambda: "1.4.0")

    calls: dict[str, object] = {}
    monkeypatch.setattr(update_mod, "_update_via_release", lambda: calls.setdefault("applied", True))

    def _record(**kwargs: object) -> None:
        calls["state"] = kwargs

    monkeypatch.setattr(update_mod, "write_update_state", _record)

    # "y": on a fork build the release path asks before leaving it (FR11). On a
    # build whose distribution IS the release channel the prompt never appears
    # and the answer goes unread.
    res = _invoke(tmp_path, stdin="y\n")
    assert res.exit_code == 0, res.output
    assert calls.get("applied") is True
    assert "Updated from 1.0.0 → 1.4.0" in res.output
    state = calls.get("state")
    assert isinstance(state, dict)
    assert state["previous_version"] == "1.0.0"
    assert state["current_version"] == "1.4.0"
    assert state["method"] == "release"


def test_update_via_release_downloads_and_runs_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The release path must download the published install.sh and run it."""
    seen: dict[str, object] = {}

    class _Resp:
        def read(self) -> bytes:  # not used by copyfileobj path but kept simple
            return b"echo installed\n"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_urlopen(url: str, timeout: int = 0) -> _Resp:
        seen["url"] = url
        return _Resp()

    def _fake_copyfileobj(src: object, dst: object) -> None:
        dst.write(b"echo installed\n")  # type: ignore[attr-defined]

    def _fake_run(cmd: list[str], env: dict[str, str], timeout: int) -> object:
        seen["cmd"] = cmd
        seen["non_interactive"] = env.get("LEMONCROW_NON_INTERACTIVE")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(update_mod.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(update_mod.shutil, "copyfileobj", _fake_copyfileobj)
    monkeypatch.setattr(update_mod.shutil, "which", lambda _name: "/usr/bin/bash")
    monkeypatch.setattr(update_mod.subprocess, "run", _fake_run)

    assert update_mod._update_via_release() is True
    assert seen["url"] == f"{update_mod._RELEASE_LATEST_URL}/{update_mod._INSTALLER_ASSET}"
    assert isinstance(seen["cmd"], list) and seen["cmd"][0] == "bash"
    assert seen["non_interactive"] == "1"


# --------------------------------------------------------------------------- #
# drift guard — update channel must match scripts/install.sh                   #
# --------------------------------------------------------------------------- #


def test_release_channel_matches_install_script() -> None:
    install_sh = (_REPO_ROOT / "scripts" / "install.sh").read_text("utf-8")
    # update.py and install.sh must point at the same GitHub release base.
    assert update_mod._RELEASE_LATEST_URL in install_sh
    assert update_mod._GH_REPO in install_sh
    # install.sh is the asset update re-runs; it is published by the release job.
    assert update_mod._INSTALLER_ASSET == "install.sh"


def test_update_has_no_pypi_or_legacy_binary_channel() -> None:
    src = (_REPO_ROOT / "src" / "lemoncrow" / "gateway" / "cli" / "commands" / "update.py").read_text("utf-8")
    # LemonCrow is not on PyPI and ships no PyInstaller "lemoncrow-binaries" asset.
    # Guard against the *usage* drifting back, not the word in the docstring.
    assert "pypi.org" not in src.lower()
    assert "_pypi_latest_version" not in src
    assert "lemoncrow-binaries" not in src


# --------------------------------------------------------------------------- #
# fork build — no update leaves it for an upstream release (PRD-739 FR11)      #
# --------------------------------------------------------------------------- #


def _fork_release_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """A release install of the fork build, with an upstream release available."""
    monkeypatch.setattr(update_mod, "_detect_method", lambda: ("release", None))
    monkeypatch.setattr(update_mod, "current_version", "1.0.0")
    monkeypatch.setattr(update_mod, "_github_latest_version", lambda: "1.4.0")
    assert update_mod._is_fork_build(), "this checkout must identify as a fork build"


def test_check_reports_fork_distribution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fork_release_install(monkeypatch)

    res = _invoke(tmp_path, "--check", "--json")
    assert res.exit_code == 1, res.output  # --check: exit 1 when an update is available
    # Additive only: `distribution` joins the existing keys, none of which move.
    assert json.loads(res.output.strip().splitlines()[-1]) == {
        "current_version": "1.0.0",
        "remote_version": "1.4.0",
        "method": "release",
        "update_available": True,
        "distribution": update_mod.DISTRIBUTION_REPO,
    }

    human = _invoke(tmp_path, "--check")
    assert update_mod.DISTRIBUTION_REPO in human.output
    assert update_mod.UPSTREAM_REPO in human.output  # --check names where a release update would come from


def test_release_update_refused_on_fork_build_without_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fork_release_install(monkeypatch)

    # Recorders, not raisers: CliRunner turns a raised AssertionError into the
    # same exit code 1 a refusal produces, so a raiser here would pass whether
    # the installer ran or not.
    calls: dict[str, object] = {}
    monkeypatch.setattr(update_mod, "_update_via_release", lambda: calls.setdefault("applied", True))

    def _record(**kwargs: object) -> None:
        calls["state"] = kwargs

    monkeypatch.setattr(update_mod, "write_update_state", _record)

    res = _invoke(tmp_path, stdin="\n")  # an empty answer takes the default, which must be No
    assert res.exit_code == 1, res.output
    assert calls == {}, "a refused update must neither run the installer nor record update state"
    assert f"Refused — still on the {update_mod.DISTRIBUTION_REPO} build." in res.output
    assert update_mod._fork_update_command() in res.output


def test_release_update_allowed_with_allow_upstream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fork_release_install(monkeypatch)

    calls: dict[str, object] = {}
    monkeypatch.setattr(update_mod, "_update_via_release", lambda: calls.setdefault("applied", True))

    def _record(**kwargs: object) -> None:
        calls["state"] = kwargs

    monkeypatch.setattr(update_mod, "write_update_state", _record)

    res = _invoke(tmp_path, "--allow-upstream")  # no stdin: reaching a prompt would abort
    assert res.exit_code == 0, res.output
    assert calls.get("applied") is True
    assert "Updated from 1.0.0 → 1.4.0" in res.output


def test_git_update_unchanged_on_fork_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The guard is release-only: a git checkout of the fork already pulls the fork."""
    project_root = tmp_path / "fork-clone"
    project_root.mkdir()
    monkeypatch.setattr(update_mod, "_detect_method", lambda: ("git", str(project_root)))
    monkeypatch.setattr(update_mod, "current_version", "1.0.0")
    monkeypatch.setattr(update_mod, "_git_remote_version", lambda _root: "1.4.0")

    calls: dict[str, object] = {}

    def _apply(root: str) -> bool:
        calls["pulled"] = root
        return True

    def _record(**kwargs: object) -> None:
        calls["state"] = kwargs

    monkeypatch.setattr(update_mod, "_update_git", _apply)
    monkeypatch.setattr(update_mod, "write_update_state", _record)

    res = _invoke(tmp_path)  # no stdin: reaching a confirmation prompt would abort
    assert res.exit_code == 0, res.output
    assert calls["pulled"] == str(project_root)
    assert "--allow-upstream" not in res.output
