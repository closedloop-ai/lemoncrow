"""The pro mypyc group must be initialised by one thread, once.

Every lazy ``from lemoncrow.pro...`` on the code path lands in a single mypyc
group whose modules initialise one another. Two threads entering it at
different points can each observe the other's half-built module, and the result
is ``AttributeError: module 'X' has no attribute 'Y'`` on the first cold call,
from an import that succeeds every time afterwards -- which is why it reads as
a phantom rather than a bug.

It is not a phantom. Two barrier-synchronised imports of ``code_context`` and
``scoped_context`` in a fresh interpreter fail 40 times out of 40::

    AttributeError: module 'lemoncrow.pro.capabilities.code_context.budget'
                    has no attribute 'BudgetPacker'

and serialising them the way :func:`_warm_pro_code_modules` does makes the same
40 runs pass. The daemon builds that shape unaided: ``_warm_stdio_code_index``
constructs the engine on a background thread while the first request is served
on another, so the collision needs no unlucky client -- only a request that
arrives during startup.

These tests pin the serialisation. The cold-interpreter race itself needs a
fresh process per attempt, so it stays a recorded reproduction rather than a
suite test.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from lemoncrow.gateway.adapters import mcp_server


@pytest.fixture
def cold(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Reset the warm flag and record what a warm pass imports."""
    imported: list[str] = []

    def fake_import(name: str) -> Any:
        imported.append(name)
        return object()

    monkeypatch.setattr(mcp_server, "_pro_modules_warmed", False)
    monkeypatch.setattr(mcp_server.importlib, "import_module", fake_import)
    return imported


def test_warm_imports_every_module_on_the_code_path(cold: list[str]) -> None:
    mcp_server._warm_pro_code_modules()
    assert cold == list(mcp_server._PRO_CODE_PATH_MODULES)


def test_warm_runs_once(cold: list[str]) -> None:
    """Steady state must be a bool read, not a repeated import sweep."""
    mcp_server._warm_pro_code_modules()
    mcp_server._warm_pro_code_modules()
    assert cold == list(mcp_server._PRO_CODE_PATH_MODULES)


def test_concurrent_callers_produce_exactly_one_warm_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: N threads, one initialisation, no interleaving."""
    imported: list[str] = []
    entered = threading.Barrier(8)

    def slow_import(name: str) -> Any:
        # Widen the window a real import would leave open.
        threading.Event().wait(0.001)
        imported.append(name)
        return object()

    monkeypatch.setattr(mcp_server, "_pro_modules_warmed", False)
    monkeypatch.setattr(mcp_server.importlib, "import_module", slow_import)

    def call() -> None:
        entered.wait()
        mcp_server._warm_pro_code_modules()

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert imported == list(mcp_server._PRO_CODE_PATH_MODULES)


def test_warm_is_reentrant(monkeypatch: pytest.MonkeyPatch) -> None:
    """These helpers call one another; a plain Lock would self-deadlock.

    Guarded by a timeout because the failure mode is a hang, and a hung test
    that never asserts is indistinguishable from a slow one.
    """
    monkeypatch.setattr(mcp_server, "_pro_modules_warmed", False)

    def reentrant_import(name: str) -> Any:
        mcp_server._warm_pro_code_modules()
        return object()

    monkeypatch.setattr(mcp_server.importlib, "import_module", reentrant_import)

    done = threading.Event()

    def call() -> None:
        mcp_server._warm_pro_code_modules()
        done.set()

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    assert done.wait(timeout=5), "_warm_pro_code_modules deadlocked against itself"


def test_a_failed_import_does_not_propagate(cold: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The warm is an optimisation; the caller's own import owns the error.

    Raising here would turn "pro is unavailable" into a failure reported from a
    helper that has no idea what the caller wanted, several frames from the
    import that actually matters.
    """

    def boom(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(mcp_server.importlib, "import_module", boom)
    mcp_server._warm_pro_code_modules()  # must not raise


def test_the_engine_helper_warms_before_it_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wiring check: the guard is worthless if the entry points skip it."""
    calls: list[str] = []
    monkeypatch.setattr(mcp_server, "_warm_pro_code_modules", lambda: calls.append("warm"))
    monkeypatch.setattr(
        mcp_server._code_engine_cache,
        "get",
        lambda key, repo_root, build: (object(), "fresh"),
    )

    mcp_server._code_context_engine(".")

    assert calls == ["warm"]
