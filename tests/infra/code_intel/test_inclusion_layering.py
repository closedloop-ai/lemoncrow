"""``lemoncrow.infra`` may not reach up into ``lemoncrow.pro`` at import time.

``infra/code_intel/inclusion.py`` owns the indexer's file-selection rules and
``pro`` calls down into them -- ``repo_map/graph.py`` for the whole-repo scan,
``code_context/engine.py`` for the incremental one. An import back up into
``pro`` at module scope would turn that into a real cycle; the reaches that
remain (``coverage.py`` reading the engine's scan and its Free-tier cap) are
deferred to call time on purpose, and this guard keeps them that way.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_INTEL = REPO_ROOT / "src" / "lemoncrow" / "infra" / "code_intel"
INCLUSION = CODE_INTEL / "inclusion.py"
_FORBIDDEN = "lemoncrow.pro"


def _imports_pro(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith(_FORBIDDEN)
    if isinstance(node, ast.Import):
        return any(alias.name.startswith(_FORBIDDEN) for alias in node.names)
    return False


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _import_time_imports(body: list[ast.stmt]) -> list[ast.stmt]:
    """Imports that execute when the module loads, ``if TYPE_CHECKING:`` aside."""
    found: list[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.append(node)
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                found.extend(_import_time_imports(node.body))
            found.extend(_import_time_imports(node.orelse))
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody):
                found.extend(_import_time_imports(block))
            for handler in node.handlers:
                found.extend(_import_time_imports(handler.body))
    return found


def test_code_intel_never_imports_pro_at_module_scope() -> None:
    offenders: dict[str, list[int]] = {}
    for path in sorted(CODE_INTEL.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = [node.lineno for node in _import_time_imports(tree.body) if _imports_pro(node)]
        if lines:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = lines
    assert not offenders, (
        "lemoncrow.infra.code_intel imports lemoncrow.pro at import time, which makes the "
        f"infra/pro dependency a real cycle -- defer the import to call time:\n{offenders}"
    )


def test_inclusion_never_imports_pro_at_any_scope() -> None:
    tree = ast.parse(INCLUSION.read_text(encoding="utf-8"), filename=str(INCLUSION))
    lines = sorted(
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom) and _imports_pro(node)
    )
    assert not lines, (
        "inclusion.py owns the indexer's file-selection rules and pro calls down into them, "
        f"so it must not reach back up into lemoncrow.pro at any scope (lines {lines})"
    )
