"""F1 -- file-graph analytics over the repo's own ``imports`` table.

The shipped ``blast_radius`` / ``dead_code`` / ``cycles`` / ``coupling`` /
``topology`` operations read ``~/.lemoncrow/semantic_file_index.json``: a
machine-global, cross-repo, LRU-bounded cache populated opportunistically by
``summarize_file``. On this repo that was 77 files mixed across four unrelated
projects -- ``dead_code`` called 30 of them dead, ``topology`` reported modules
under ``~/.claude/plugins/``, and ``blast_radius`` on a file with three real
importers returned an empty list and risk "low".

This module answers the same questions from ``code_context.imports``, which is
built from the repo and only the repo.

**The limitation that made the old implementation look plausible.** Only
intra-repo imports resolve: on this repo 6,007 of 8,096 ``imports`` rows have a
NULL ``target_file`` (stdlib and third-party, plus any resolution failure).
Every response therefore reports ``resolved_edges`` and ``unresolved_edges``, so
a caller can tell a genuinely small blast radius from an unresolved one. An
analysis that quietly returns "no importers" is indistinguishable from a broken
one, which is exactly how this bug survived.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any

from lemoncrow.infra.code_intel.store import CodeIntelStore, IndexSnapshot

__all__ = ["FileGraph", "open_file_graph"]

_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__", "testing"})

# Blast-radius tiers by transitive-importer count. Deliberately coarse: the
# number itself is in the response, and a finer scale would imply a precision
# the name-resolution below does not have.
_RISK_TIERS: tuple[tuple[int, str], ...] = ((0, "low"), (5, "medium"), (20, "high"))
_RISK_TOP = "critical"


def _is_test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    name = parts[-1]
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return True
    return any(part in _TEST_DIR_NAMES for part in parts[:-1])


def _risk_for(count: int) -> str:
    for threshold, tier in _RISK_TIERS:
        if count <= threshold:
            return tier
    return _RISK_TOP


def _module_of(path: str) -> str:
    parent = PurePosixPath(path).parent
    return str(parent) if str(parent) != "." else ""


def _sccs(nodes: Iterable[str], forward: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's strongly-connected components, iteratively.

    Iterative because a repo import graph is deep enough to blow the recursion
    limit, and a crash in analytics is worse than a slow answer.
    """
    counter = 0
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    components: list[list[str]] = []

    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work: list[tuple[str, Iterator[str]]] = [(root, iter(sorted(forward.get(root, set()))))]
        while work:
            node, successors = work[-1]
            descended = False
            for successor in successors:
                if successor not in index:
                    index[successor] = low[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack[successor] = True
                    work.append((successor, iter(sorted(forward.get(successor, set())))))
                    descended = True
                    break
                if on_stack.get(successor):
                    low[node] = min(low[node], index[successor])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


@dataclass
class _Edges:
    forward: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    #: Every file that appears as an import *source*, resolved or not. A file
    #: whose only imports are stdlib still proves its language was analysed.
    sources: set[str] = field(default_factory=set)
    resolved: int = 0
    unresolved: int = 0


class FileGraph:
    """Repo-scoped file dependency graph built from ``code_context.imports``.

    Holds an open read-only store; use as a context manager or call
    :meth:`close`.
    """

    def __init__(self, store: CodeIntelStore, repo_root: Path) -> None:
        self.repo_root: Path = repo_root
        self._store: CodeIntelStore = store
        self._snapshot: IndexSnapshot = store.snapshot()
        self._files: dict[str, str] = {row.file_path: row.language for row in store.files()}
        self._edges: _Edges = self._build_edges()
        self._import_languages: frozenset[str] = frozenset(
            self._files[source] for source in self._edges.sources if source in self._files
        )
        self._entry_points: frozenset[str] = self._resolve_entry_points()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> FileGraph:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._store.close()

    # -- construction ------------------------------------------------------

    def _build_edges(self) -> _Edges:
        edges = _Edges()
        for row in self._store.imports():
            edges.sources.add(row.source_file)
            if row.target_file is None:
                edges.unresolved += 1
                continue
            edges.resolved += 1
            edges.forward[row.source_file].add(row.target_file)
            edges.reverse[row.target_file].add(row.source_file)
        return edges

    def _resolve_entry_points(self) -> frozenset[str]:
        """Files that are reachable without anything importing them.

        Without this, every CLI entry point and every package ``__init__`` reads
        as dead code -- which is how a dead-code report becomes noise nobody
        acts on.
        """
        entries: set[str] = set()
        for path in self._store_paths():
            name = PurePosixPath(path).name
            if name in {"__main__.py", "__init__.py", "conftest.py", "setup.py"} or _is_test_path(path):
                entries.add(path)
        entries |= self._console_script_paths()
        return frozenset(entries)

    def _console_script_paths(self) -> set[str]:
        """Files named by ``[project.scripts]`` / ``[project.gui-scripts]``.

        Fail-open: a missing or malformed pyproject.toml costs entry-point
        detection, not the whole analysis.
        """
        pyproject = self.repo_root / "pyproject.toml"
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError, ValueError):
            return set()
        project = data.get("project")
        if not isinstance(project, dict):
            return set()
        targets: list[str] = []
        for table_name in ("scripts", "gui-scripts"):
            table = project.get(table_name)
            if isinstance(table, dict):
                targets.extend(str(value) for value in table.values())

        suffixes: set[str] = set()
        for target in targets:
            module = target.split(":", 1)[0].strip()
            if not module:
                continue
            as_path = module.replace(".", "/")
            suffixes.add(f"{as_path}.py")
            suffixes.add(f"{as_path}/__init__.py")

        # Match by suffix so a src/ layout resolves without hard-coding it.
        return {
            path for path in self._store_paths() for suffix in suffixes if path == suffix or path.endswith(f"/{suffix}")
        }

    def _store_paths(self) -> list[str]:
        return [row.file_path for row in self._store.files()]

    # -- shared response envelope -----------------------------------------

    def _envelope(self) -> dict[str, Any]:
        """Provenance every response carries.

        ``unresolved_edges`` is the honest caveat: a small answer may mean a
        small dependency footprint or an unresolved one, and only this number
        tells them apart.
        """
        return {
            "analyzed_files": len(self._files),
            "resolved_edges": self._edges.resolved,
            "unresolved_edges": self._edges.unresolved,
            "engine_index_version": self._snapshot.index_version,
        }

    def _scope(self, paths: list[str] | None) -> frozenset[str] | None:
        """Normalise a caller's path filter to repo-relative prefixes."""
        if not paths:
            return None
        prefixes: set[str] = set()
        for raw in paths:
            candidate = Path(raw)
            if candidate.is_absolute():
                try:
                    candidate = candidate.resolve().relative_to(self.repo_root)
                except ValueError:
                    continue
            prefixes.add(candidate.as_posix().rstrip("/"))
        return frozenset(prefixes) or None

    @staticmethod
    def _in_scope(path: str, scope: frozenset[str] | None) -> bool:
        if scope is None:
            return True
        return any(path == prefix or path.startswith(f"{prefix}/") for prefix in scope)

    # -- operations --------------------------------------------------------

    def blast_radius(self, path: str) -> dict[str, Any]:
        """Reverse-import transitive closure of *path*, plus affected tests."""
        target = self._normalise(path)
        direct = sorted(self._edges.reverse.get(target, set()))

        seen: set[str] = set()
        frontier = list(direct)
        while frontier:
            current = frontier.pop()
            if current in seen or current == target:
                continue
            seen.add(current)
            frontier.extend(self._edges.reverse.get(current, set()))

        transitive = sorted(seen - set(direct))
        affected_tests = sorted(candidate for candidate in seen if _is_test_path(candidate))
        result = self._envelope()
        result.update(
            {
                "modified_file": target,
                "indexed": target in self._files,
                "direct_importers": direct,
                "transitive_importers": transitive,
                "affected_tests": affected_tests,
                "risk_level": _risk_for(len(seen)),
            }
        )
        return result

    def dead_code(self, limit: int = 50, paths: list[str] | None = None) -> dict[str, Any]:
        """Files nothing imports, excluding entry points.

        Restricted to languages the import extractor actually produced edges
        for. Without that, every Markdown, JSON, and shell file in the repo is
        "dead" -- 345 of 517 hits on this repo -- and the report becomes noise.
        The set is derived from the data, so a language the engine learns later
        starts being reported with no change here.
        """
        scope = self._scope(paths)
        dead = [
            path
            for path, language in sorted(self._files.items())
            if language in self._import_languages
            and not self._edges.reverse.get(path)
            and path not in self._entry_points
            and self._in_scope(path, scope)
        ]
        rows = [self._dead_row(path) for path in dead[: max(limit, 0)]]
        result = self._envelope()
        result.update(
            {
                "dead_file_count": len(dead),
                "dead_files": rows,
                "truncated": len(dead) > len(rows),
                "excluded_entry_points": len(self._entry_points),
                "import_languages": sorted(self._import_languages),
            }
        )
        return result

    def _dead_row(self, path: str) -> dict[str, Any]:
        symbols = self._store.symbols(file_path=path)
        exports = [row.symbol_name for row in symbols if row.parent_symbol is None]
        lines_total = max((row.end_line for row in symbols), default=0)
        return {
            "path": path,
            "language": self._files.get(path, ""),
            "exports": exports,
            "lines_total": lines_total,
            "efferent": len(self._edges.forward.get(path, set())),
        }

    def cycles(self, limit: int = 50, paths: list[str] | None = None) -> dict[str, Any]:
        """Import cycles: strongly-connected components of size >= 2."""
        scope = self._scope(paths)
        nodes = sorted(set(self._files) | set(self._edges.forward) | set(self._edges.reverse))
        found = [
            sorted(component)
            for component in _sccs(nodes, self._edges.forward)
            if len(component) >= 2 and any(self._in_scope(member, scope) for member in component)
        ]
        found.sort(key=lambda component: (-len(component), component[0]))
        rows = found[: max(limit, 0)]
        result = self._envelope()
        result.update(
            {
                "cycle_count": len(found),
                "cycles": rows,
                "truncated": len(found) > len(rows),
            }
        )
        return result

    def coupling(self, limit: int = 50, paths: list[str] | None = None) -> dict[str, Any]:
        """Afferent/efferent coupling and Martin instability ``Ce/(Ca+Ce)``."""
        scope = self._scope(paths)
        rows: list[dict[str, Any]] = []
        candidates = set(self._files) | set(self._edges.forward) | set(self._edges.reverse)
        for path in sorted(candidates):
            if not self._in_scope(path, scope):
                continue
            afferent = len(self._edges.reverse.get(path, set()))
            efferent = len(self._edges.forward.get(path, set()))
            total = afferent + efferent
            if total == 0:
                continue
            rows.append(
                {
                    "path": path,
                    "afferent": afferent,
                    "efferent": efferent,
                    "total_coupling": total,
                    "instability": round(efferent / total, 4),
                }
            )
        rows.sort(key=lambda row: (-int(row["total_coupling"]), str(row["path"])))
        top = rows[: max(limit, 0)]
        result = self._envelope()
        result.update(
            {
                "coupled_file_count": len(rows),
                "files": top,
                "truncated": len(rows) > len(top),
            }
        )
        return result

    def topology(self, limit: int = 50, paths: list[str] | None = None) -> dict[str, Any]:
        """Directory-level module graph, plus the most-coupled files."""
        scope = self._scope(paths)
        members: dict[str, set[str]] = defaultdict(set)
        for path in set(self._files) | set(self._edges.forward) | set(self._edges.reverse):
            members[_module_of(path)].add(path)

        depends: dict[str, set[str]] = defaultdict(set)
        for source, targets in self._edges.forward.items():
            source_module = _module_of(source)
            for target in targets:
                target_module = _module_of(target)
                if target_module != source_module:
                    depends[source_module].add(target_module)

        afferent_modules: dict[str, int] = defaultdict(int)
        for targets in depends.values():
            for target_module in targets:
                afferent_modules[target_module] += 1

        rows: list[dict[str, Any]] = [
            {
                "module": module,
                "files": len(paths_in_module),
                "depends_on": sorted(depends.get(module, set())),
                "efferent_modules": len(depends.get(module, set())),
                "afferent_modules": afferent_modules.get(module, 0),
            }
            for module, paths_in_module in sorted(members.items())
            if scope is None or any(self._in_scope(path, scope) for path in paths_in_module)
        ]
        rows.sort(
            key=lambda row: (
                -(int(row["efferent_modules"]) + int(row["afferent_modules"])),
                str(row["module"]),
            )
        )
        top = rows[: max(limit, 0)]
        result = self._envelope()
        result.update(
            {
                "module_count": len(rows),
                "modules": top,
                "hotspots": self.coupling(limit=min(limit, 10), paths=paths)["files"],
                "truncated": len(rows) > len(top),
            }
        )
        return result

    # -- helpers -----------------------------------------------------------

    def _normalise(self, path: str) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(self.repo_root)
            except ValueError:
                return candidate.as_posix()
        return candidate.as_posix()


def open_file_graph(repo_root: Path | str = ".") -> FileGraph:
    """Open a :class:`FileGraph` over *repo_root*'s code index."""
    root = Path(repo_root).expanduser().resolve()
    return FileGraph(CodeIntelStore(root), root)
