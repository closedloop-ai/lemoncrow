"""Read-only accessors over the closed engine's code-intel SQLite databases.

``src/lemoncrow/pro/`` owns the DDL, the extraction, and the write lock for
``code_context.sqlite`` / ``intel.sqlite`` / ``fts.sqlite`` / ``vectors.sqlite``.
Open code reads those databases and never writes to them: a ``--reindex`` drops
and rebuilds them, and concurrent writes race the autosync worker plus the
edit-triggered reindex thread. Every connection handed out here is opened with
``mode=ro`` so that rule is enforced structurally rather than by discipline --
an accidental ``INSERT`` raises ``sqlite3.OperationalError`` instead of
corrupting the engine's state.

Anything open code needs to *persist* belongs in
:mod:`lemoncrow.infra.code_intel.sidecar`, which is ours to write.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

__all__ = [
    "CODE_CONTEXT_DB",
    "FTS_DB",
    "INTEL_DB",
    "REPO_MAP_TAGS_DB",
    "VECTORS_DB",
    "CallEdgeRow",
    "CentralityRow",
    "CodeIntelStore",
    "CodeIntelUnavailable",
    "FileRow",
    "ImportRow",
    "IndexSnapshot",
    "ReferenceRow",
    "SymbolRow",
    "open_ro",
    "workspace_dir",
]

CODE_CONTEXT_DB = "code_context.sqlite"
INTEL_DB = "intel.sqlite"
FTS_DB = "fts.sqlite"
VECTORS_DB = "vectors.sqlite"
REPO_MAP_TAGS_DB = "repo_map_tags.sqlite"

# The engine holds WAL writers during a reindex; wait rather than fail fast.
_BUSY_TIMEOUT_MS = 5_000


class CodeIntelUnavailable(RuntimeError):
    """The engine's databases are absent or unreadable for this workspace."""


@dataclass(frozen=True)
class FileRow:
    """One row of ``code_context.files``."""

    file_path: str
    language: str
    content_hash: str
    size_bytes: int
    mtime_ns: int
    indexed_at: str


@dataclass(frozen=True)
class SymbolRow:
    """One row of ``code_context.symbols``."""

    symbol_id: str
    file_path: str
    language: str
    symbol_name: str
    qualified_name: str
    kind: str
    signature: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent_symbol: str | None
    content_hash: str


@dataclass(frozen=True)
class ImportRow:
    """One row of ``code_context.imports``.

    ``target_file`` is ``None`` for imports the engine could not resolve to a
    file inside the repo -- stdlib and third-party modules, and any intra-repo
    import whose resolution failed. Callers that treat a missing target as
    "no edge" silently understate the graph; report the unresolved count.
    """

    source_file: str
    raw_import: str
    target_file: str | None


@dataclass(frozen=True)
class CallEdgeRow:
    """One row of ``intel.call_edges``.

    The callee is stored as raw call text with no ``symbol_id``, so any reverse
    lookup against it is name-matched and over-reports on common names. Consumers
    must say which matching they used rather than implying resolution.
    """

    caller_symbol_name: str
    caller_qualified_name: str
    caller_file_path: str
    caller_start_line: int
    caller_end_line: int
    callee_name: str
    callee_short_name: str
    call_line: int
    call_column: int


@dataclass(frozen=True)
class ReferenceRow:
    """One row of ``intel."references"``."""

    symbol_name: str
    file_path: str
    line: int
    column: int
    end_column: int
    enclosing_symbol_name: str | None
    enclosing_qualified_name: str | None
    snippet: str


@dataclass(frozen=True)
class CentralityRow:
    """One row of ``intel.centrality_map``.

    ``name_key`` is a bare name, not a symbol id -- the same name-keyed graph
    limitation as :class:`CallEdgeRow`.
    """

    name_key: str
    score: float
    index_version: int


@dataclass(frozen=True)
class IndexSnapshot:
    """The engine generation a derived result was computed from.

    Stamp this onto anything built out of the engine's tables so a stale answer
    is detectable instead of silently wrong.
    """

    repo_id: str
    index_version: int
    indexer_semantics_version: int
    files: int
    symbols: int
    imports: int
    imports_resolved: int
    intel_available: bool
    call_edges: int
    references: int
    centrality: int


def workspace_dir(repo_root: Path | str = ".") -> Path:
    """Return ``<repo_root>/.lemoncrow/workspace``.

    *repo_root* is passed verbatim to
    :func:`~lemoncrow.core.foundation.paths.resolve_workspace_store_dir`, which
    skips git/marker discovery (and therefore never raises) when the caller
    already knows the root.
    """
    return resolve_workspace_store_dir(workspace_root=repo_root)


def open_ro(db: str, repo_root: Path | str = ".") -> sqlite3.Connection:
    """Open one of the engine's databases read-only.

    Raises :class:`CodeIntelUnavailable` when the workspace has not been indexed
    yet, so callers can distinguish "nothing there" from "never indexed".
    """
    path = workspace_dir(repo_root) / db
    if not path.exists():
        raise CodeIntelUnavailable(f"code-intel database not found: {path} (has this workspace been indexed?)")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:  # pragma: no cover - platform/permission dependent
        raise CodeIntelUnavailable(f"cannot open {path} read-only: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return conn


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


class CodeIntelStore:
    """Lazily-opened read-only handle over one workspace's code-intel databases.

    ``code_context.sqlite`` is required; ``intel.sqlite`` is optional and its
    accessors degrade to empty rather than raising, because the call graph is
    built by a later pass than the symbol index and can legitimately be absent.

    Use as a context manager, or call :meth:`close` -- connections are not
    closed on garbage collection.
    """

    def __init__(self, repo_root: Path | str = ".") -> None:
        self.repo_root: Path = Path(repo_root)
        self._code: sqlite3.Connection | None = None
        self._intel: sqlite3.Connection | None = None
        self._intel_checked: bool = False
        self._repo_id: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> CodeIntelStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for conn in (self._code, self._intel):
            if conn is not None:
                conn.close()
        self._code = None
        self._intel = None
        self._intel_checked = False

    # -- connections -------------------------------------------------------

    @property
    def code(self) -> sqlite3.Connection:
        """``code_context.sqlite`` -- files, symbols, imports."""
        conn = self._code
        if conn is None:
            conn = open_ro(CODE_CONTEXT_DB, self.repo_root)
            self._code = conn
        return conn

    @property
    def intel(self) -> sqlite3.Connection | None:
        """``intel.sqlite`` -- call edges, references, centrality. May be absent."""
        if not self._intel_checked:
            self._intel_checked = True
            try:
                self._intel = open_ro(INTEL_DB, self.repo_root)
            except CodeIntelUnavailable:
                self._intel = None
        return self._intel

    @property
    def intel_available(self) -> bool:
        return self.intel is not None

    def repo_id_or_none(self) -> str | None:
        """The repo id, or ``None`` when the databases exist but hold no rows.

        An empty index is a legitimate state -- a fresh workspace, or one whose
        index was just dropped -- and the accessors below degrade to empty
        rather than raising, so a caller can still report "nothing is indexed"
        instead of failing.
        """
        cached = self._repo_id
        if cached is not None:
            return cached
        row = self.code.execute("SELECT repo_id FROM files LIMIT 1").fetchone()
        if row is None:
            row = self.code.execute("SELECT repo_id FROM symbols LIMIT 1").fetchone()
        if row is None:
            return None
        resolved = str(row["repo_id"])
        self._repo_id = resolved
        return resolved

    @property
    def repo_id(self) -> str:
        """The single repo id this workspace's databases are keyed by.

        Raises when the index is empty; use :meth:`repo_id_or_none` to treat
        that as data rather than an error.
        """
        resolved = self.repo_id_or_none()
        if resolved is None:
            raise CodeIntelUnavailable(
                f"{workspace_dir(self.repo_root) / CODE_CONTEXT_DB} has no indexed files (empty index)"
            )
        return resolved

    # -- code_context accessors -------------------------------------------

    def files(self, paths: list[str] | None = None) -> list[FileRow]:
        repo_id = self.repo_id_or_none()
        if repo_id is None:
            return []
        sql = (
            "SELECT file_path, language, content_hash, size_bytes, mtime_ns, indexed_at " "FROM files WHERE repo_id = ?"
        )
        params: list[object] = [repo_id]
        if paths is not None:
            if not paths:
                return []
            sql += f" AND file_path IN ({_placeholders(len(paths))})"
            params.extend(paths)
        return [
            FileRow(
                file_path=str(row["file_path"]),
                language=str(row["language"]),
                content_hash=str(row["content_hash"]),
                size_bytes=int(row["size_bytes"]),
                mtime_ns=int(row["mtime_ns"]),
                indexed_at=str(row["indexed_at"]),
            )
            for row in self.code.execute(sql, params)
        ]

    def symbols(
        self,
        file_path: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[SymbolRow]:
        repo_id = self.repo_id_or_none()
        if repo_id is None:
            return []
        sql = (
            "SELECT symbol_id, file_path, language, symbol_name, qualified_name, kind, signature, "
            "start_byte, end_byte, start_line, end_line, parent_symbol, content_hash "
            "FROM symbols WHERE repo_id = ?"
        )
        params: list[object] = [repo_id]
        if file_path is not None:
            sql += " AND file_path = ?"
            params.append(file_path)
        if kind is not None:
            sql += " AND lower(kind) = ?"
            params.append(kind.lower())
        sql += " ORDER BY file_path, start_line"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [
            SymbolRow(
                symbol_id=str(row["symbol_id"]),
                file_path=str(row["file_path"]),
                language=str(row["language"]),
                symbol_name=str(row["symbol_name"]),
                qualified_name=str(row["qualified_name"]),
                kind=str(row["kind"]),
                signature=str(row["signature"]),
                start_byte=int(row["start_byte"]),
                end_byte=int(row["end_byte"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                parent_symbol=_opt_str(row["parent_symbol"]),
                content_hash=str(row["content_hash"]),
            )
            for row in self.code.execute(sql, params)
        ]

    def symbol_counts_by_file(self) -> dict[str, int]:
        """``file_path -> symbol count``. A file with zero symbols is absent here."""
        repo_id = self.repo_id_or_none()
        if repo_id is None:
            return {}
        rows = self.code.execute(
            "SELECT file_path, COUNT(*) AS n FROM symbols WHERE repo_id = ? GROUP BY file_path",
            (repo_id,),
        )
        return {str(row["file_path"]): int(row["n"]) for row in rows}

    def imports(self, resolved_only: bool = False) -> list[ImportRow]:
        repo_id = self.repo_id_or_none()
        if repo_id is None:
            return []
        sql = "SELECT source_file, raw_import, target_file FROM imports WHERE repo_id = ?"
        if resolved_only:
            sql += " AND target_file IS NOT NULL"
        return [
            ImportRow(
                source_file=str(row["source_file"]),
                raw_import=str(row["raw_import"]),
                target_file=_opt_str(row["target_file"]),
            )
            for row in self.code.execute(sql, (repo_id,))
        ]

    # -- intel accessors ---------------------------------------------------

    def call_edges(
        self,
        callee_name: str | None = None,
        caller_file_path: str | None = None,
        limit: int | None = None,
    ) -> list[CallEdgeRow]:
        conn = self.intel
        repo_id = self.repo_id_or_none()
        if conn is None or repo_id is None:
            return []
        sql = (
            "SELECT caller_symbol_name, caller_qualified_name, caller_file_path, caller_start_line, "
            "caller_end_line, callee_name, callee_short_name, call_line, call_column "
            "FROM call_edges WHERE repo_id = ?"
        )
        params: list[object] = [repo_id]
        if callee_name is not None:
            sql += " AND (callee_name = ? OR callee_short_name = ?)"
            params.extend((callee_name, callee_name))
        if caller_file_path is not None:
            sql += " AND caller_file_path = ?"
            params.append(caller_file_path)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [
            CallEdgeRow(
                caller_symbol_name=str(row["caller_symbol_name"]),
                caller_qualified_name=str(row["caller_qualified_name"]),
                caller_file_path=str(row["caller_file_path"]),
                caller_start_line=int(row["caller_start_line"]),
                caller_end_line=int(row["caller_end_line"]),
                callee_name=str(row["callee_name"]),
                callee_short_name=str(row["callee_short_name"]),
                call_line=int(row["call_line"]),
                call_column=int(row["call_column"]),
            )
            for row in conn.execute(sql, params)
        ]

    def references(
        self,
        symbol_name: str | None = None,
        file_path: str | None = None,
        limit: int | None = None,
    ) -> list[ReferenceRow]:
        conn = self.intel
        repo_id = self.repo_id_or_none()
        if conn is None or repo_id is None:
            return []
        sql = (
            "SELECT symbol_name, file_path, line, column, end_column, enclosing_symbol_name, "
            'enclosing_qualified_name, snippet FROM "references" WHERE repo_id = ?'
        )
        params: list[object] = [repo_id]
        if symbol_name is not None:
            sql += " AND symbol_name = ?"
            params.append(symbol_name)
        if file_path is not None:
            sql += " AND file_path = ?"
            params.append(file_path)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [
            ReferenceRow(
                symbol_name=str(row["symbol_name"]),
                file_path=str(row["file_path"]),
                line=int(row["line"]),
                column=int(row["column"]),
                end_column=int(row["end_column"]),
                enclosing_symbol_name=_opt_str(row["enclosing_symbol_name"]),
                enclosing_qualified_name=_opt_str(row["enclosing_qualified_name"]),
                snippet=str(row["snippet"]),
            )
            for row in conn.execute(sql, params)
        ]

    def centrality(self, limit: int | None = None) -> list[CentralityRow]:
        conn = self.intel
        repo_id = self.repo_id_or_none()
        if conn is None or repo_id is None:
            return []
        sql = "SELECT name_key, score, index_version FROM centrality_map WHERE repo_id = ? ORDER BY score DESC"
        params: list[object] = [repo_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [
            CentralityRow(
                name_key=str(row["name_key"]),
                score=float(row["score"]),
                index_version=int(row["index_version"]),
            )
            for row in conn.execute(sql, params)
        ]

    # -- provenance --------------------------------------------------------

    def engine_state(self, key: str, default: int = 0) -> int:
        row = self.code.execute("SELECT value FROM engine_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return int(str(row["value"]))
        except ValueError:
            return default

    def snapshot(self) -> IndexSnapshot:
        """Row counts plus the engine generation they were read at.

        An empty index yields zero counts and an empty ``repo_id`` rather than
        raising -- "nothing is indexed" is an answer, not a failure.
        """
        repo_id = self.repo_id_or_none()
        conn = self.intel
        if repo_id is None:
            return IndexSnapshot(
                repo_id="",
                index_version=self.engine_state("index_version"),
                indexer_semantics_version=self.engine_state("indexer_semantics_version"),
                files=0,
                symbols=0,
                imports=0,
                imports_resolved=0,
                intel_available=conn is not None,
                call_edges=0,
                references=0,
                centrality=0,
            )
        counts = self.code.execute(
            "SELECT (SELECT COUNT(*) FROM files WHERE repo_id = ?1) AS files, "
            "(SELECT COUNT(*) FROM symbols WHERE repo_id = ?1) AS symbols, "
            "(SELECT COUNT(*) FROM imports WHERE repo_id = ?1) AS imports, "
            "(SELECT COUNT(*) FROM imports WHERE repo_id = ?1 AND target_file IS NOT NULL) AS resolved",
            (repo_id,),
        ).fetchone()
        call_edges = references = centrality = 0
        if conn is not None:
            intel_counts = conn.execute(
                "SELECT (SELECT COUNT(*) FROM call_edges WHERE repo_id = ?1) AS call_edges, "
                '(SELECT COUNT(*) FROM "references" WHERE repo_id = ?1) AS refs, '
                "(SELECT COUNT(*) FROM centrality_map WHERE repo_id = ?1) AS centrality",
                (repo_id,),
            ).fetchone()
            call_edges = int(intel_counts["call_edges"])
            references = int(intel_counts["refs"])
            centrality = int(intel_counts["centrality"])
        return IndexSnapshot(
            repo_id=repo_id,
            index_version=self.engine_state("index_version"),
            indexer_semantics_version=self.engine_state("indexer_semantics_version"),
            files=int(counts["files"]),
            symbols=int(counts["symbols"]),
            imports=int(counts["imports"]),
            imports_resolved=int(counts["resolved"]),
            intel_available=conn is not None,
            call_edges=call_edges,
            references=references,
            centrality=centrality,
        )
