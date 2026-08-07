"""``sidecar.sqlite`` -- the code-intel database open source owns and may write.

The engine's databases are read-only to us (see
:mod:`lemoncrow.infra.code_intel.store`). Anything derived that needs to persist
-- resolved call edges, clone pairs, open embeddings -- lands here instead, in
the same workspace directory, under a schema with no dependency on the engine's
DDL.

Every derived table registers a :func:`stamp` recording which engine generation
it was built from, so stale data is *detectable* rather than silently wrong: a
``--reindex`` bumps ``engine_state.index_version`` and every sidecar table built
against the old generation immediately reports as stale.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from lemoncrow.infra.code_intel.store import workspace_dir

__all__ = [
    "SCHEMA_VERSION",
    "SIDECAR_DB",
    "TableStamp",
    "is_stale",
    "open_sidecar",
    "schema_version",
    "sidecar_path",
    "stamp",
    "stamp_of",
]

SIDECAR_DB = "sidecar.sqlite"

#: Bump when adding a migration below. Never renumber an existing one.
SCHEMA_VERSION = 1

_BUSY_TIMEOUT_MS = 5_000

# (version, statements). Applied in order, skipping versions already present in
# ``PRAGMA user_version``. Migrations are append-only and must be idempotent in
# the sense that each runs exactly once per database.
_MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS sidecar_meta (
                table_name           TEXT    NOT NULL PRIMARY KEY,
                schema_version       INTEGER NOT NULL,
                engine_index_version INTEGER NOT NULL,
                built_at             TEXT    NOT NULL
            )
            """,
        ),
    ),
)


class TableStamp:
    """Provenance for one derived sidecar table."""

    def __init__(
        self,
        table_name: str,
        schema_version: int,
        engine_index_version: int,
        built_at: str,
    ) -> None:
        self.table_name: str = table_name
        self.schema_version: int = schema_version
        self.engine_index_version: int = engine_index_version
        self.built_at: str = built_at

    def __repr__(self) -> str:
        return (
            f"TableStamp(table_name={self.table_name!r}, schema_version={self.schema_version!r}, "
            f"engine_index_version={self.engine_index_version!r}, built_at={self.built_at!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableStamp):
            return NotImplemented
        return (
            self.table_name == other.table_name
            and self.schema_version == other.schema_version
            and self.engine_index_version == other.engine_index_version
            and self.built_at == other.built_at
        )


def sidecar_path(repo_root: Path | str = ".") -> Path:
    """Return ``<repo_root>/.lemoncrow/workspace/sidecar.sqlite``."""
    return workspace_dir(repo_root) / SIDECAR_DB


def open_sidecar(repo_root: Path | str = ".") -> sqlite3.Connection:
    """Open (creating if needed) the sidecar database with migrations applied.

    Unlike the engine's databases this one is read-write -- we own it.
    """
    path = sidecar_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    for version, statements in _MIGRATIONS:
        if version <= current:
            continue
        for statement in statements:
            conn.execute(statement)
        # PRAGMA cannot be parameterised; the value is our own int constant.
        conn.execute(f"PRAGMA user_version = {int(version)}")
    conn.commit()


def schema_version(conn: sqlite3.Connection) -> int:
    """The migration level this database is currently at."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def stamp(conn: sqlite3.Connection, table_name: str, engine_index_version: int) -> TableStamp:
    """Record that *table_name* was just built from *engine_index_version*."""
    built_at = datetime.now(UTC).isoformat()
    record = TableStamp(
        table_name=table_name,
        schema_version=schema_version(conn),
        engine_index_version=int(engine_index_version),
        built_at=built_at,
    )
    conn.execute(
        "INSERT INTO sidecar_meta (table_name, schema_version, engine_index_version, built_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(table_name) DO UPDATE SET "
        "schema_version = excluded.schema_version, "
        "engine_index_version = excluded.engine_index_version, "
        "built_at = excluded.built_at",
        (record.table_name, record.schema_version, record.engine_index_version, record.built_at),
    )
    conn.commit()
    return record


def stamp_of(conn: sqlite3.Connection, table_name: str) -> TableStamp | None:
    """Return the recorded provenance for *table_name*, or ``None`` if never built."""
    row = conn.execute(
        "SELECT table_name, schema_version, engine_index_version, built_at " "FROM sidecar_meta WHERE table_name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        return None
    return TableStamp(
        table_name=str(row["table_name"]),
        schema_version=int(row["schema_version"]),
        engine_index_version=int(row["engine_index_version"]),
        built_at=str(row["built_at"]),
    )


def is_stale(conn: sqlite3.Connection, table_name: str, engine_index_version: int) -> bool:
    """True when *table_name* was never built, or built from an older engine index.

    A never-built table is stale rather than fresh: the caller must rebuild it
    either way, and treating "absent" as "current" is the failure mode this
    whole mechanism exists to prevent.
    """
    recorded = stamp_of(conn, table_name)
    if recorded is None:
        return True
    return recorded.engine_index_version != int(engine_index_version) or recorded.schema_version != schema_version(conn)
