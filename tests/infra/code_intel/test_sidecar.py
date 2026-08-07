"""F0: the sidecar database open source owns and may write."""

from __future__ import annotations

from pathlib import Path

from lemoncrow.infra.code_intel.sidecar import (
    SCHEMA_VERSION,
    SIDECAR_DB,
    is_stale,
    open_sidecar,
    schema_version,
    sidecar_path,
    stamp,
    stamp_of,
)
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, workspace_dir


def test_sidecar_lives_beside_the_engine_databases(tmp_path: Path) -> None:
    assert sidecar_path(tmp_path) == workspace_dir(tmp_path) / SIDECAR_DB
    assert sidecar_path(tmp_path).parent == (workspace_dir(tmp_path) / CODE_CONTEXT_DB).parent


def test_open_sidecar_creates_and_migrates(tmp_path: Path) -> None:
    conn = open_sidecar(tmp_path)
    try:
        assert sidecar_path(tmp_path).exists()
        assert schema_version(conn) == SCHEMA_VERSION
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "sidecar_meta" in names
    finally:
        conn.close()


def test_reopening_does_not_reapply_migrations(tmp_path: Path) -> None:
    first = open_sidecar(tmp_path)
    try:
        stamp(first, "resolved_call_edges", engine_index_version=8)
    finally:
        first.close()

    second = open_sidecar(tmp_path)
    try:
        assert schema_version(second) == SCHEMA_VERSION
        recorded = stamp_of(second, "resolved_call_edges")
        assert recorded is not None
        assert recorded.engine_index_version == 8
    finally:
        second.close()


def test_stamp_round_trip_and_overwrite(tmp_path: Path) -> None:
    conn = open_sidecar(tmp_path)
    try:
        written = stamp(conn, "symbol_clones", engine_index_version=8)
        assert stamp_of(conn, "symbol_clones") == written
        assert written.schema_version == SCHEMA_VERSION
        assert written.built_at.endswith("+00:00")

        rebuilt = stamp(conn, "symbol_clones", engine_index_version=9)
        assert rebuilt.engine_index_version == 9
        rows = conn.execute("SELECT COUNT(*) FROM sidecar_meta WHERE table_name = 'symbol_clones'").fetchone()
        assert rows[0] == 1  # upsert, not a second row
    finally:
        conn.close()


def test_never_built_counts_as_stale(tmp_path: Path) -> None:
    """Absent must never read as current -- that is the failure this prevents."""
    conn = open_sidecar(tmp_path)
    try:
        assert is_stale(conn, "symbol_clones", engine_index_version=8) is True
    finally:
        conn.close()


def test_staleness_tracks_the_engine_index_version(tmp_path: Path) -> None:
    conn = open_sidecar(tmp_path)
    try:
        stamp(conn, "symbol_clones", engine_index_version=8)
        assert is_stale(conn, "symbol_clones", engine_index_version=8) is False
        # A --reindex bumps the engine generation; everything derived is now stale.
        assert is_stale(conn, "symbol_clones", engine_index_version=9) is True
    finally:
        conn.close()


def test_unknown_table_stamp_is_none(tmp_path: Path) -> None:
    conn = open_sidecar(tmp_path)
    try:
        assert stamp_of(conn, "never_built") is None
    finally:
        conn.close()
