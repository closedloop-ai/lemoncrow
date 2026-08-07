"""Portable index archives: round-trip, refusal, and hostile input (F7).

The interesting assertions are the refusals. An archive is a file format someone
else can write, and an index imported across an extractor-semantics boundary
does not produce a stale graph -- it produces one whose edges mean something
else, which answers confidently and wrongly. So a mismatch, a bad digest, and a
member named ``../../etc/passwd`` all have to be errors, not warnings.
"""

from __future__ import annotations

import io
import json
import lzma
import tarfile
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from lemoncrow.infra.code_intel.portable import (
    ARCHIVE_FORMAT_VERSION,
    MANIFEST_NAME,
    PortableIndexError,
    available_codec,
    export_index,
    import_index,
    read_manifest,
)
from lemoncrow.infra.code_intel.store import CODE_CONTEXT_DB, CodeIntelStore, workspace_dir

WorkspaceFactory = Callable[..., Path]

_FILES = [{"file_path": "src/a.py"}, {"file_path": "src/b.py"}]
_SYMBOLS = [
    {"file_path": "src/a.py", "symbol_name": "alpha", "kind": "function"},
    {"file_path": "src/a.py", "symbol_name": "beta", "kind": "function"},
    {"file_path": "src/b.py", "symbol_name": "Gamma", "kind": "class"},
]
_IMPORTS = [{"source_file": "src/b.py", "raw_import": "a", "target_file": "src/a.py"}]
_EDGES = [{"caller_symbol_name": "Gamma", "caller_file_path": "src/b.py", "callee_name": "alpha"}]
_REFERENCES = [{"symbol_name": "alpha", "file_path": "src/b.py", "line": 4}]


@pytest.fixture
def source_repo(make_workspace: WorkspaceFactory) -> Path:
    return make_workspace(
        files=_FILES,
        symbols=_SYMBOLS,
        imports=_IMPORTS,
        call_edges=_EDGES,
        references=_REFERENCES,
        index_version=31,
        indexer_semantics_version=2,
        name="source",
    )


def _snapshot(root: Path) -> dict[str, object]:
    with CodeIntelStore(root) as store:
        snapshot = store.snapshot()
        return {
            "files": snapshot.files,
            "symbols": snapshot.symbols,
            "imports": snapshot.imports,
            "call_edges": snapshot.call_edges,
            "references": snapshot.references,
            "index_version": snapshot.index_version,
            "names": sorted(row.symbol_name for row in store.symbols()),
        }


def _archive_with(tmp_path: Path, members: Mapping[str, bytes], name: str = "crafted.tar.xz") -> Path:
    """Hand-build an archive so the extractor's refusals can be exercised."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for arcname, payload in members.items():
            info = tarfile.TarInfo(arcname)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    path = tmp_path / name
    with lzma.open(path, "wb") as handle:
        handle.write(raw.getvalue())
    return path


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #


def test_round_trip_preserves_row_counts_and_query_results(
    source_repo: Path,
    tmp_path: Path,
) -> None:
    before = _snapshot(source_repo)
    result = export_index(repo_root=source_repo)
    assert result.path.exists()
    assert result.size_bytes > 0
    assert CODE_CONTEXT_DB in result.databases

    clone = tmp_path / "clone"
    imported = import_index(archive=result.path, repo_root=clone)

    assert CODE_CONTEXT_DB in imported.restored
    assert imported.verified_against == "empty workspace"
    assert _snapshot(clone) == before


def test_round_trip_survives_a_sampled_query(source_repo: Path, tmp_path: Path) -> None:
    archive = export_index(repo_root=source_repo).path
    clone = tmp_path / "clone"
    import_index(archive=archive, repo_root=clone)

    with CodeIntelStore(clone) as store:
        assert [row.symbol_name for row in store.symbols(kind="class")] == ["Gamma"]
        assert [edge.caller_symbol_name for edge in store.call_edges(callee_name="alpha")] == ["Gamma"]
        assert [ref.file_path for ref in store.references(symbol_name="alpha")] == ["src/b.py"]


def test_export_writes_beside_the_workspace_by_default(source_repo: Path) -> None:
    result = export_index(repo_root=source_repo)
    assert result.path.parent == workspace_dir(source_repo).parent
    assert result.path.name.startswith("index.tar.")


def test_export_honours_an_explicit_out_path(source_repo: Path, tmp_path: Path) -> None:
    destination = tmp_path / "nested" / f"custom.tar.{'zst' if available_codec() == 'zstd' else 'xz'}"
    result = export_index(repo_root=source_repo, out=destination)
    assert result.path == destination.resolve()
    assert destination.exists()


def test_both_tiers_produce_importable_archives(source_repo: Path, tmp_path: Path) -> None:
    suffix = "zst" if available_codec() == "zstd" else "xz"
    for index, tier in enumerate(("best", "fast")):
        archive = export_index(repo_root=source_repo, out=tmp_path / f"t{index}.tar.{suffix}", tier=tier)
        assert archive.tier == tier
        clone = tmp_path / f"clone{index}"
        import_index(archive=archive.path, repo_root=clone)
        assert _snapshot(clone)["symbols"] == 3


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def test_manifest_records_the_provenance_an_importer_needs(source_repo: Path) -> None:
    manifest = export_index(repo_root=source_repo).manifest
    assert manifest["format_version"] == ARCHIVE_FORMAT_VERSION
    assert manifest["engine_index_version"] == 31
    assert manifest["indexer_semantics_version"] == 2
    assert manifest["row_counts"]["symbols"] == 3
    assert manifest["row_counts"]["references"] == 1
    assert manifest["codec"] == available_codec()
    assert set(manifest["databases"]) <= set(export_index(repo_root=source_repo).databases)
    assert manifest["lemoncrow_version"]


def test_read_manifest_does_not_touch_a_workspace(source_repo: Path, tmp_path: Path) -> None:
    archive = export_index(repo_root=source_repo).path
    clone = tmp_path / "untouched"
    manifest = read_manifest(archive)
    assert manifest["engine_index_version"] == 31
    assert not clone.exists()


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #


def test_export_refuses_an_unindexed_workspace(tmp_path: Path) -> None:
    with pytest.raises(PortableIndexError, match="nothing to export"):
        export_index(repo_root=tmp_path / "never-indexed")


def test_export_refuses_an_unknown_tier(source_repo: Path) -> None:
    with pytest.raises(PortableIndexError, match="unknown tier"):
        export_index(repo_root=source_repo, tier="maximum")


def test_import_refuses_a_semantics_mismatch(
    source_repo: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    archive = export_index(repo_root=source_repo).path
    target = make_workspace(
        files=_FILES,
        symbols=_SYMBOLS,
        indexer_semantics_version=99,
        name="target",
    )
    with pytest.raises(PortableIndexError, match="indexer_semantics_version mismatch"):
        import_index(archive=archive, repo_root=target)


def test_force_does_not_override_a_semantics_mismatch(
    source_repo: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    """force is for "replace what is here", never for "ignore what it means"."""
    archive = export_index(repo_root=source_repo).path
    target = make_workspace(files=_FILES, indexer_semantics_version=99, name="target")
    with pytest.raises(PortableIndexError, match="indexer_semantics_version mismatch"):
        import_index(archive=archive, repo_root=target, force=True)


def test_import_refuses_a_populated_workspace_without_force(
    source_repo: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    archive = export_index(repo_root=source_repo).path
    target = make_workspace(files=_FILES, symbols=_SYMBOLS, indexer_semantics_version=2, name="target")
    with pytest.raises(PortableIndexError, match="already holds an index"):
        import_index(archive=archive, repo_root=target)


def test_force_replaces_a_matching_workspace(
    source_repo: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    archive = export_index(repo_root=source_repo).path
    target = make_workspace(
        files=[{"file_path": "only.py"}],
        symbols=[{"file_path": "only.py", "symbol_name": "solo"}],
        index_version=4,
        indexer_semantics_version=2,
        name="target",
    )
    result = import_index(archive=archive, repo_root=target, force=True)
    assert "indexer_semantics_version 2" in result.verified_against
    assert _snapshot(target)["names"] == ["Gamma", "alpha", "beta"]


def test_import_refuses_a_missing_archive(tmp_path: Path) -> None:
    with pytest.raises(PortableIndexError, match="does not exist"):
        import_index(archive=tmp_path / "nope.tar.xz", repo_root=tmp_path)


def test_import_refuses_an_unrecognised_extension(source_repo: Path, tmp_path: Path) -> None:
    archive = export_index(repo_root=source_repo).path
    renamed = tmp_path / "index.tar.gz"
    renamed.write_bytes(archive.read_bytes())
    with pytest.raises(PortableIndexError, match="cannot tell the codec"):
        import_index(archive=renamed, repo_root=tmp_path / "clone")


# --------------------------------------------------------------------------- #
# corrupt and hostile input
# --------------------------------------------------------------------------- #


def test_truncated_archive_is_an_error(source_repo: Path, tmp_path: Path) -> None:
    archive = export_index(repo_root=source_repo).path
    broken = tmp_path / "broken.tar.xz"
    broken.write_bytes(archive.read_bytes()[: max(1, archive.stat().st_size // 2)])
    with pytest.raises(PortableIndexError):
        import_index(archive=broken, repo_root=tmp_path / "clone")


def test_garbage_bytes_are_an_error(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.tar.xz"
    garbage.write_bytes(b"this is not a compressed tar archive" * 10)
    with pytest.raises(PortableIndexError, match="not a readable"):
        import_index(archive=garbage, repo_root=tmp_path / "clone")


def test_missing_manifest_is_an_error(tmp_path: Path) -> None:
    archive = _archive_with(tmp_path, {CODE_CONTEXT_DB: b"\x00" * 16})
    with pytest.raises(PortableIndexError, match=f"no {MANIFEST_NAME}"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


def test_unreadable_manifest_is_an_error(tmp_path: Path) -> None:
    archive = _archive_with(tmp_path, {MANIFEST_NAME: b"{not json"})
    with pytest.raises(PortableIndexError, match="not readable JSON"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


def test_future_format_version_is_refused(tmp_path: Path) -> None:
    manifest = json.dumps({"format_version": ARCHIVE_FORMAT_VERSION + 1}).encode()
    archive = _archive_with(tmp_path, {MANIFEST_NAME: manifest})
    with pytest.raises(PortableIndexError, match="format version"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


def test_manifest_without_databases_is_refused(tmp_path: Path) -> None:
    manifest = json.dumps({"format_version": ARCHIVE_FORMAT_VERSION, "databases": {}}).encode()
    archive = _archive_with(tmp_path, {MANIFEST_NAME: manifest})
    with pytest.raises(PortableIndexError, match="lists no databases"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


def test_manifest_listing_an_absent_database_is_refused(tmp_path: Path) -> None:
    manifest = json.dumps(
        {"format_version": ARCHIVE_FORMAT_VERSION, "databases": {CODE_CONTEXT_DB: {"sha256": ""}}}
    ).encode()
    archive = _archive_with(tmp_path, {MANIFEST_NAME: manifest})
    with pytest.raises(PortableIndexError, match="does not contain it"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


def test_tampered_database_fails_its_digest(tmp_path: Path) -> None:
    payload = b"\x00" * 32
    manifest = json.dumps(
        {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "indexer_semantics_version": 2,
            "databases": {CODE_CONTEXT_DB: {"sha256": "0" * 64}},
        }
    ).encode()
    archive = _archive_with(tmp_path, {MANIFEST_NAME: manifest, CODE_CONTEXT_DB: payload})
    with pytest.raises(PortableIndexError, match="failed its digest check"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


@pytest.mark.parametrize(
    "hostile_name",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "nested/code_context.sqlite",
        "code_context.sqlite.bak",
    ],
)
def test_members_outside_the_allow_list_are_refused(tmp_path: Path, hostile_name: str) -> None:
    """The format has a fixed, flat membership; anything else is refused outright."""
    archive = _archive_with(tmp_path, {hostile_name: b"payload"}, name="hostile.tar.xz")
    with pytest.raises(PortableIndexError, match="unexpected archive member"):
        import_index(archive=archive, repo_root=tmp_path / "clone")
    assert not (tmp_path / "clone").exists()


def test_symlink_members_are_refused(tmp_path: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    archive = tmp_path / "symlink.tar.xz"
    with lzma.open(archive, "wb") as handle:
        handle.write(raw.getvalue())

    with pytest.raises(PortableIndexError, match="not a regular file"):
        import_index(archive=archive, repo_root=tmp_path / "clone")


def test_a_refused_import_leaves_the_target_untouched(
    source_repo: Path,
    make_workspace: WorkspaceFactory,
) -> None:
    archive = export_index(repo_root=source_repo).path
    target = make_workspace(
        files=[{"file_path": "only.py"}],
        symbols=[{"file_path": "only.py", "symbol_name": "solo"}],
        indexer_semantics_version=99,
        name="target",
    )
    with pytest.raises(PortableIndexError):
        import_index(archive=archive, repo_root=target, force=True)
    assert _snapshot(target)["names"] == ["solo"]
