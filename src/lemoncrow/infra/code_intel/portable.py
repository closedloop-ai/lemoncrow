"""F7 -- move a built index between machines instead of rebuilding it.

Indexing a large repository costs minutes of CPU that every teammate and every
CI runner pays again from scratch. The databases are already self-contained, so
the useful primitive is small: compact each one, tar them with a manifest, and
refuse to unpack the result anywhere it would produce a subtly wrong graph.

That refusal is the load-bearing part. The closed engine owns ``index_version``
and ``indexer_semantics_version``, and open code cannot migrate its data. An
archive built by a different generation of the extractor is not "slightly out of
date" -- it is a graph whose edges mean something else. Importing it would
produce answers that look right and are not, which is worse than not importing
at all. So a mismatch is an error, never a warning.

The archive is treated as untrusted input on the way back in: every member is
checked against an allow-list of plain filenames, and every database is checked
against the digest recorded at export. A tar file is a file format an attacker
can write, and "it came from a teammate" is not provenance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.infra.code_intel.sidecar import SIDECAR_DB
from lemoncrow.infra.code_intel.store import (
    CODE_CONTEXT_DB,
    FTS_DB,
    INTEL_DB,
    REPO_MAP_TAGS_DB,
    VECTORS_DB,
    workspace_dir,
)

__all__ = [
    "ARCHIVE_FORMAT_VERSION",
    "EXPORTABLE_DBS",
    "MANIFEST_NAME",
    "TIERS",
    "ExportResult",
    "ImportResult",
    "PortableIndexError",
    "available_codec",
    "export_index",
    "import_index",
    "read_manifest",
]

ARCHIVE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"

#: Everything a workspace needs to answer queries without re-indexing. The five
#: engine databases plus the sidecar we own. Absent files are skipped, not
#: faked: ``vectors.sqlite`` only exists once embeddings have been built.
EXPORTABLE_DBS: tuple[str, ...] = (
    CODE_CONTEXT_DB,
    INTEL_DB,
    FTS_DB,
    VECTORS_DB,
    REPO_MAP_TAGS_DB,
    SIDECAR_DB,
)

#: ``best`` for a shared artifact, ``fast`` for an incremental refresh.
TIERS: dict[str, int] = {"best": 9, "fast": 3}

_MEMBER_NAMES = frozenset({MANIFEST_NAME, *EXPORTABLE_DBS})
_HASH_CHUNK = 1 << 20
_GIT_TIMEOUT_S = 10.0


class PortableIndexError(RuntimeError):
    """The archive is unusable, or unsafe to unpack here."""


@dataclass(frozen=True)
class ExportResult:
    path: Path
    codec: str
    tier: str
    size_bytes: int
    databases: tuple[str, ...]
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "codec": self.codec,
            "tier": self.tier,
            "size_bytes": self.size_bytes,
            "databases": list(self.databases),
            "manifest": self.manifest,
        }


@dataclass(frozen=True)
class ImportResult:
    archive: Path
    workspace: Path
    restored: tuple[str, ...]
    manifest: dict[str, Any]
    verified_against: str
    #: Databases cleared from the workspace that the archive did not replace.
    #: Reported rather than silently dropped: their absence changes what the
    #: workspace can answer until the engine's next pass rebuilds them.
    removed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive": str(self.archive),
            "workspace": str(self.workspace),
            "restored": list(self.restored),
            "removed": list(self.removed),
            "verified_against": self.verified_against,
            "manifest": self.manifest,
        }


# --------------------------------------------------------------------------- #
# codec
# --------------------------------------------------------------------------- #


def available_codec() -> str:
    """``"zstd"`` when :mod:`zstandard` is installed, else ``"xz"``.

    zstd is the better codec and lives behind the ``portable`` extra. Falling
    back to stdlib lzma keeps export/import working on a base install rather
    than making the feature depend on an optional wheel -- the archive names its
    codec, and import reads that rather than assuming.
    """
    try:
        import zstandard  # noqa: F401
    except ImportError:
        return "xz"
    return "zstd"


def _suffix(codec: str) -> str:
    return ".tar.zst" if codec == "zstd" else ".tar.xz"


def _codec_for_archive(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zst") or name.endswith(".zstd"):
        return "zstd"
    if name.endswith(".xz") or name.endswith(".lzma"):
        return "xz"
    raise PortableIndexError(f"cannot tell the codec from {path.name!r}; expected .tar.zst or .tar.xz")


def _compress(tar_path: Path, out_path: Path, codec: str, level: int) -> None:
    if codec == "zstd":
        import zstandard

        compressor = zstandard.ZstdCompressor(level=level)
        with tar_path.open("rb") as source, out_path.open("wb") as target:
            compressor.copy_stream(source, target)
        return
    import lzma

    with tar_path.open("rb") as source, lzma.open(out_path, "wb", preset=min(level, 9)) as target:
        shutil.copyfileobj(source, target)


def _decompress(archive: Path, tar_path: Path, codec: str) -> None:
    try:
        if codec == "zstd":
            try:
                import zstandard
            except ImportError as exc:
                raise PortableIndexError(
                    "this archive is zstd-compressed; install the 'portable' extra (zstandard) to read it"
                ) from exc
            decompressor = zstandard.ZstdDecompressor()
            with archive.open("rb") as source, tar_path.open("wb") as target:
                decompressor.copy_stream(source, target)
            return
        import lzma

        with lzma.open(archive, "rb") as source, tar_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    except PortableIndexError:
        raise
    except Exception as exc:
        raise PortableIndexError(f"{archive.name} is not a readable {codec} stream: {exc}") from exc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_into(source: Path, target: Path) -> None:
    """Copy *source* to *target* as a single compacted file, WAL folded in.

    ``VACUUM INTO`` is read-only with respect to the source and produces a
    defragmented copy. Some builds refuse it against a read-only handle, so the
    online backup API is the fallback: it costs a little archive size and is
    otherwise equivalent.
    """
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        try:
            conn.execute("VACUUM INTO ?", (str(target),))
            return
        except sqlite3.Error:
            target.unlink(missing_ok=True)
        backup = sqlite3.connect(target)
        try:
            conn.backup(backup)
        finally:
            backup.close()
    except sqlite3.Error as exc:
        raise PortableIndexError(f"cannot read {source.name}: {exc}") from exc
    finally:
        conn.close()


def _engine_versions(workspace: Path) -> tuple[int, int]:
    db = workspace / CODE_CONTEXT_DB
    if not db.exists():
        return (0, 0)
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:  # pragma: no cover - permission dependent
        return (0, 0)
    try:
        rows = dict(conn.execute("SELECT key, value FROM engine_state").fetchall())
    except sqlite3.Error:
        return (0, 0)
    finally:
        conn.close()

    def _as_int(key: str) -> int:
        try:
            return int(str(rows.get(key, 0)))
        except ValueError:
            return 0

    return (_as_int("index_version"), _as_int("indexer_semantics_version"))


def _row_counts(workspace: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for db, tables in ((CODE_CONTEXT_DB, ("files", "symbols", "imports")), (INTEL_DB, ("call_edges", "references"))):
        path = workspace / db
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:  # pragma: no cover - permission dependent
            continue
        try:
            for table in tables:
                quoted = f'"{table}"' if table == "references" else table
                try:
                    counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
                except sqlite3.Error:
                    continue
        finally:
            conn.close()
    return counts


def _sidecar_schema_version(workspace: Path) -> int | None:
    path = workspace / SIDECAR_DB
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:  # pragma: no cover - permission dependent
        return None
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:  # pragma: no cover - corrupt sidecar
        return None
    finally:
        conn.close()


def _repo_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else None


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def export_index(
    repo_root: Path | str = ".",
    out: Path | str | None = None,
    tier: str = "best",
) -> ExportResult:
    """Compact this workspace's index into a single portable archive.

    The two tiers differ only in compression level. The plan also called for
    ``best`` to drop derived indexes; it does not, because the engine's DDL is
    closed and an index we drop is one we cannot recreate -- the import would
    hand back a database the engine expects to be complete.
    """
    if tier not in TIERS:
        raise PortableIndexError(f"unknown tier {tier!r}; expected one of {', '.join(sorted(TIERS))}")
    root = Path(repo_root).resolve()
    workspace = workspace_dir(root)
    if not (workspace / CODE_CONTEXT_DB).exists():
        raise PortableIndexError(f"nothing to export: {workspace / CODE_CONTEXT_DB} does not exist")

    codec = available_codec()
    destination = Path(out).resolve() if out is not None else workspace.parent / f"index{_suffix(codec)}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    from lemoncrow import __version__ as lemoncrow_version

    index_version, semantics_version = _engine_versions(workspace)
    with tempfile.TemporaryDirectory(prefix="lemoncrow-index-export-") as tmp:
        staging = Path(tmp)
        present: list[str] = []
        databases: dict[str, dict[str, Any]] = {}
        for name in EXPORTABLE_DBS:
            source = workspace / name
            if not source.exists():
                continue
            compacted = staging / name
            _compact_into(source, compacted)
            present.append(name)
            databases[name] = {"bytes": compacted.stat().st_size, "sha256": _sha256(compacted)}

        manifest: dict[str, Any] = {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "lemoncrow_version": lemoncrow_version,
            "engine_index_version": index_version,
            "indexer_semantics_version": semantics_version,
            "sidecar_schema_version": _sidecar_schema_version(workspace),
            "repo_head": _repo_head(root),
            "codec": codec,
            "tier": tier,
            "databases": databases,
            "row_counts": _row_counts(workspace),
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        tar_path = staging / "bundle.tar"
        with tarfile.open(tar_path, "w") as tar:
            for name in (MANIFEST_NAME, *present):
                tar.add(staging / name, arcname=name)
        _compress(tar_path, destination, codec, TIERS[tier])

    return ExportResult(
        path=destination,
        codec=codec,
        tier=tier,
        size_bytes=destination.stat().st_size,
        databases=tuple(present),
        manifest=manifest,
    )


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def _safe_extract(tar: tarfile.TarFile, into: Path) -> list[str]:
    """Extract only plain files whose names are on the allow-list.

    A tar archive can name ``../../etc/whatever``, or be a symlink pointing
    anywhere on the filesystem. This index format has a fixed, flat membership,
    so anything else is refused outright rather than sanitised -- there is no
    legitimate archive that needs the general case.
    """
    extracted: list[str] = []
    for member in tar.getmembers():
        if not member.isfile():
            raise PortableIndexError(f"archive member {member.name!r} is not a regular file")
        if member.name not in _MEMBER_NAMES:
            raise PortableIndexError(f"unexpected archive member {member.name!r}")
        source = tar.extractfile(member)
        if source is None:  # pragma: no cover - unreachable for isfile() members
            raise PortableIndexError(f"archive member {member.name!r} could not be read")
        with source, (into / member.name).open("wb") as target:
            shutil.copyfileobj(source, target)
        extracted.append(member.name)
    return extracted


def _open_archive(archive: Path, staging: Path) -> list[str]:
    codec = _codec_for_archive(archive)
    tar_path = staging / "bundle.tar"
    _decompress(archive, tar_path, codec)
    try:
        with tarfile.open(tar_path, "r") as tar:
            return _safe_extract(tar, staging)
    except PortableIndexError:
        raise
    except tarfile.TarError as exc:
        raise PortableIndexError(f"{archive.name} is not a readable tar archive: {exc}") from exc


def _load_manifest(staging: Path) -> dict[str, Any]:
    path = staging / MANIFEST_NAME
    if not path.exists():
        raise PortableIndexError(f"archive has no {MANIFEST_NAME}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PortableIndexError(f"{MANIFEST_NAME} is not readable JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PortableIndexError(f"{MANIFEST_NAME} is not an object")
    if manifest.get("format_version") != ARCHIVE_FORMAT_VERSION:
        raise PortableIndexError(
            f"archive format version {manifest.get('format_version')!r} "
            f"is not {ARCHIVE_FORMAT_VERSION}; this LemonCrow cannot read it"
        )
    return manifest


def read_manifest(archive: Path | str) -> dict[str, Any]:
    """The archive's manifest, without unpacking anything into a workspace."""
    path = Path(archive).resolve()
    if not path.exists():
        raise PortableIndexError(f"{path} does not exist")
    with tempfile.TemporaryDirectory(prefix="lemoncrow-index-manifest-") as tmp:
        staging = Path(tmp)
        _open_archive(path, staging)
        return _load_manifest(staging)


def import_index(
    archive: Path | str,
    repo_root: Path | str = ".",
    force: bool = False,
) -> ImportResult:
    """Unpack *archive* into this workspace, or refuse and say why.

    Refuses when the target workspace already holds an index whose
    ``indexer_semantics_version`` differs from the archive's. The engine owns
    that number and open code cannot migrate its data, so a mismatched import
    does not yield a stale graph -- it yields one whose edges mean something
    else, answering confidently and wrongly. *force* overrides the
    already-populated check, never the semantics check.
    """
    path = Path(archive).resolve()
    if not path.exists():
        raise PortableIndexError(f"{path} does not exist")
    root = Path(repo_root).resolve()
    workspace = workspace_dir(root)

    with tempfile.TemporaryDirectory(prefix="lemoncrow-index-import-") as tmp:
        staging = Path(tmp)
        members = _open_archive(path, staging)
        manifest = _load_manifest(staging)

        databases = manifest.get("databases")
        if not isinstance(databases, dict) or not databases:
            raise PortableIndexError(f"{MANIFEST_NAME} lists no databases")

        for name, meta in databases.items():
            if name not in members:
                raise PortableIndexError(f"{MANIFEST_NAME} lists {name!r} but the archive does not contain it")
            expected = str(meta.get("sha256", ""))
            if not expected:
                # The manifest travels inside the archive it vouches for, so a
                # missing digest is not a gap in the check -- it IS the attack,
                # and bit-rot looks identical. Export always writes one, so an
                # absent digest is never a legitimate archive from this codebase.
                raise PortableIndexError(
                    f"{name} has no recorded digest in {MANIFEST_NAME}; refusing to import it unverified"
                )
            actual = _sha256(staging / name)
            if actual != expected:
                raise PortableIndexError(
                    f"{name} failed its digest check (expected {expected[:12]}, got {actual[:12]}); "
                    "the archive is corrupt or was modified"
                )

        local_index_version, local_semantics = _engine_versions(workspace)
        archive_semantics = int(manifest.get("indexer_semantics_version") or 0)
        populated = (workspace / CODE_CONTEXT_DB).exists()

        if populated and local_semantics and archive_semantics and local_semantics != archive_semantics:
            raise PortableIndexError(
                f"indexer_semantics_version mismatch: archive {archive_semantics}, "
                f"workspace {local_semantics}. The engine owns this number and open code cannot "
                "migrate its data; re-index instead of importing."
            )
        if populated and not force:
            raise PortableIndexError(
                f"{workspace} already holds an index (index_version {local_index_version}); " "pass force to replace it"
            )
        verified_against = f"workspace indexer_semantics_version {local_semantics}" if populated else "empty workspace"

        workspace.mkdir(parents=True, exist_ok=True)
        restored: list[str] = []
        removed: list[str] = []
        for name in EXPORTABLE_DBS:
            target = workspace / name
            # Clear every database first, including ones the archive does not
            # carry. A source that never built embeddings or never ran the
            # call-graph pass exports without vectors.sqlite / intel.sqlite, and
            # leaving the local copies behind pairs a fresh code_context.sqlite
            # with sidecars keyed to a superseded generation's symbol ids --
            # edges that mean something else, which is the exact failure the
            # semantics check exists to prevent. It cannot catch this one: it
            # reads only code_context.sqlite, and that file *is* replaced.
            #
            # WAL/SHM siblings describe the file being replaced, not the new one.
            for sibling in (f"{name}-wal", f"{name}-shm"):
                (workspace / sibling).unlink(missing_ok=True)
            if name not in databases:
                if target.exists():
                    target.unlink()
                    removed.append(name)
                continue
            target.unlink(missing_ok=True)
            shutil.move(str(staging / name), str(target))
            restored.append(name)

    return ImportResult(
        archive=path,
        workspace=workspace,
        restored=tuple(restored),
        removed=tuple(removed),
        manifest=manifest,
        verified_against=verified_against,
    )
