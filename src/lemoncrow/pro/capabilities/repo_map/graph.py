"""Reference graph construction for repo maps."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation._graph import DiGraph

from lemoncrow.infra.code_intel.inclusion import (
    load_lemoncrow_ignore_spec,
    scan_selects,
    should_skip_path,
    should_skip_relative_path,
    source_file_patterns,
)
from lemoncrow.infra.tree_sitter.tags import Tag, detect_language, extract_tags
from lemoncrow.pro.capabilities.repo_map.tag_cache import TagCache

# In-process cache: building the reference graph parses every source file with
# tree-sitter (~14-37 s for a mid-size repo). The result is pure-functional given
# the repo root + file list, so a single dict cache makes repeated calls free.
_REFERENCE_GRAPH_CACHE: dict[
    tuple[str, tuple[str, ...] | None],
    tuple[DiGraph, dict[str, list[Tag]]],
] = {}


def iter_source_files(
    repo_root: Path,
    include_globs: list[str] | None = None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    patterns = include_globs or source_file_patterns()
    ignore_spec = load_lemoncrow_ignore_spec(repo_root)
    files = _iter_git_visible_source_files(repo_root, patterns, ignore_spec, progress_callback=progress_callback)
    if files:
        return files
    files = _iter_glob_source_files(repo_root, patterns, ignore_spec, progress_callback=progress_callback)
    return files


def _iter_git_visible_source_files(
    repo_root: Path,
    patterns: list[str],
    ignore_spec: Any | None = None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    # ``--recurse-submodules`` lists tracked files inside submodules too, but git
    # rejects it combined with ``--others``, so untracked files are fetched in a
    # separate top-level-only call.
    entries: list[bytes] = []
    for extra_args in (
        ("--cached", "--recurse-submodules"),
        ("--others", "--exclude-standard"),
    ):
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), "ls-files", "-z", *extra_args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=False,
            )
        except OSError:
            return []
        if completed.returncode != 0:
            return []
        entries.extend(entry for entry in completed.stdout.split(b"\x00") if entry)
    files: list[Path] = []
    total_raw = len(entries)
    for i, raw_entry in enumerate(entries):
        if progress_callback is not None:
            progress_callback(i, total_raw)
        rel = raw_entry.decode("utf-8", errors="replace")
        if not scan_selects(rel, patterns):
            continue
        path = (repo_root / rel).resolve()
        if not path.is_file():
            continue
        if should_skip_path(path, repo_root=repo_root):
            continue
        if ignore_spec is not None and ignore_spec.match_file(rel):
            continue
        if detect_language(path) is None:
            continue
        files.append(path)
    return sorted(set(files))


def _iter_glob_source_files(
    repo_root: Path,
    patterns: list[str],
    ignore_spec: Any | None = None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    files: list[Path] = []
    seen_inode: set[int] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if progress_callback is not None:
                progress_callback(len(files), 0)  # type: ignore[arg-type]
            if not path.is_file():
                continue
            if should_skip_path(path, repo_root=repo_root):
                continue
            if ignore_spec is not None:
                try:
                    rel = path.relative_to(repo_root).as_posix()
                except ValueError:
                    rel = path.name
                if ignore_spec.match_file(rel):
                    continue
            if detect_language(path) is None:
                continue
            # Deduplicate by inode to handle case-insensitive filesystems
            # (e.g. macOS APFS) where Makefile/makefile etc. refer to the same
            # file but pathlib treats them as distinct Path objects.
            try:
                ino = path.stat().st_ino
            except OSError:
                ino = 0
            if ino and ino in seen_inode:
                continue
            if ino:
                seen_inode.add(ino)
            files.append(path)
    return sorted(set(files))


def build_reference_graph(
    repo_root: str | Path, files: list[str] | None = None
) -> tuple[DiGraph, dict[str, list[Tag]]]:
    """Build a file graph from symbol references to definitions."""
    root = Path(repo_root)
    cache_key: tuple[str, tuple[str, ...] | None] = (
        str(root.resolve()),
        tuple(sorted(files)) if files is not None else None,
    )
    if cache_key in _REFERENCE_GRAPH_CACHE:
        return _REFERENCE_GRAPH_CACHE[cache_key]
    paths = [root / file for file in files] if files else iter_source_files(root)
    tags_by_file: dict[str, list[Tag]] = {}
    definitions: dict[str, set[str]] = {}
    # Persistent, mtime-keyed tag cache (default-on; LEMONCROW_REPOMAP_TAG_CACHE
    # disables). extract_tags() is the dominant cost; the cache lets fresh
    # processes skip re-parsing files whose (mtime, size) are unchanged. The
    # cache is correctness-preserving via mtime invalidation and degrades to
    # in-memory on any DB failure, so graph building behaves identically.
    cache = TagCache.for_repo(root)
    try:
        for path in paths:
            tags = cache.get(path)
            if tags is None:
                try:
                    tags = extract_tags(path)
                except OSError:
                    tags = []
                else:
                    cache.put(path, tags)
            rel = str(path.relative_to(root)) if path.is_absolute() or path.exists() else str(path)
            tags_by_file[rel] = tags
            for tag in tags:
                if tag.kind == "definition":
                    definitions.setdefault(tag.name, set()).add(rel)
    finally:
        cache.close()

    graph = DiGraph()
    for rel in tags_by_file:
        graph.add_node(rel)
    for rel, tags in tags_by_file.items():
        for tag in tags:
            if tag.kind != "reference":
                continue
            for def_file in definitions.get(tag.name, set()):
                if def_file == rel:
                    continue
                weight = float(graph.get_edge_data(rel, def_file, {}).get("weight", 0.0)) + 1.0
                graph.add_edge(rel, def_file, weight=weight)
    _REFERENCE_GRAPH_CACHE[cache_key] = (graph, tags_by_file)
    return graph, tags_by_file


__all__ = [
    "build_reference_graph",
    "iter_source_files",
    "should_skip_path",
    "should_skip_relative_path",
]
