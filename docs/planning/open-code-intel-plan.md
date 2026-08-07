# Open Code-Intel Plan

Deliver ten code-intelligence features in the public repo, without access to the
closed `lemoncrow.pro` engine source.

**Scope note.** All ten items below are deliverable. "Adding a language" was a
*separate* blocked item — it needs `_LANG_CONFIG` inside
`pro/capabilities/semantic_file_memory/treesitter_ast.so` — and was never one of
the ten. Nothing is dropped from this plan. F9 adds LSP *server adapters*, which
lives in open source (`infra/code_intel/lsp/registry.py`) and is unrelated to the
indexer's language table.

---

## 0. Constraints — read before writing code

### 0.1 The engine is closed and must stay untouched

`src/lemoncrow/pro/` is 275 `.so` files with zero `.py`. It owns:

- the DDL for `code_context.sqlite`, `intel.sqlite`, `fts.sqlite`, `vectors.sqlite`
- all extraction (symbols, imports, call edges, references)
- `engine_state.index_version` (currently 6) and `indexer_semantics_version` (2)
- the write lock `code_context.sqlite.indexlock` (`IndexLockTimeout`,
  `LEMONCROW_INDEX_LOCK_TIMEOUT_S`)

**Rule: read the engine's databases, never write to them.** A `--reindex` drops
and rebuilds them, and concurrent writes race the autosync worker
(`_autosync_poll_ms=10000`, `_autosync_debounce_ms=500`) plus the edit-triggered
reindex thread at `mcp_server.py:7290-7316`. All new state goes in a sidecar DB
we own (F0).

### 0.2 Shadowed modules in vendored-engine checkouts

When a compiled engine is vendored next to the source tree, ten open modules
resolve from the mypyc group `.so` instead of their `.py` once the engine loads.
Import order decides which copy you get, so they behave differently under
`pytest` than under the running MCP server:

```
core.environment                     infra.storage.vector
core.capabilities.pricing            infra.tree_sitter.tags
core.capabilities.workflow_spawn     infra.internal_llm.litellm_client
infra.code_intel.languages           infra.internal_llm.openai_client
infra.embeddings.ollama_embedder     infra.internal_llm.result
```

**Rule: never put new logic in those ten files.** New modules can never be
shadowed — the group only contains modules that existed at build time. Every
feature here lands in a new module plus a wiring edit to `mcp_server.py`, which
is live source.

One consequence worth stating: `HIDDEN_LLM_TOOLS` lives in the shadowed
`core/environment.py`, so tool visibility must be changed through the live-source
wrapper instead:

```python
# mcp_server.py:310-311
def _tool_visible_to_llm(tool_name: str, spec: dict[str, Any]) -> bool:
    return mcp_tool_visible_to_llm(tool_name)
```

### 0.3 mypyc safety

Release wheels compile nearly all of `src/lemoncrow/` with mypyc. New code must
survive that. Enforced by `tests/test_mypyc_compile_safety.py` (AST scan, always
on):

- native instances have **no `__dict__`** — no `self.__dict__.setdefault(...)`,
  declare real `__init__` attributes
- native instances have **no `__weakref__`** — route through
  `core/foundation/weakref_token.py`'s `WeakRefToken`
- native functions reject keyword args through the compiled wrapper — the
  `@mcp_tool` handler already takes a single `args: dict`, so this only bites
  direct internal calls

`make typecheck` is `mypy --strict src`. New modules must be fully annotated.

### 0.4 Tool registration

`@mcp_tool` (`gateway/adapters/mcp/framework.py:173`) derives the MCP input
schema from the function signature via Pydantic `create_model`, strips schema
noise, coerces stringified scalars, and rejects unknown args. Adding a tool is:
write the function, decorate it, and decide its visibility. The docstring becomes
the advertised description, so keep first lines tight — they cost tokens on every
request.

---

## F0. Shared foundation — build this first

Two new modules everything else depends on.

### `src/lemoncrow/infra/code_intel/store.py` — read-only engine accessor

```python
def workspace_dir(repo_root: Path | str = ".") -> Path
def open_ro(db: str, repo_root: Path | str = ".") -> sqlite3.Connection
```

- locate the workspace with the existing open helper
  `core.foundation.paths.resolve_workspace_store_dir(workspace_root=...)`,
  which returns `<repo>/.lemoncrow/workspace`
- open with `sqlite3.connect(f"file:{path}?mode=ro", uri=True)` — read-only is
  structural protection against 0.1, not just discipline
- set `busy_timeout`; the engine holds WAL writers
- typed accessors returning dataclasses/TypedDicts over `symbols`, `imports`,
  `call_edges`, `references`, `files`, `centrality_map`
- a `snapshot()` returning `engine_state.index_version` + row counts, so every
  feature can stamp its output with the index generation it read

### `src/lemoncrow/infra/code_intel/sidecar.py` — our writable DB

- `sidecar.sqlite` in the same workspace dir, owned entirely by open code
- `sidecar_meta(schema_version, engine_index_version, built_at)` — every table
  records which engine generation it was derived from, so stale sidecar data is
  detectable rather than silently wrong
- migrations by `schema_version` integer; no dependency on engine DDL

**Effort:** 3-5 days including tests.

---

## F1. Repo-correct file-graph analytics

### Problem

`blast_radius`, `dead_code`, `cycles`, `coupling`, and `topology` do not read the
repo's `imports` table. They call
`_semantic_file_memory(...).graph_analytics()`, which reads
`~/.lemoncrow/semantic_file_index.json` — a machine-global, cross-repo,
2000-entry LRU populated opportunistically by `summarize_file`.

Measured on this repo: 51 analyzed files mixed across four unrelated projects,
`dead_code` calls 20 of 51 "dead", and `blast_radius` on
`gateway/adapters/mcp_server.py` returns **empty importers, risk "low"**.

The real table answers correctly:

```sql
SELECT source_file FROM imports WHERE target_file LIKE '%telemetry/emit.py';
-- telemetry/__init__.py, telemetry/frustration.py, tests/core/test_product_telemetry.py
```

### Delivery

New `src/lemoncrow/infra/code_intel/file_graph.py`:

- build an in-memory adjacency map from `imports` where `target_file IS NOT NULL`
- `blast_radius(path)` — reverse-import transitive closure, plus affected tests
  (paths matching the repo's test roots), plus a risk tier from closure size
- `dead_code()` — files with no inbound importers, excluding entry points
  (`__main__`, console scripts from `pyproject.toml`, `conftest.py`, test files)
- `cycles()` — Tarjan SCCs of size >= 2
- `coupling()` — afferent/efferent counts and Martin instability `Ce/(Ca+Ce)`
- `topology()` — module-boundary/god-module discovery over the same graph

Wire `_op_graph` (`mcp_server.py:8780-8844`) to call this instead of the
semantic-file-index path. Keep the existing response shape so no caller changes.

### Known limitation to surface, not hide

**6,007 of 8,096 `imports` rows have `target_file` NULL** — only intra-repo
modules resolve; stdlib and third-party are dangling. Report
`resolved_edges` / `unresolved_edges` in the response so a caller can tell a
small blast radius from an unresolved one. This is exactly the failure mode that
made the current implementation look plausible while returning nothing.

### Tests

`tests/infra/code_intel/test_file_graph.py` — fixture repo with a known import
DAG: closure correctness, cycle detection, instability arithmetic, entry-point
exclusion, and a regression asserting non-empty importers for a file that has
them (the bug this fixes).

**Effort:** 5-8 days. **Risk:** low. Highest value in the plan.

---

## F2. `detect_changes` — diff to impacted symbols

### Delivery

New `src/lemoncrow/infra/code_intel/change_impact.py`, new tool
`code_changes`:

```
code_changes(base_ref="HEAD", paths=None, depth=1, limit=100)
```

Pipeline:

1. `git diff --unified=0 <base>...<head>` → changed line ranges per file
   (subprocess, not the `bash` tool — this is library code)
2. map ranges onto `symbols.start_line`/`end_line` → the set of changed symbols
   with `symbol_id`, `qualified_name`, `kind`
3. reverse lookup through `call_edges` on `callee_name`/`callee_short_name` and
   through `references` on `symbol_name` → impacted callers
4. expand to `depth` (default 1; the graph is name-keyed, so deep traversal
   compounds imprecision — see 0.1 of F9)
5. classify risk per changed symbol: exported vs private, caller count,
   test-file coverage among callers

### Precision caveat, stamped in the response

`call_edges` stores the callee as raw dotted call text with no `symbol_id`, so
reverse lookup is name-matched and over-reports on common names. Return
`match_kind: "name" | "resolved"` per edge — `"resolved"` only once F9 lands and
a sidecar resolution exists. Callers can then filter.

### Tests

`tests/infra/code_intel/test_change_impact.py` — synthetic git repo via
`git init` in `tmp_path`: line-range→symbol mapping including multi-hunk files,
renames, deletions (symbol gone from the index), and a symbol whose name
collides with a builtin.

**Effort:** 8-12 days. **Risk:** medium — git edge cases.

**Shipped** as `f866112d` — `infra/code_intel/change_impact.py`, tool
`code_changes`, advertised. Diffs run against `git merge-base <base> HEAD` so a
base branch that moved independently does not pollute the result.

---

## F3. Index coverage

### Delivery

New tool `code_coverage_check` in a new
`src/lemoncrow/infra/code_intel/coverage.py`:

```
code_coverage_check(paths=None, repo_root=".")
```

For each requested path (or the whole repo):

- **indexed** — present in `files` with matching `content_hash`
- **stale** — present but `content_hash`/`mtime_ns` differ from disk
- **missing** — on disk, not in `files`
- **excluded** — matched by the indexer's ignore rules
- **unparsed** — in `files` but zero rows in `symbols` (language unsupported or
  parse failed)

Return `engine_index_version` and total counts alongside.

### Why this matters more than it looks

It is the primitive that makes "I searched and found nothing" auditable. Every
other feature here, and any downstream reviewer, can distinguish *absent* from
*not indexed*. Cheap, and it de-risks everything else.

### Tests

`tests/infra/code_intel/test_coverage.py` — each of the five states constructed
explicitly, including a file mutated after indexing (stale) and a `.txt` file
(unparsed).

**Effort:** 3-5 days. **Risk:** low.

---

## F4. Fix the tool broker

### Problem

`_tool_broker_handler` (`mcp_server.py:10763`) is documented as the
*"deterministic fallback for rare LemonCrow tools hidden by the core profile"*
but can reach exactly one tool, `statusline_segment`:

- `action: "call"` rejects anything in `_CORE_MCP_TOOLS` with *"is already
  exposed; call it directly"* (line 10793) — but `relations` and `grep` are in
  **both** `_CORE_MCP_TOOLS` and `HIDDEN_LLM_TOOLS`, so they are refused as
  "exposed" while never being advertised
- it then rejects anything failing `_tool_visible_to_llm` unless it is in
  `_BROKER_HIDDEN_CALLABLE = frozenset({"statusline_segment"})`
- `action: "search"` skips core tools *and* non-visible tools (10771-10774) —
  every hidden tool fails the second filter, so it can never return a match

Verified live: `tool(action="call", name="graph")` →
`unknown or unavailable tool`; three different `action="search"` queries →
`{"matches": []}`.

### Delivery

Change the guard from "is it in the core profile" to "is it actually advertised
right now":

- reject only when the tool is genuinely in the current `tools/list` response
- allow call-through for any registered, non-mutating tool that is hidden
- `action: "search"` should search **hidden** tools — invert the visibility
  filter, which is the only thing that makes the action meaningful

Keep a deny-list for tools that must never be brokered (`agent`, `workflow`,
`sql`, `codemod`, `mcp`) rather than an allow-list of one.

### Tests

Extend `tests/gateway/test_cap_tools_list_gate.py`: assert `relations` and
`graph` are callable through the broker under `LEMONCROW_MCP_TOOL_PROFILE=core`,
that `search` returns non-empty for a hidden tool, and that the deny-list holds.

**Effort:** 2-3 days. **Risk:** low. Unblocks `relations`/`graph` for every
agent, including code-review workers.

---

## F5. Graph query API

### Delivery

New `src/lemoncrow/infra/code_intel/query.py`, new tool `code_query`.

Not Cypher. A constrained, validated predicate API compiled to SQL over
`symbols` + `imports` + `call_edges` + `references`:

```
code_query(
  select="symbols" | "callers" | "callees" | "importers" | "references",
  where={"kind": "function", "file_path_like": "src/%", "name_regex": "^_"},
  order_by="centrality" | "name" | "callers",
  limit=50,
)
```

**Never** accept raw SQL. Whitelist columns and operators; bind every value.
This is a code-search surface exposed to model-generated input, so injection is a
real threat, not a theoretical one.

This covers most of what cbm's `query_graph` is used for in practice —
"functions matching X with no callers", "files importing Y ordered by
centrality" — without a query-language parser to maintain.

### Tests

`tests/infra/code_intel/test_query.py` — each predicate; explicit injection
attempts (`'; DROP TABLE`, unicode quotes, nested dicts); limit enforcement.

**Effort:** 8-12 days. **Risk:** medium — API design churn. Ship narrow.

**Shipped** as `e832fc70` — `infra/code_intel/query.py`, tool `code_query`,
advertised. `describe=true` returns the field whitelist as data. Ordering by
`centrality` and `callers` is applied in Python over a capped scan rather than
in SQL, because those scores live in a different database from `symbols`; the
response carries `scanned` and `scan_capped` so the cap is never invisible.

---

## F6. Clone detection

### Delivery

`core/foundation/_minhash.py` already implements MinHash (SHA-1 32-bit + M61
linear permutations, `jaccard()` at L60-64, drop-in for `datasketch`, zero deps).
It is currently wired only to context/retrieval dedup.

New `src/lemoncrow/infra/code_intel/clones.py`:

- for each symbol, read `file_path` + `start_byte`/`end_byte` from `symbols`,
  slice the source, normalise (strip comments/whitespace), shingle at k=5 tokens
- MinHash signature at 128 permutations (matches `_MINHASH_PERMUTATIONS` used
  elsewhere)
- **LSH banding** to avoid O(n²) — 23,562 symbols in this repo alone makes the
  all-pairs comparison untenable. This is the one genuinely new algorithm.
- persist to sidecar `symbol_clones(symbol_a, symbol_b, jaccard, built_at)`
- skip symbols under a minimum token count (mirrors `_MIN_DEDUP_TOKENS`)

Expose via `code_query(select="clones")` (F5) rather than a new tool, to hold the
surface flat.

### Tests

`tests/infra/code_intel/test_clones.py` — exact duplicate scores 1.0;
rename-only clone scores high; unrelated functions score low; LSH recall against
a brute-force baseline on a small fixture.

**Effort:** 8-12 days. **Risk:** medium — LSH tuning. Depends on F0, F5.

### Shipped — `461ec349`

`lc code clones` builds it; `code_query(select="clones")` reads it. Measured on
this repo: 12,066 symbols compared in 33s, 3,409 LSH candidates, 1,909 pairs at
threshold 0.8. Banding is 16 bands x 8 rows.

Three deviations, all forced by what the repository actually contains:

1. **`_MINHASH_PERMUTATIONS` and `_MIN_DEDUP_TOKENS` do not exist.** The plan
   cites them as constants to match; neither appears anywhere in the tree, and
   `_minhash.py` has no open-source importers at all. `clones.py` declares its
   own `NUM_PERM = 128` (the `MinHash` default) and `MIN_TOKENS = 40`.
2. **Normalisation had to placeholder identifiers, not merely strip comments and
   whitespace.** As specified, a copied function whose locals were all renamed
   scored **0.039** — at k=5 nearly every shingle contains at least one renamed
   identifier, so nearly every shingle differs. That fails the section's own
   acceptance criterion ("rename-only clone scores high"), so identifiers and
   numeric literals are replaced by placeholders before shingling. Keywords
   survive, which is what keeps control-flow shape in the signature.
3. **String literals are kept; only docstrings are stripped.** Stripping every
   string, as "strip comments/whitespace" implies, produced 2,968 pairs whose
   top scorers were every `to_dict` in the codebase matching every other at
   1.000 — once identifiers are placeholdered, a `to_dict`'s dict keys are the
   only thing distinguishing it. Keeping literals cut that to 1,909 and put real
   duplication at the top.

One addition the plan did not call for: the clone table is the first sidecar-
backed `code_query` select, so it is the first that can be *absent* rather than
empty. `load_clones` and `code_query` raise `ClonesStale` when it was never
built or the engine has reindexed past it, rather than returning the zero rows
that would read as "this code has no duplicates".

---

## F7. Index export / import

### Delivery

New `src/lemoncrow/infra/code_intel/portable.py`, CLI subcommands under the
existing `lc code` group (`gateway/cli/commands/code.py`, live source):

```
lc code export [--out .lemoncrow/index.tar.zst] [--tier best|fast]
lc code import [--from .lemoncrow/index.tar.zst]
```

- `VACUUM INTO` each of the five DBs into a temp dir (compacts, drops WAL)
- tar + zstd; two tiers — `best` (zstd 9, drop derived indexes) on explicit
  export, `fast` (zstd 3) for incremental refresh
- manifest: engine `index_version`, `indexer_semantics_version`, LemonCrow
  version, repo HEAD sha, row counts, sidecar `schema_version`
- **import refuses on version mismatch** rather than producing a subtly wrong
  graph. The engine owns those numbers; we cannot migrate its data.
- import bootstraps into an empty workspace, then the engine's normal
  incremental pass fills the local diff

`zstandard` is a new dependency — put it behind an extra, not the base install.

### Tests

`tests/infra/code_intel/test_portable.py` — round-trip fidelity (row counts and
a sampled query match), version-mismatch refusal, corrupt-archive handling.

**Effort:** 5-8 days. **Risk:** low-medium. Deliberately does **not** commit the
artifact to git by default.

**Shipped** as `8235fdaf` — `infra/code_intel/portable.py`, plus `lc code
export` / `lc code import`. Two deviations:

> **`zstandard` is an accelerator, not a requirement.** It sits behind a new
> `portable` extra as planned, but export falls back to stdlib lzma when it is
> absent rather than failing. The manifest names the codec and import reads it,
> so the feature works on a base install and gets smaller archives with the
> extra.
>
> **The `best` tier does not drop derived indexes.** The engine's DDL is closed;
> an index dropped on export is one open code cannot recreate, so the import
> would hand back a database the engine expects to be complete. The two tiers
> differ by compression level only.

One addition the plan did not call for: the archive is treated as untrusted
input. Members must be regular files whose names are on a fixed allow-list, so a
traversal path or a symlink is refused outright. A tar file is a format someone
else can write, and "a teammate sent it" is not provenance.

---

## F8. Cross-service and IaC edges

### Delivery

New `src/lemoncrow/infra/code_intel/edges_ext.py`, writing to sidecar tables.

There is precedent to follow rather than invent: `_synthesize_edges_for_paths`
(`mcp_server.py:8758-8777`) already returns heuristic edges tagged
`provenance="heuristic"`, deliberately kept in a **separate list, never merged**
into the static call graph. Keep that discipline.

- **HTTP routes** — server-side decorators/handlers (Flask, FastAPI, Express,
  Spring) → `Route` nodes; client-side `fetch`/`axios`/`requests` calls with
  literal or resolvable-constant paths → `HTTP_CALLS` edges matched to routes by
  path template + method, with a confidence score
- **Channels** — `EMITS`/`LISTENS_ON` for Socket.IO, `EventEmitter`, and generic
  pub-sub, with constant resolution
- **IaC** — Dockerfile, Kubernetes manifests, Kustomize overlays as nodes with
  `IMPORTS`-style references. Note the grammar pack ships a `dockerfile` grammar
  but the engine gates it out, so parse these directly rather than through the
  indexer.

Every edge carries `provenance` and `confidence`. Never present a heuristic edge
with the same authority as a parsed one.

### Tests

`tests/infra/code_intel/test_edges_ext.py` — fixture services in Python and
TypeScript with a known route↔call pairing; a deliberately unresolvable dynamic
path asserting *no* edge rather than a guess.

**Effort:** 15-25 days, and it grows with every framework. **Risk:** high —
heuristic surface with an unbounded tail. Ship two frameworks, measure, then
decide.

---

## F9. LSP resolution sidecar

### The gap this attacks

`call_edges` is name-keyed: `callee_name` is raw dotted call text, no
`symbol_id`, no import binding. Lookup is
`WHERE callee_name = X OR callee_short_name = X`. The consequence is measurable —
top centrality nodes on this repo are `str` (in-degree 4,741), `len` (2,154),
`isinstance` (2,100). Builtins. 48% of the 81,095 edges are dotted attribute
calls, and `_code_context_engine` is defined in two files whose edges collapse
into one name bucket.

We cannot change extraction. We *can* resolve after the fact.

### Delivery

`infra/code_intel/lsp/` is open and advertises itself as the seam:

> *"This is the extensibility seam. To add a language, append one entry mapping
> its canonical name to the argv of a stdio language server."*

Only `kotlin` is wired, and nothing in production calls `LspClient`.

1. **Adapters** — add `python` (pyright/pylsp), `typescript`/`javascript`
   (typescript-language-server), `go` (gopls), `rust` (rust-analyzer) to
   `SERVER_COMMANDS`. Availability stays a runtime decision via the transport's
   `is_available`.
2. **Resolver** — new `infra/code_intel/resolve.py`: for each `call_edges` row,
   ask `textDocument/definition` at `(caller_file_path, call_line, call_column)`
   — the columns are already stored — and map the result back to a
   `symbols.symbol_id`.
3. **Sidecar table** —
   `resolved_call_edges(caller_symbol_id, callee_symbol_id, confidence, method, engine_index_version)`
   with `method` in `lsp | name | ambiguous`, mirroring the tiering already
   described in `tests/core/test_edge_resolution.py`.
4. **Consumers prefer resolved edges**, falling back to name matching and saying
   which they used — F2's `match_kind` becomes meaningful here.

Run as a bounded background pass, not inline: batch by file, cap wall time,
checkpoint, resume. Fail-open per site — an unresolvable call stays name-matched
rather than disappearing.

### Tests

`tests/infra/code_intel/test_resolve.py` — fake LSP transport (the existing
`test_lsp_client.py` pattern) covering the two-definitions-same-name case,
unresolvable dynamic dispatch, and server-unavailable degradation.

**Effort:** 20-30 days. **Risk:** high — LSP servers are slow, chatty, and
fail in bespoke ways at scale. Highest-value item after F1; also the most likely
to need a second pass.

---

## F10. Open semantic search

### Current state

- `search` (`tool_smart_search`, `mcp_server.py:10607`) is a **real** hybrid
  implementation, not a stub. Its lexical and graph legs work today.
- Seven embedders are implemented and live source: `openai`, `letta`, `ollama`,
  `bge` (BGE-Code-v1), `nomic` (nomic-embed-code), `hf`, `null`.
  `factory.py:155` returns `NullEmbedder` when `LEMONCROW_CODE_EMBEDDER` is unset.
- `vectors.sqlite` has a complete `symbol_vectors` schema — `embedder_name`,
  `embedding_dim`, `index_version`, `vector_blob` — and **0 rows**.
- `SymbolAnnIndex` exists in the closed engine but `_HNSW is None`
  (*"HNSW removed: candidate_ids always returns None"*), so it degrades to exact
  cosine, behind `LEMONCROW_ANN_RETRIEVAL` (default false).
- Visibility is a literal `"search"` in `HIDDEN_LLM_TOOLS`
  (`core/environment.py:66`) with no runtime embedder probe.

### Delivery

Bypass the closed ANN entirely rather than fight it.

1. **Backfill** — new `infra/code_intel/embed_index.py`: read `symbols`, slice
   source, embed via the open `make_code_embedder()`, write vectors to a
   **sidecar** table `symbol_vectors_open` (not the engine's `vectors.sqlite` —
   see 0.1). Stamp `embedder_name` + `embedding_dim` + `engine_index_version`;
   refuse to mix vector spaces, matching the engine's own provenance discipline.
2. **Search** — brute-force cosine in numpy over the backfilled matrix. At
   23,562 symbols × 768 dims this is ~72 MB float32 and single-digit
   milliseconds; ANN is not needed at this scale. Add LSH only if a repo makes it
   necessary.
3. **Surface** — new tool `code_semantic_search`, hybrid-ranked against the
   existing lexical results.
4. **Visibility** — flip it on through `_tool_visible_to_llm`
   (`mcp_server.py:310-311`), gated on a real runtime check: embedder configured
   **and** backfill non-empty. That is strictly better than the current static
   list, which cannot tell whether an embedder exists.

`nomic-embed-code` needs the `semantic` extra (torch, sentence-transformers) —
keep it optional. `ollama` gives a no-API-key path without the torch dependency.

### Tests

`tests/infra/code_intel/test_embed_index.py` — backfill idempotence,
vector-space mismatch refusal, cosine ranking against a known-good ordering with
a deterministic fake embedder, and the visibility gate in both states.

**Effort:** 10-15 days. **Risk:** medium — mostly dependency weight.

### Not built — deliberately, and the reason is a precondition this plan missed

Every part of F10 downstream of the backfill is gated on an embedder existing.
`LEMONCROW_CODE_EMBEDDER` is unset in the target environment, `factory.py`
returns `NullEmbedder`, and there is no `ollama` binary — so a shipped
`code_semantic_search` would advertise a tool that returns nothing, and its
acceptance could only ever have been exercised against a fake embedder in
tests. The requester declined it on exactly that basis: do not build something
that cannot be verified or that will not function.

What F10 needs before it is worth building is therefore not engineering time but
a decision about the embedding backend — `ollama` (no API key, no torch) or the
`semantic` extra (BGE-Code-v1, ~2GB of torch). Note also that **`numpy` is not a
base dependency**, only a member of the `vector` and `semantic` extras, so
step 2's "brute-force cosine in numpy" cannot be written against the base
install as the plan assumes.

F10 stays open. It is unblocked the moment an embedder is chosen.

---

## F11. Evict the cached engine on an index-version bump

### The bug, observed in production

`_code_engine_cache` (`mcp_server.py:8330`) is populate-once, keyed by resolved
repo path, and **never invalidated in production**. The only `.clear()` in the
file is inside `_reset_runtime_cache_for_testing()` at `:1071`:

```python
engine = _code_engine_cache.get(cache_key)
if engine is None:                        # the only condition that ever rebuilds
    ...
    engine = CodeContextEngine(resolved)
    _code_engine_cache[cache_key] = engine
return engine
```

A daemon builds one `CodeContextEngine` per repo and reuses it for its whole
life, across any number of reindexes. Measured on `~/Dev/claude-plugins`: a
daemon started at `index_version 1` was still serving at version 23 — 22
reindexes later — and `code_search` returned **empty for every query**,
including symbols plainly present in the index:

```
# daemon (4h old, engine built at index_version 1)
empty_telemetry  -> no results

# fresh interpreter, same install, same on-disk index
empty_telemetry  -> exact_match: True,
                    plugins/code-review/tools/python/code_review_schema.py
```

It fails **silently**: no error, no warning, just nothing found. An agent reads
that as "this symbol has no callers" and proceeds on a false premise. That is
strictly worse than an exception.

### What it is not

Not stale file handles — `lsof` shows the daemon holding live inodes at current
sizes, no `(deleted)` markers, so reopening connections would not help. Not
stale *code* either: the daemon reaper already restarts on a
`_code_fingerprint()` mismatch (`mcp_daemon.py:753`, 30-minute drain), and that
mechanism is healthy and unrelated. The staleness is in-process engine state
built against a superseded generation, so the cached object itself has to go.

### Delivery

Stamp the cache entry with the index version it was built against and rebuild on
mismatch. F0's `store.snapshot().index_version` already reads it over an
independent read-only connection, so this needs no engine cooperation:

- store `(engine, built_at_index_version, last_checked_monotonic)` per cache key
- re-read `engine_state.index_version` at most every `N` seconds (start at 5) so
  a hot `code_search` path does not pay a SQLite open per call
- on mismatch, drop the entry and rebuild under the existing lock
- log at INFO when an eviction fires — this failure was invisible for hours, and
  the fix should not be equally quiet

### Fail loud, never empty

Eviction alone is not enough. A reindex is not atomic: for roughly two minutes
while the index rolled 1 → 23, **every** query returned `no exact match` with
zero results. Empty is the single worst way for a code-intelligence tool to
fail, because it is indistinguishable from a true negative — a reviewer reads
"no matches" as "this symbol has no callers" and files a wrong finding with
full confidence. Silence is a wrong answer delivered as a right one.

So every read path must be able to say which of three things happened:

| state | meaning | response |
|---|---|---|
| `fresh` | engine version == on-disk version | results, as today |
| `rebuilt` | versions differed; engine was evicted and rebuilt | results, plus a note |
| `rebuilding` | the index is mid-write (lock held, or tables incomplete) | **raise**, do not return `[]` |

Carry the state on the response rather than inventing a new tool: F0's
`snapshot()` already returns `index_version`, and F3 already established the
precedent that a negative result has to be auditable. An empty result set from a
`rebuilding` index must be an error the caller can catch, not an empty list it
can mistake for truth.

Applies to `_code_engine_cache` and to `_scoped_context_cache`
(`mcp_server.py:8332`), which wraps the engine and therefore inherits the defect
transitively — a `ScopedContextCapability` captures the engine it was
constructed with, so leaving it cached hands back the object just evicted.

> **Deviation (shipped).** This plan also named `_semantic_file_memory_cache`
> (`mcp_server.py:5440`). It was **not** changed. The shape is the same but the
> staleness source is not: that capability indexes
> `~/.lemoncrow/semantic_file_index.json`, not the engine's databases, and its
> `SymbolIndex` already memoizes by corpus snapshot. Gating it on
> `engine_state.index_version` would couple it to a generation number that says
> nothing about its corpus, and would rebuild a 140 ms BM25 index on every
> unrelated reindex.

### Tests

`tests/gateway/test_code_engine_cache_invalidation.py` — a fixture workspace
whose `engine_state.index_version` is bumped between two calls asserts a
rebuilt engine; an unchanged version asserts the *same* object (no rebuild
churn); the throttle is respected; concurrent callers rebuild once, not N times.
Separately: a workspace with the index lock held asserts a raised error rather
than an empty result set — the regression that matters most.

**Effort:** 3-4 days. **Risk:** low. Depends on F0 (merged).

**Shipped** as `8ea1eb15` — `infra/code_intel/freshness.py` (`index_state`,
`VersionedEngineCache`, `IndexRebuilding`). The tests landed in two files rather
than the one named above: `tests/infra/code_intel/test_freshness.py` for the
module (it needs the `make_workspace` fixture, which is only reachable from that
directory) and `tests/gateway/test_code_engine_cache_invalidation.py` for the
server wiring.

### Not in scope here

Exhaustive callsite enumeration is **already solved** and needs no work: it is
`relations`, not `code_search`. The two answer different questions and the
distinction is the whole point — `code_search` ranks by relevance and returns
the best matches; `relations` enumerates from `intel.references` and returns
all of them with `reference_count` and `truncated`. Verified against
`merge_telemetry` in `claude-plugins`: 7 callsites, `truncated: false`, grouped
by file, each anchored with caller name, line and column range — matching a
`grep` oracle exactly. F4 made it reachable through the broker; the remaining
work is routing callers to the right tool, not building a new one.

---

## Sequencing

| Phase | Items | Rationale | Effort |
|---|---|---|---|
| **A** | F0, F4, F3, F1 | Foundation, then the broker fix and the two features that make everything else auditable and correct | 13-21 d |
| **B** | F11, F2, F5, F7 | The correctness fix first, then the review-facing capability, the query surface, the shareable artifact | 23-35 d |
| **C** | F6, F10 | Both depend on F0 and benefit from F5's surface | 18-27 d |
| **D** | F9, F8 | Highest risk, highest cost; F9 retroactively improves F2 and F5 | 35-55 d |

**Total: 89-138 engineer-days, roughly 4-7 engineer-months.**

**Phase A shipped** — PR #1, merged as `71fface1`. Both acceptance checks pass:
`blast_radius` on `telemetry/emit.py` returns its three real importers (was
empty, risk "low"), and `graph` is reachable through the broker under the core
profile.

**F11 goes first in Phase B.** It is two days, it is a silent-wrong-answer bug
rather than a missing feature, and every later phase inherits it: F2's impact
analysis, F5's query surface and F6's clone detection all read through the same
cached engine, so shipping them onto an engine that can silently serve a
superseded generation just multiplies the blast radius of the defect.

**Phase B built** — branch `worktree-pln-1633-phase-b`, five commits, unpushed:

| Item | Commit | Surface |
|---|---|---|
| F11 | `8ea1eb15` | `infra/code_intel/freshness.py` |
| F2 | `f866112d` | `code_changes` tool |
| F5 | `e832fc70` | `code_query` tool |
| F7 | `8235fdaf` | `lc code export` / `import` |
| F12 | `67d56649` | routing + completeness contract (below) |

Deviations are recorded against each item above. The whole `tests/gateway/`
suite fails identically to `origin/main` (53 failures, byte-for-byte the same
list) and `mypy --strict src` reports the same 257 pre-existing errors, none of
them in the new modules.

---

## F12. Say which objective answered the question

Added during Phase B, not in the original ten. It closes the gap the
code-review plugin reported and Phase B's other items only half-addressed.

### The gap

Two objectives live in this subsystem and nothing said which one a caller got.
`code_search` **ranks** — given a question, return the smallest sufficient
context. `relations`, `code_changes`, `code_query` and the file-graph analytics
**enumerate** — given a subject, return everything that matches. A ranked
top-N read as a complete caller set is how a confident wrong finding gets filed.

Worse, the enumerative tool was unreachable. Measured:

```
before: core advertised (5): bash, code_search, edit, read, web_fetch
        relations advertised: False under BOTH profiles
after:  core advertised (7): + relations, code_coverage_check
        full advertised (9): + relations
```

`relations` sat in both `_CORE_MCP_TOOLS` and `HIDDEN_LLM_TOOLS`, so no profile
ever advertised it. An agent under `core` saw one code-intel tool — the ranked
one — and zero enumerative ones. **`code_changes` does not substitute:** a
builder about to edit a symbol has a symbol, not a diff, and would have to make
the edit first and then ask what broke.

### Delivery

- `_FORCE_VISIBLE_TOOLS` in `mcp_server.py` (live source — `HIDDEN_LLM_TOOLS`
  resolves from a compiled `.so` under a vendored engine, per 0.2). `_tool_mode`
  follows it: advertised-and-hidden is not a state anything downstream reads.
- `code_coverage_check` joins `_CORE_MCP_TOOLS`; nothing else in that set can
  audit a negative result.
- `infra/code_intel/completeness.py` — `objective` on every retrieval response,
  beside the pre-limit count and `truncated` most surfaces already carried.
  Consumers evaluate `objective == "exhaustive" and not truncated` instead of
  inferring completeness from a tool name.
- Engine ops with no oracle (`pattern`, `node`) are **absent** from the map
  rather than guessed. Absent reads as unclassified, never as exhaustive.
- `blast_radius` is bounded, closing a gap F1 left: it returned every importer
  inline (390 paths, 18 KB for `telemetry/emit.py`). Counts are computed before
  the cut and `risk_level` from the full closure, so truncation costs detail and
  never the answer to "how big is this change".

### Deliberately not built

A referral field on `code_search` ("this looks like enumeration — try
`relations`") was designed and dropped. Detecting enumeration intent from a
query string is a heuristic in the routing layer, and heuristics there are what
produce the NL-vs-keyword sensitivity the plugin measured. With `relations`
visible the agent can route on its own.

**Effort:** ~1 day. **Risk:** low, but it moved three surface-lock tests — each
narrowed to the new contract, none loosened.

**Phase C partially shipped** — F6 landed on `worktree-pln-1633-phase-c`
(`461ec349`). F10 was not built: it is gated on an embedder that does not exist
in the target environment, so it would have shipped a tool that cannot function.
See its section for what unblocks it. Phase C is therefore not closed, and
"remaining work" is F10, F9, F8.

If the plan gets cut, cut from D.

---

## Verification

Per feature: `uv run pytest tests/infra/code_intel/<file> -q`.

Before any merge: `make pre-commit` (format, lint, `mypy --strict`, docs, tests).

Manual acceptance for the two that must not silently regress:

```bash
# F1 — must return the three real importers, not empty
lc mcp ... graph kind=blast_radius path=src/lemoncrow/core/service/telemetry/emit.py

# F4 — must succeed under the core profile
LEMONCROW_MCP_TOOL_PROFILE=core lc mcp ... tool action=call name=relations
```

A vendored-engine checkout additionally needs the segfault resolved before F2,
F5, or F9 can be exercised end to end: `get_symbol`, `find_references`,
`tool_callers`, `tool_callees`, and `tool_usages` all SIGSEGV (rc=139) inside the
compiled extension, consistent with an ABI mismatch. `call_graph_centrality`,
`tool_index`, and the file-graph analytics are unaffected — which is why F1 and
F3 can proceed immediately.

---

## Open questions

1. **Does F9's precision justify its cost?** Deliver F1-F3 first, measure how
   often name-matching actually misleads on real reviews, then decide. The
   builtin-dominated centrality suggests it matters; that is a hypothesis, not a
   measurement.
2. **Should the sidecar be per-repo or shared?** Per-repo (this plan) is simpler
   and matches the engine. Cross-repo edges would need a shared store.
3. ~~**Who rebuilds the sidecar, and when?**~~ **Settled: poll
   `engine_state.index_version`.** F11 has to build exactly that throttled
   version check to fix the cached-engine bug, and the sidecar can reuse it.
   Polling needs no engine cooperation, which the alternatives (piggybacking the
   edit-triggered reindex thread, or an autosync hook) both do.
4. **Upstreaming.** Several of these belong in the engine, not beside it. If
   private-repo access appears, F9 in particular should move into extraction
   rather than remain a post-hoc pass.
