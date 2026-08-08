# PLN-1633 Phase C — review handoff

Branch `worktree-pln-1633-phase-c`, 3 commits on top of `origin/main`
(`8d77d599`), unpushed. 8 files, +1510/−33.

| Commit | What |
|---|---|
| `461ec349` | F6 — clone detection: `clones.py`, sidecar migration 2, `code_query(select="clones")`, `lc code clones` |
| `5de4394d` | Plan updated: what F6 shipped, and why F10 was not built |
| `ad4fb277` | Resolution of code review cr-67523 — 7 findings fixed, 1 declined |

**One feature shipped: F6.** F10 (open semantic search) was **not built** — see
§5. Phase C is therefore not closed.

---

## 1. What is new to call

```
lc code clones [--threshold 0.8] [--min-tokens 40] [--limit 20] [--json]
code_query(select="clones", where={...}, limit=50)      # MCP tool, new select
```

`lc code clones` is a full pass over the repository — 12,223 symbols in ~33s
here. It is a command rather than something a tool call triggers implicitly,
and `code_query` never builds implicitly either.

The six existing selects (`symbols`, `callers`, `callees`, `importers`,
`references`) are unchanged. `clones` is the first backed by the **sidecar**
rather than the engine, which is the only structural difference: it is derived
data we produce, so it can be *absent* rather than merely empty.

## 2. How it decides two symbols are clones

Source is sliced from each symbol's `start_byte`/`end_byte`, tokenised,
shingled into runs of 5, MinHashed at 128 permutations, and compared under LSH
banding (16 bands × 8 rows). Candidates are then **scored individually**, so
banding costs recall and never precision — a reported pair was always measured.

Two normalisation choices decide almost everything about the output, and both
were forced by measurement rather than chosen:

- **Identifiers and numeric literals are placeholdered; keywords survive.**
  With verbatim tokens a copied function whose locals were all renamed scored
  **0.039** — at k=5 nearly every shingle holds at least one renamed identifier.
- **String literals are kept; only docstrings are stripped.** Dropping strings
  produced 2,968 pairs whose top scorers were every `to_dict` matching every
  other at 1.000, because once names are gone a `to_dict`'s dict keys are all
  that distinguish it.

So this is a **Type-2 clone detector**: copy-paste, with or without renaming.
It is not semantic — two different implementations of one idea score low. That
was F10's territory.

## 3. How to verify

**A. It runs and reports its own coverage**

```
lc code clones --limit 12
```

Expect a pair count, plus `symbols compared` / `skipped as too short` /
`unreadable` / `candidate pairs from banding`. Roughly half of all symbols are
skipped as shorter than `MIN_TOKENS = 40`; that is intended, not a failure.

> **Numbers drift between runs, and that is expected.** The engine reindexes on
> its own, so `index_version` and the symbol count move. Across this session's
> runs: 11,488 → 12,223 symbols compared, 1,909 → 1,864 pairs. Compare shapes,
> not exact totals.

**B. The score ordering is the default**

```
code_query select=clones limit=8
```

Expect `jaccard` descending. This was a real bug fixed in `ad4fb277` — the
select ordered by name, so the default `limit=50` over ~1,900 pairs returned
the alphabetically-first and reported `truncated: true`, making the score
unreachable.

**C. Freshness is per pair, and only "never built" refuses**

```
code_query select=clones          # on a workspace where lc code clones never ran
```

Expect `ClonesStale`, not an empty list: *"the clone table has never been
built; run `lc code clones` to build it"*. That is the one refusal left, and
it exists because an empty list would read as "this code has no duplicates"
when the truth is nobody looked.

> **This changed after the first handoff draft, and the change matters.**
> Freshness was originally keyed on `engine_index_version` — a global counter
> any file's reindex bumps — so one unrelated edit invalidated an answer still
> correct for every symbol it described. Measured live: the engine moved
> 244→260, 274→284, 284→285 inside one session and the table refused within
> minutes of each build. Each pair now carries the two `symbols.content_hash`
> values it was measured on, so a reader returns the subset that still holds.

Every response therefore carries coverage:

```
coverage: 1.0                 # fraction of live symbols the build examined
stale_symbols: 0              # examined-count complement
superseded_rows: 0            # pairs dropped because their source moved
built_from_index_version: 327
engine_index_version: 327
```

**Read `coverage` before reading an absence.** At 1.0, "no pair reported for X"
means measured-and-none. Below it, some symbols changed after the last build
and were never examined, so absence proves nothing. Verification happens
*before* `limit` is applied, so a stale top-scoring pair never eats a result
slot.

`ClonesStale` propagates as an ordinary tool failure (`isError`), **not** as a
JSON-RPC `-32602` argument fault — deliberately matching `IndexRebuilding`.

**C2. Rebuilds are incremental**

```
lc code clones      # twice in a row
```

Cold: `signatures reused: 0 | computed: 11,886 | symbols read from disk: 24,577`
in **34.4s**. Warm: `reused: 11,886 | computed: 0 | symbols read: 0` in
**1.07s**, byte-identical output. Signatures are cached against the content
hash they were taken over, and symbols too short to sign are recorded as
examined-but-unsigned so they are neither re-read nor counted against coverage.

`build_clones` now refuses a torn index (`IndexRebuilding`) — F11 built that
probe, and clone detection was reading around it.

**D. The precision properties, on real output**

```
code_query select=clones where={"file_path_a_like":"src/%","file_path_b_like":"src/%"} limit=8
```

Current top pairs are all genuine duplication —
`CursorAdapter.prime_context` ↔ `HermesAdapter.prime_context`,
`DiGraph.add_edge` ↔ `Graph.add_edge`,
`SymbolIntelProvider.find_callers` ↔ `find_callees`. Two things that should
**not** appear, both regressions fixed in `ad4fb277`:

- a symbol paired with its own parent (`X` ↔ `X.method`) — byte-range
  containment within a file is now skipped before scoring
- every `to_dict`-shaped method matching every other

**E. Comment markers inside string literals**

```python
normalise_tokens('@click.option("--limit", type=int, help="Pairs to print.")')
```

Expect `"--limit"` and `"Pairs to print."` present as whole tokens. Before
`ad4fb277` this returned six tokens — `['@', ID, '.', ID, '(', '"']` — because
the comment regexes ran over raw text before tokenising and fired inside
literals, truncating every click option and every URL in the repo.

## 4. Known limits — stated, not hidden

- **The table is only as fresh as the last `lc code clones`.** No background
  rebuild exists. Symbols changed since then are simply not covered, and
  `coverage` says by how much — never silently served as complete.
- **Raw pair counts are misleading.** ~1,860 pairs repo-wide, but most are
  outside `src/` and intentional: duplicated benchmark fixtures under
  `benchmarks/codebench/cg_tasks/`, the generated per-host `SKILL.md` set, docs
  install pages. Filter before drawing conclusions; the signal is in the
  filtered query.
- **A worktree inside the repo gets indexed.** `.claude/worktrees/…` copies of
  a file pair against their originals at 1.000. That is an engine ignore-rule
  gap, not a detector fault.
- **Markdown headings are symbols**, so docs sections participate. Useful, but
  surprising if unexpected.
- **`--` is not treated as a comment marker.** It serves SQL/Lua/Haskell and is
  a decrement operator in the C-family languages that dominate here, so SQL
  comments survive as tokens. Deliberate: mild noise beats truncated code.
- **Structurally identical but unrelated code still matches.** `ABRow` ↔
  `PairwiseQualityResult` (two same-shaped dataclasses) is the standing
  false-positive class. `MIN_TOKENS` and the 0.8 threshold hold it down; the
  reported score is what lets a reader judge.

## 5. What was not built, and why

**F10 (open semantic search) was descoped by the requester**, not deferred for
time. Every step after its backfill needs an embedder:
`LEMONCROW_CODE_EMBEDDER` is unset, `factory.py` returns `NullEmbedder`, and
there is no `ollama` binary — so `code_semantic_search` would have advertised a
tool returning nothing, verifiable only against a fake embedder in tests. Two
facts for whoever picks it up: `numpy` is **not** a base dependency (only in
the `vector` and `semantic` extras), so the plan's "brute-force cosine in
numpy" cannot be written against a base install; and the choice of backend
(`ollama`, no key or torch — vs the `semantic` extra, ~2GB of torch) is the
decision that unblocks it, not engineering time.

## 6. cr-67523, already resolved

A prior review of `461ec349` returned `NEEDS_ATTENTION` with 9 verified
findings, 0 rejected, 0 pending, 0 coverage gaps. `ad4fb277` fixed 7 and
declined 1 (`design_critic_f0`, adding `stamp_table` to `_Select` for OCP —
its payoff needs a second sidecar-backed select, which is F10). Re-review
should start from `ad4fb277`, not `461ec349`.

Two of those findings were correctness bugs that defeated F6's own stated
invariants, and both had already been reported to the user as working. Worth
weighting: the module's docstrings assert measured properties, and **the
docstrings have been wrong before** — the banding recall figures in `clones.py`
said ~92% at s=0.8 and ~9% at s=0.5 when the real values are 94.7% and 6.1%.
That is now pinned by
`test_banding_recall_curve_matches_the_documented_probabilities`. Treat any
remaining stated number as a claim to check, not a given.

## 7. Verification already run

```
tests/infra/code_intel/test_clones.py            53 passed
focused suite incl. gateway + mypyc-safety      372 passed
mypy --strict src                               257 errors — the pre-existing
                                                baseline exactly, none in new code
ruff (touched files)                            All checks passed
git diff --check                                clean
real run                                        12,223 symbols, 3,551 candidates,
                                                1,864 pairs, ~33s
```

Not run: `make pre-commit` in full. `tests/gateway/test_mcp_jsonrpc_e2e.py` and
`tests/gateway/test_p0_mcp_surfaces.py` **SIGSEGV** in this worktree — verified
pre-existing by reverting to `HEAD` and reproducing the identical crash; it is
the compiled-pro/source-open ABI mismatch, not this branch. Those two files
therefore verify nothing here.
