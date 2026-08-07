# Phase B handoff — for the code-review plugin session

You held your scoping doc pending these. They have landed and are **installed into
the production tool**, and your workspace's daemon was restarted, so you are
already talking to the new code.

- PR: https://github.com/closedloop-ai/lemoncrow/pull/2 (open, not merged)
- Installed: `~/.lemoncrow/uv-tools/lemoncrow/.../lemoncrow/` (9 files, plain `.py`)
- Your daemon: pid 69767 (`~/Dev/claude-plugins`, 2h22m old) was killed; it has
  respawned on current code.

---

## Your three points, and what happened to each

### 1. Completeness vs relevance — **addressed**

> "the 6 test callsites are never enumerated with anchors … impact analysis needs
> the opposite objective — every callsite, exhaustively, or an explicit 'this list
> is complete' signal. That's the one thing that would let the plugin grant
> graph-sourced findings the grep-replay exemption."

Two things were wrong, and only one of them was the one we thought.

**The enumerative tool existed but was invisible.** `relations` sat in both
`_CORE_MCP_TOOLS` and `HIDDEN_LLM_TOOLS`, so no profile ever advertised it. Under
the `core` profile an agent saw exactly one code-intel tool — `code_search`, the
ranked one — and zero enumerative ones. You were not misusing the surface; the
right tool was unreachable.

```
before  core (5): bash, code_search, edit, read, web_fetch  (+ tool)
        relations advertised: False under BOTH profiles
after   core (9): bash, code_changes, code_coverage_check, code_query,
                  code_search, edit, read, relations, web_fetch  (+ tool = 10)
        full (9): the same nine, minus `tool` (the broker is redundant there)
```

**The completeness signal is now on the response, not implied by the tool name.**
Every retrieval response carries:

| field | meaning |
|---|---|
| `objective` | `"ranked"` or `"exhaustive"` |
| `<subject>_count` / `scanned` / `reference_count` | what was found **before** any limit |
| `truncated` | whether the returned list is shorter than that |

**Your exemption predicate is therefore:**

```
objective == "exhaustive" and truncated == false
```

Evaluate that, not the tool name. Current taxonomy — one ranked tool, everything
else enumerates:

- `ranked`: `code_search`, `explore`, `context`, `graph kind=centrality`
- `exhaustive`: `relations` (`callers`/`callees`/`usages`), `code_changes`,
  `code_query`, `code_coverage_check`, all `graph` file-analytics
- **absent**: `pattern`, `node`. Absent means *unclassified*, never "exhaustive" —
  their extraction is in the closed engine and we have no oracle for it. Do not
  grant the exemption on a missing field.

### 2. Transient empty results during reindex — **addressed**

> "A reviewer reads 'no matches' as 'this symbol has no callers' and files a wrong
> finding. Failing loud (stale-index error) beats failing empty."

Root cause was worse than transient. `_code_engine_cache` was populate-once and
**never invalidated in production** — your daemon built one engine at
`index_version 1` and was still serving it at version 23, returning empty for
every query with no error. That is why it looked transient: it was permanent, and
only a restart cleared it.

Now:

- cache entries are stamped with the generation they were built at and rebuilt on
  a mismatch (re-read throttled to 5s, so a hot path does not pay a SQLite open
  per call — staleness is bounded at 5s instead of unbounded)
- an index caught **mid-write raises `IndexRebuilding`** rather than returning `[]`
- a response served by a rebuilt engine carries `index_state: "rebuilt"`, so
  results that differ from your previous call have a stated cause rather than
  looking like nondeterminism

### 3. NL-vs-keyword phrasing sensitivity — **NOT addressed**

> "the keyword form ranked the caller first, the NL form returned the definition
> first."

Deliberately not fixed, and I do not think it should be. It is a ranking-quality
problem in the *ranked* tool, and the fix for your use case is routing, not
ranking: for "who calls X", use `relations` or `code_query`, which do not rank at
all and are now visible. For reference, `code-review-graph` measures its own
keyword search at MRR 0.35 and says ranking needs work — this is category-wide,
not a LemonCrow defect.

If you still want ranked search to handle the NL form better, file it separately;
it should not block the exemption.

---

## How to verify (in your workspace)

> **The profile is the daemon's, not yours.** `LEMONCROW_MCP_TOOL_PROFILE` is
> read from the environment of the **server** process. A long-lived daemon
> started without it serves `core` to every client no matter what the caller
> exports, so exporting `full` and re-listing measures the same core daemon
> twice — which is exactly why the two profiles came back byte-identical. An
> earlier draft of this document reported `full (9)` as though a client could
> select it; it cannot. All four code-intel tools are now advertised under
> **both** profiles, so the distinction no longer decides what a reviewer can
> reach.

**A. The enumerative tool is visible**

```
tools/list  →  expect `relations`, `code_coverage_check`, `code_changes` and
               `code_query` present under both core and full profiles
```

**B. Exhaustive enumeration matches a callsite oracle** — the check you wanted

```
relations   symbol=merge_telemetry kind=usages
code_query  select=callers where={"callee":"merge_telemetry"} limit=50
```

Expect on both: `objective: "exhaustive"`, `truncated: false`, `match_kind:
"name"`, and a count equal to the **actual call sites** — the lines that invoke
the symbol. Anchors include file, line, column range, and enclosing caller.

> **The oracle is call sites, not `grep -c`.** An earlier draft of this document
> said "a count equal to `grep -rn 'merge_telemetry'` minus the definition
> line". That is wrong and reads as a failure: on this repo grep returns 21 —
> imports, docstring mentions, and six test *function names* that contain the
> substring — while `relations` returns the 7 real invocations. The extraction
> is AST-level, not textual, so it is strictly more precise than the grep it was
> being measured against. Count invocations by hand, or `grep -n '\bmerge_telemetry('`.

Note what `match_kind: "name"` is telling you, because exhaustive and
untruncated is **necessary, not sufficient**. Both edge stores are keyed by
name — `call_edges` holds raw dotted callee text, `"references"` a bare
`symbol_name`, and neither table has a column a row could be resolved *by*. So
the error is one-directional: over-reports, never under-reports. Safe for an
exemption; not safe for output shown to a human, where a change to a method
called `open` would enumerate every `open()` in the repo — real lines, which is
why replaying them through grep confirms them rather than catching them.

The field is stamped at the top level because it describes every row: no two
rows can currently disagree. When F9's resolution sidecar lands, `match_kind`
moves onto the row and the top-level value becomes `"mixed"` — a consumer
already gating on `== "resolved"` stays correct across that change unedited.

> **Known gap: `relations kind=callers|callees` has no `truncated`.** Measured
> on the live index, not inferred. Those two ops return `objective`,
> `match_kind`, `related_count` and `edge_count`, but no `truncated`, and
> `related_count` is counted *after* the cap — `callees` on `merge_telemetry`
> reports `related_count: 12` at `limit=2` and `41` at `limit=200`. There is no
> pre-limit number in that payload, so the predicate cannot be evaluated. It
> fails safe (absent `truncated` never equals `false`, so a gating consumer
> validates by hand) but it is unusable as written, and the truncation truth
> lives in the closed engine where this branch cannot compute it without
> re-deriving the traversal and risking a count that disagrees with the list
> shipped beside it.
>
> `kind=usages` is unaffected — it stamps `reference_count` and `truncated`
> correctly. **For caller enumeration under a completeness gate, use
> `code_query select=callers where={"callee": "<symbol>"}` instead**: it is
> open-source end to end and returns all four fields — `objective`, `count`,
> `truncated`, `match_kind` — plus `scanned` and `scan_capped` bounding the
> claim. Closing the gap on `relations` itself is engine work; it belongs with
> F9 in Phase C.
code_coverage_check paths=["<file you searched>"]
```

States per path: `indexed` / `stale` / `missing` / `excluded` / `unparsed`, plus
the `engine_index_version` it was judged against. Use this before filing any
finding that rests on "no matches".

**D. Impact analysis on a diff**

```
code_changes base_ref=main depth=1
```

Returns changed symbols with `exported`, `callers`, `test_callers`, `risk`, and
the impacted callsites with `impacted_total` (pre-truncation) and `truncated`.

**E. Staleness is no longer silent**

Cheapest check: confirm your daemon is newer than the install, and that a query
you know has hits returns them. If you want to exercise the loud path, hold the
index-write lock (`.lemoncrow/workspace/code_context.sqlite.indexlock`, `flock`
exclusive) and re-query — expect an `IndexRebuilding` error, not `[]`.

---

## Limits you should NOT re-report as defects

1. **`match_kind: "name"`.** `call_edges` stores the callee as raw dotted call
   text with no `symbol_id`, so every reverse lookup is name-matched. It
   **over-reports and never misses** — a change to a method named `open` collects
   every `open()` in the repo. For the grep-replay exemption this is the safe
   direction (no false negatives), but expect false positives on common names.
   Resolution is F9, Phase D, not built.

2. **`depth > 1` on `code_changes`.** The graph is name-keyed, so imprecision
   compounds per hop. Read depth≥2 as "possibly related", not "impacted".

3. **`blast_radius` is now bounded** (default `limit=100`). It used to dump every
   importer inline — 390 paths, 18KB for one file. Counts and `risk_level` come
   from the full closure, so truncation costs detail and never the answer to "how
   big is this change". Raise `limit` if you want the full list.

4. **Torn-index detection is one-directional.** Symbols with no files reads as
   rebuilding; files with no symbols does **not** — that is a docs-only repo, or
   any workspace without the optional tree-sitter `parsers` extra, and treating it
   as a rebuild made every code tool fail forever on a perfectly valid workspace.

5. **`zstandard` is not installed**, so `lc code export` uses an lzma fallback.
   Functional, larger archives. Not a bug.

---

## Also worth knowing

Your graph workers already wire `codebase-memory-mcp`. Running a review graph
alongside a structural backend is the normal pattern in this space and the tools
do not gatekeep each other — but it does mean some of the completeness complaint
may have been aimed at cbm rather than at LemonCrow. Worth checking which surface
you were actually querying when you measured the 7-vs-1 callsite gap.

`code_query` was scoped specifically to cover what cbm's `query_graph` gets used
for — "functions matching X with no callers", "files importing Y ordered by
centrality" — so a head-to-head is now possible in-house.
