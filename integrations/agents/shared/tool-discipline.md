<!-- lc:section write -->

## Tool discipline

Always use LemonCrow for every file read, search, edit and shell command — every one, no exceptions. ONE `edit` call carries every hunk across every file, ONE `read` call every path and range you already need, independent calls go in ONE message — each round-trip skipped never re-bills the conversation — use lc: `bash`, `read`, `edit`, `code_search`.

- **No lc tools → stop.** lc tools absent or erroring on every call → refuse to proceed: never fall back to host tools, report "LemonCrow MCP not connected" and halt.
- ****Read what Need, not might-need.** Batching is free; a speculative `:full` is not. Region known → `path:Lx-Ly`.
- **Known path → straight to `read`; any repo search → `code_search`** — symbols, strings, regex, every term in one query. Inline source is already read.
- **`bash` = execution only.** Never `grep -r`/`rg`/`git grep`/`find`/`sed`/`cat`/`head` over the repo; shell search only logs, scratch, command output.
- Large output → a file, never prose.

<!-- lc:end -->

<!-- lc:section read-only -->

## Tool discipline

Always use LemonCrow for every file read and search — every one, no exceptions. ONE `read` call returns every path and range you already need, independent calls go in ONE message — each round-trip skipped never re-bills the conversation — use lc: `bash`, `read`, `code_search`.

- **No lc tools → stop.** lc tools absent or erroring on every call → refuse to proceed: never fall back to host tools, report "LemonCrow MCP not connected" and halt.
- **Read what Need, not might-need.** Batching is free; a speculative `:full` is not. Region known → `path:Lx-Ly`.
- **Read-only — `bash` never mutates.** Inspection/validation only: no redirects, `sed -i`, `tee`, or Git state changes.
- **Known path → straight to `read`, no `code_search`.** Task, error, or stack trace names the file → don't explore first; otherwise start with `code_search`. Never shell `sed`/`cat`/`head`/`tail`/grep to read, search, or recheck indexed results.

<!-- lc:end -->
