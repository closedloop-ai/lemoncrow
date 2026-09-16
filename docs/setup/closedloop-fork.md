# The ClosedLoop fork build

`closedloop-ai/lemoncrow` is a fork of `lemoncrow-lab/lemoncrow`. It carries the
code-intelligence tools review agents depend on — `code_query`, `code_changes`
and `code_coverage_check` — which upstream does not ship. An upstream release
has none of them, so an operator who installs or updates from one loses the
tools and reviews quietly fall back to `grep`.

The fork publishes no releases of its own, so a clone is the only install path
that produces the fork build.

## Install from a fork clone

```bash
git clone https://github.com/closedloop-ai/lemoncrow.git
cd lemoncrow
bash scripts/local.sh
```

`scripts/local.sh` installs the checkout as a uv tool: a non-editable snapshot
copy of the clone in its own venv, wired into your hosts. A source edit does not
reach the running `lc` until you re-run the script, which is deliberate — a
half-finished edit can never break the tool your session is using.

## Confirm you are on the fork build

```bash
lc update --check --json
```

`distribution` names the repository the build came from; on the fork build it is
`closedloop-ai/lemoncrow`. (Both checks need network: the command asks GitHub
for the latest release.)

Then list LemonCrow's tools in your host — in Claude Code, `/mcp`, then
`lemoncrow`. The fork build advertises `code_query`, `code_changes`,
`code_coverage_check` and `relations` under both server profiles. A tool list
that has `code_search` and `read` but no `code_query` is an upstream build.

## Updating

From the clone:

```bash
git pull && bash scripts/local.sh
```

`lc update` does the same thing on a git checkout: it pulls the clone's own
`origin` — the fork — and re-syncs. That path is unchanged.

On a **release** install, `lc update` re-runs upstream's published `install.sh`,
which would replace the fork build. It now refuses first:

```text
  ! Release updates come from lemoncrow-lab/lemoncrow (upstream), not closedloop-ai/lemoncrow.
```

and points at the fork command above. Answering yes at the prompt, or passing
`--allow-upstream`, applies it anyway; nothing switches you to upstream without
one of those.

The background service can also auto-update, and its release path is off unless
you set `LEMONCROW_AUTO_UPDATE_RELEASE=1`. A `scripts/local.sh` install skips
auto-update altogether, because that script marks the install as a dev install.

## Reinstalling after an upstream merge

Merging an upstream release into the fork moves dependencies and the pinned
companion binaries, and the installed `lc` is a copy rather than a link, so
re-run the installer instead of relying on the running snapshot:

```bash
git pull
bash scripts/local.sh
```

Then reconnect the MCP server in your host so it picks up the new tool list.

## Why not `install.sh`

The published installer — the `curl … releases/latest/download/install.sh | bash`
line in [Installation](./installation.md) — downloads a pre-compiled release
asset built from `lemoncrow-lab/lemoncrow`. It is the right path for upstream's
build and the wrong one here: it produces an upstream build, with no
`code_query`, `code_changes` or `code_coverage_check`, whatever it replaced.
