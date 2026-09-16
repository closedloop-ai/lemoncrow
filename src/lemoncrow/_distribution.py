"""Which LemonCrow this build is distributed from.

Upstream ships no ``lemoncrow._distribution``, so the module's presence — and the
repository it names — is what identifies a fork build. ``lc update``'s release
path re-runs the installer published by :data:`UPSTREAM_REPO`; when that is not
the repository this build came from, applying it would silently replace the fork
build and the tools only it ships (PRD-739 FR11).

That absence is enforced, not assumed: ``scripts/public-paths.txt`` denies this
file from the public mirror, and ``lc update`` resolves to upstream's own
identity when the import fails.
"""

from __future__ import annotations

DISTRIBUTION_REPO = "closedloop-ai/lemoncrow"

#: The repository the published release channel (``scripts/install.sh``) serves.
UPSTREAM_REPO = "lemoncrow-lab/lemoncrow"


def is_fork_build() -> bool:
    """True when a release installs a different repository than this build came from.

    Lives here, not next to either caller: ``lc update`` and the servicectl
    auto-updater both guard on it, and a predicate copied into two modules is the
    drift the shared constants were pulled up here to end.
    """
    return DISTRIBUTION_REPO != UPSTREAM_REPO


def fork_update_command() -> str:
    """The command that updates a fork build, for an operator to paste.

    One cwd for both halves: ``git -C <clone> pull`` would leave ``bash
    scripts/local.sh`` resolving against wherever the operator is standing, and
    by construction they are not in the clone when this prints.
    """
    return f"cd <your {DISTRIBUTION_REPO} clone> && git pull && bash scripts/local.sh"
