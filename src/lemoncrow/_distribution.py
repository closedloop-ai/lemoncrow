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
