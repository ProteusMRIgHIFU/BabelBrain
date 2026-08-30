#!/usr/bin/env bash
#
# Generate the macOS PKG's installer scripts into a directory that is then
# passed to `productbuild --scripts`.
#
#     make_pkg_scripts.sh <build_id> <scripts_dir>
#
# The only script is a `postinstall` that records the build just seeded into
# /Users/Shared/BabelBrain/versions. Without it the PKG adds a version to the
# store but the Hub keeps running whatever the user selected before, so a fresh
# install (in particular a new dev build) silently does not become the default.
# The Hub reads this marker on start and adopts it once per user — see
# Hub/state.py:adopt_installer_default.
#
set -euo pipefail

BUILD_ID="${1:?usage: make_pkg_scripts.sh <build_id> <scripts_dir>}"
SCRIPTS_DIR="${2:?usage: make_pkg_scripts.sh <build_id> <scripts_dir>}"

mkdir -p "$SCRIPTS_DIR"

# BUILD_ID is expanded now (the postinstall must be self-contained); everything
# else is escaped so it is evaluated at install time.
cat > "$SCRIPTS_DIR/postinstall" <<EOF
#!/bin/sh
# Written by BabelBrain/Hub/make_pkg_scripts.sh at build time.
# Tell the BabelBrain Hub which build this installer just seeded, so it becomes
# the default version instead of leaving the previous selection in place.
# Fixed path, not the installer's \$3 destination volume: the payload itself is
# staged at /Users/Shared and the Hub only ever looks there.
MARKER_DIR="/Users/Shared/BabelBrain"
MARKER="\$MARKER_DIR/default_build.json"

/bin/mkdir -p "\$MARKER_DIR" || exit 0
/bin/cat > "\$MARKER" <<JSON
{
  "build_id": "$BUILD_ID",
  "installed_at": "\$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "pkg"
}
JSON

# World-readable: every user of this machine reads the marker, and each adopts
# it at most once.
/bin/chmod 644 "\$MARKER" 2>/dev/null

# Never fail the install over the marker — a missing default is a nuisance, a
# failed install is not.
exit 0
EOF

chmod +x "$SCRIPTS_DIR/postinstall"
echo "postinstall written to $SCRIPTS_DIR (build_id: $BUILD_ID)"
