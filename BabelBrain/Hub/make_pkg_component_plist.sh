#!/usr/bin/env bash
#
# Generate the component property list that STOPS the macOS Installer from
# relocating our app bundles, then pass it to `pkgbuild --component-plist`.
#
#     make_pkg_component_plist.sh <staged_pkg_root> <out.plist>
#
# Why this exists
# ---------------
# `pkgbuild` marks every .app in the payload as *relocatable* by default. At
# install time the Installer then asks LaunchServices where a bundle with that
# CFBundleIdentifier already lives and installs THERE instead of at the payload
# path. For BabelBrain that is catastrophic: the version store is full of
# BabelBrain.app bundles, so /Applications/BabelBrain.app got redirected on top
# of a random version folder and never appeared in /Applications at all. (Seen
# for real: a 128 MB launcher overwrote the 1.1 GB version in
# /Users/Shared/BabelBrain/versions/<build_id>/.)
#
# Every path this PKG writes is deliberate, so no bundle may ever be relocated.
# verify_pkg_scripts.sh fails the build if this wiring is lost again.
#
set -euo pipefail

ROOT="${1:?usage: make_pkg_component_plist.sh <staged_pkg_root> <out.plist>}"
OUT="${2:?usage: make_pkg_component_plist.sh <staged_pkg_root> <out.plist>}"

PY="$(command -v python3 || echo /usr/bin/python3)"
[[ -x "$PY" ]] || { echo "error: python3 not found" >&2; exit 1; }

pkgbuild --analyze --root "$ROOT" "$OUT" >/dev/null

"$PY" - "$OUT" <<'PYEOF'
import plistlib
import sys

path = sys.argv[1]
with open(path, 'rb') as fh:
    components = plistlib.load(fh)

for component in components:
    # The only correct destination is the payload path.
    component['BundleIsRelocatable'] = False
    # Version checking would make the Installer SKIP a bundle whose destination
    # already holds a higher CFBundleShortVersionString — i.e. report success
    # and change nothing. Every path here is deliberate, so the payload always
    # wins; this also keeps downgrades (pinning an older release) working.
    component['BundleIsVersionChecked'] = False
    component['BundleOverwriteAction'] = 'upgrade'

with open(path, 'wb') as fh:
    plistlib.dump(components, fh)

print(f'{len(components)} bundle(s) pinned to their payload path:')
for component in components:
    print(f"  {component.get('RootRelativeBundlePath')}")
PYEOF
