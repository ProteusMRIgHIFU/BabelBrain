#!/usr/bin/env bash
#
# Assert that a built macOS PKG is wired correctly, before anyone installs it.
#
#     verify_pkg_scripts.sh <path/to/installer.pkg>
#
# Two checks, both for failures that are INVISIBLE in the build log and only
# show up on a user's machine:
#
#   1. the postinstall is attached to the component package (below);
#   2. no bundle is marked relocatable, and no two bundles share an identifier.
#
# The postinstall is what records the seeded build as the default version (see
# make_pkg_scripts.sh). It only runs if it is attached to the *component*
# package, which means that component's PackageInfo must carry a <scripts>
# element. A script placed at the distribution level instead — what
# `productbuild --root --scripts` produces — is embedded in the archive and
# silently never executed, so merely finding a `postinstall` file somewhere in
# the PKG proves nothing. This check exists because that is exactly how it
# shipped broken once.
#
set -euo pipefail

PKG="${1:?usage: verify_pkg_scripts.sh <installer.pkg>}"
[[ -f "$PKG" ]] || { echo "error: $PKG not found" >&2; exit 1; }

WORK="$(mktemp -d -t bbverify)"
trap 'rm -rf "$WORK"' EXIT
pkgutil --expand "$PKG" "$WORK/x" >/dev/null

FOUND=0
for INFO in "$WORK"/x/*.pkg/PackageInfo; do
  [[ -f "$INFO" ]] || continue
  if grep -q '<postinstall' "$INFO"; then
    FOUND=1
    echo ">> postinstall wired into $(basename "$(dirname "$INFO")")"
  fi
done

if [[ "$FOUND" -ne 1 ]]; then
  echo "error: $PKG has no postinstall declared in any component PackageInfo." >&2
  echo "       The seeded version will install but will NOT become the default." >&2
  echo "       Build the component with 'pkgbuild --scripts', then wrap it with" >&2
  echo "       'productbuild --package' — productbuild --root --scripts does not" >&2
  echo "       attach scripts to the component." >&2
  exit 1
fi
echo ">> PKG script wiring OK"

# --------------------------------------------------------------------------
# No bundle may be relocatable, and identifiers must be unique.
#
# A relocatable bundle is installed wherever LaunchServices already knows one
# with the same CFBundleIdentifier, NOT at its payload path. BabelBrain ships
# several BabelBrain.app bundles (the launcher plus one per version in the
# store), so relocation silently drops /Applications/BabelBrain.app on top of a
# version folder and the user is left without a main app. Fix by building the
# component with 'pkgbuild --component-plist' from
# Hub/make_pkg_component_plist.sh, and by giving each app its own identifier.
# --------------------------------------------------------------------------
BAD=0
for INFO in "$WORK"/x/*.pkg/PackageInfo; do
  [[ -f "$INFO" ]] || continue

  RELOCATED="$(/usr/bin/sed -n '/<relocate>/,/<\/relocate>/p' "$INFO" \
               | /usr/bin/grep -o 'id="[^"]*"' | /usr/bin/sed 's/id="//;s/"//' || true)"
  if [[ -n "$RELOCATED" ]]; then
    echo "error: $(basename "$(dirname "$INFO")") marks these bundles relocatable:" >&2
    echo "$RELOCATED" | /usr/bin/sed 's/^/         /' >&2
    echo "       They will be installed wherever LaunchServices already has a" >&2
    echo "       bundle with that id, not at their payload path." >&2
    echo "       Build with: pkgbuild --component-plist <Hub/make_pkg_component_plist.sh output>" >&2
    BAD=1
  fi

  DUPES="$(/usr/bin/grep -o '<bundle id="[^"]*"[^>]*path=' "$INFO" \
           | /usr/bin/sed 's/<bundle id="//;s/".*//' | /usr/bin/sort | /usr/bin/uniq -d || true)"
  if [[ -n "$DUPES" ]]; then
    echo "error: $(basename "$(dirname "$INFO")") ships two payload bundles with" >&2
    echo "       the same identifier:" >&2
    echo "$DUPES" | /usr/bin/sed 's/^/         /' >&2
    echo "       Give each app its own bundle_identifier in its .spec file." >&2
    BAD=1
  fi
done
[[ "$BAD" -eq 0 ]] || exit 1
echo ">> PKG bundle placement OK (nothing relocatable, identifiers unique)"
