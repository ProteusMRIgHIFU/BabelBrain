#!/usr/bin/env bash
#
# Assert that a built macOS PKG will actually RUN its postinstall.
#
#     verify_pkg_scripts.sh <path/to/installer.pkg>
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
