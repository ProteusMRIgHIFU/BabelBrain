#!/usr/bin/env bash
#
# Repackage a legacy macOS release DMG into a relocatable "-version.zip" that the
# BabelBrain Hub can download and install, WITHOUT rebuilding from source.
#
# Old releases shipped a BabelBrain.app inside a DMG (either drag-install, i.e.
# the .app sits at the DMG root, or PKG-in-DMG for newer ones). This script:
#   1. downloads the DMG from the GitHub release,
#   2. mounts it and extracts BabelBrain.app (unpacking the PKG if needed),
#   3. zips the app with `ditto --keepParent` (so the archive's top entry is
#      BabelBrain.app, which is what the Hub expects),
#   4. computes the sha256,
#   5. optionally uploads the zip back to the release, and
#   6. prints the releases.json snippet to paste on `main`.
#
# The app is NOT modified, so its existing code signature / notarization stay
# intact — the Hub synthesizes build_info.json next to the app at install time.
#
# Requires: gh (authenticated), hdiutil, ditto, shasum, python3. macOS only.
#
# Example:
#   Hub/repackage_legacy_asset.sh \
#       --tag 0.4.1 --dmg BabelBrain_ARM64.dmg --platform-key macos-arm64 \
#       --commit 795e3b5 --upload
#
set -euo pipefail

REPO="ProteusMRIgHIFU/BabelBrain"
TAG=""
DMG=""
PLATFORM_KEY=""       # macos-arm64 | macos-x64  (the releases.json asset key)
VERSION=""            # defaults to TAG with a leading 'v' stripped
COMMIT=""
CHANNEL="stable"
UPLOAD="no"
WORKDIR=""

usage() {
  sed -n '2,30p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)         REPO="$2"; shift 2;;
    --tag)          TAG="$2"; shift 2;;
    --dmg)          DMG="$2"; shift 2;;
    --platform-key) PLATFORM_KEY="$2"; shift 2;;
    --version)      VERSION="$2"; shift 2;;
    --commit)       COMMIT="$2"; shift 2;;
    --channel)      CHANNEL="$2"; shift 2;;
    --upload)       UPLOAD="yes"; shift;;
    -h|--help)      usage 0;;
    *) echo "Unknown argument: $1" >&2; usage 1;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "This script must run on macOS." >&2; exit 1; }
[[ -n "$TAG" && -n "$DMG" && -n "$PLATFORM_KEY" ]] || \
  { echo "error: --tag, --dmg and --platform-key are required." >&2; usage 1; }

case "$PLATFORM_KEY" in
  macos-arm64) ASSET_BASE="BabelBrain-macOS-arm64";;
  macos-x64)   ASSET_BASE="BabelBrain-macOS-x64";;
  *) echo "error: --platform-key must be macos-arm64 or macos-x64." >&2; exit 1;;
esac

[[ -n "$VERSION" ]] || VERSION="${TAG#v}"          # strip a leading 'v' if present
OUT_ZIP="${ASSET_BASE}-version.zip"
URL="https://github.com/${REPO}/releases/download/${TAG}/${OUT_ZIP}"

WORKDIR="$(mktemp -d -t bbrepack)"
MOUNT=""
cleanup() {
  [[ -n "$MOUNT" ]] && hdiutil detach "$MOUNT" -quiet 2>/dev/null || true
  [[ -n "$WORKDIR" ]] && rm -rf "$WORKDIR" 2>/dev/null || true
}
trap cleanup EXIT

echo ">> Downloading $DMG from $REPO@$TAG"
gh release download "$TAG" --repo "$REPO" --pattern "$DMG" --dir "$WORKDIR"

echo ">> Mounting DMG"
MOUNT="$WORKDIR/mnt"
mkdir -p "$MOUNT"
hdiutil attach "$WORKDIR/$DMG" -nobrowse -readonly -mountpoint "$MOUNT" -quiet

APP=""
# Case 1: drag-install DMG — BabelBrain.app is on the mounted volume.
APP="$(find "$MOUNT" -maxdepth 2 -name 'BabelBrain.app' -type d 2>/dev/null | head -n1 || true)"

STAGED_APP="$WORKDIR/BabelBrain.app"
if [[ -n "$APP" ]]; then
  echo ">> Found app on DMG (drag-install layout)"
  ditto "$APP" "$STAGED_APP"
else
  # Case 2: PKG-in-DMG — expand the installer package to reach the payload.
  PKG="$(find "$MOUNT" -maxdepth 2 -name '*.pkg' 2>/dev/null | head -n1 || true)"
  [[ -n "$PKG" ]] || { echo "error: no BabelBrain.app or .pkg found in $DMG" >&2; exit 1; }
  echo ">> No bare app; expanding PKG: $(basename "$PKG")"
  EXP="$WORKDIR/pkg_expanded"
  pkgutil --expand-full "$PKG" "$EXP"
  PKG_APP="$(find "$EXP" -name 'BabelBrain.app' -type d 2>/dev/null | head -n1 || true)"
  [[ -n "$PKG_APP" ]] || { echo "error: BabelBrain.app not found inside the PKG payload." >&2; exit 1; }
  ditto "$PKG_APP" "$STAGED_APP"
fi

hdiutil detach "$MOUNT" -quiet; MOUNT=""

echo ">> Creating $OUT_ZIP"
ditto -c -k --keepParent "$STAGED_APP" "$WORKDIR/$OUT_ZIP"
SHA="$(shasum -a 256 "$WORKDIR/$OUT_ZIP" | awk '{print $1}')"
# Keep the finished zip next to where the script was invoked.
cp "$WORKDIR/$OUT_ZIP" "./$OUT_ZIP"
echo ">> Wrote ./$OUT_ZIP"
echo ">> sha256: $SHA"

if [[ "$UPLOAD" == "yes" ]]; then
  echo ">> Uploading to release $TAG"
  gh release upload "$TAG" --repo "$REPO" "./$OUT_ZIP" --clobber
fi

echo ""
echo "=================== releases.json asset (paste on main) ==================="
echo "For \"$VERSION\" (build id ${VERSION}${COMMIT:++${COMMIT}}), platform \"$PLATFORM_KEY\":"
python3 "$(dirname "$0")/gen_manifest_entry.py" \
  --version "$VERSION" ${COMMIT:+--commit "$COMMIT"} --channel "$CHANNEL" \
  --asset "${PLATFORM_KEY}=${URL}:${SHA}"
echo "=========================================================================="
echo "Note: this snippet lists only the $PLATFORM_KEY asset. Run the script once"
echo "per platform (arm64 + x64) and merge the assets into a single version entry,"
echo "or edit the assets block in releases.json directly."
