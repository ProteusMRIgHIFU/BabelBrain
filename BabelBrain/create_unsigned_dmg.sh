#!/usr/bin/env bash
#
# Build the BabelBrain Hub installer DMG locally, UNSIGNED — a fast alternative
# to waiting for the GitHub Actions release build. It mirrors the macOS steps in
# .github/workflows/build-release.yml (build the version -> stamp build_info ->
# build the Hub -> nest the version inside the Hub -> make a DMG), just without
# code signing / notarization and without the PKG (a PKG is only useful signed).
#
# The resulting DMG is a drag-install containing BabelBrain.app (the Hub, with a
# bundled BabelBrain version inside). Since it is built locally (no quarantine
# attribute), it launches without a Gatekeeper prompt.
#
# Run from anywhere with the babelbrain conda env active:
#     ./create_unsigned_dmg.sh                 # full build
#     ./create_unsigned_dmg.sh --skip-version-build   # reuse dist/version, rebuild Hub+DMG only
#
# For the very fastest inner loop you can also skip the DMG entirely and just run
#     dist/hub/BabelBrain.app/Contents/MacOS/BabelBrain
#
set -euo pipefail

# Resolve our own path BEFORE changing directory, then operate from the
# BabelBrain/ directory (where the specs live).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$SCRIPT_DIR"

SKIP_VERSION_BUILD="no"
for arg in "$@"; do
  case "$arg" in
    --skip-version-build) SKIP_VERSION_BUILD="yes";;
    -h|--help) sed -n '2,30p' "$SCRIPT_PATH"; exit 0;;
    *) echo "Unknown argument: $arg" >&2; exit 1;;
  esac
done

command -v pyinstaller >/dev/null || { echo "pyinstaller not found — activate the babelbrain conda env first." >&2; exit 1; }

case "$(uname -m)" in
  arm64)  ARCHKEY="arm64";;
  x86_64) ARCHKEY="x64";;
  *)      ARCHKEY="$(uname -m)";;
esac
DMG="BabelBrain-macOS-${ARCHKEY}-unsigned.dmg"

VERSION_APP="dist/version/BabelBrain.app"
HUB_APP="dist/hub/BabelBrain.app"

# ---------------------------------------------------------------------------
# 1. Build the BabelBrain version (the heavy part) + stamp build_info.json.
# ---------------------------------------------------------------------------
if [[ "$SKIP_VERSION_BUILD" == "yes" && -d "$VERSION_APP" ]]; then
  echo ">> Reusing existing $VERSION_APP (--skip-version-build)"
else
  echo ">> Generating build_info.json (channel=dev)"
  python Hub/gen_build_info.py --channel dev --out build_info.json
  cat build_info.json
  echo ">> Building BabelBrain version with PyInstaller"
  pyinstaller BabelBrain.spec --noconfirm --clean \
    --distpath dist/version --workpath build/version
  cp build_info.json "$VERSION_APP/Contents/Resources/build_info.json"
fi
[[ -d "$VERSION_APP" ]] || { echo "error: $VERSION_APP missing." >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. Build the Hub launcher (fast).
# ---------------------------------------------------------------------------
echo ">> Building Hub launcher with PyInstaller"
pyinstaller BabelBrainHub.spec --noconfirm --clean \
  --distpath dist/hub --workpath build/hub

# ---------------------------------------------------------------------------
# 3. Nest the version inside the Hub at the executable-relative bundled path.
# ---------------------------------------------------------------------------
echo ">> Nesting version into Hub"
DEST="$HUB_APP/Contents/Resources/bundled"
mkdir -p "$DEST"
rm -rf "$DEST/BabelBrain.app"
/usr/bin/ditto "$VERSION_APP" "$DEST/BabelBrain.app"

# ---------------------------------------------------------------------------
# 4. Build the (unsigned) drag-install DMG.
# ---------------------------------------------------------------------------
echo ">> Building DMG: $DMG"
[[ -f "$DMG" ]] && rm -f "$DMG"
STAGE="$(mktemp -d -t bbdmg)"
trap 'rm -rf "$STAGE"' EXIT
/usr/bin/ditto "$HUB_APP" "$STAGE/BabelBrain.app"

if command -v create-dmg >/dev/null; then
  create-dmg \
    --volname "BabelBrain Installer (unsigned)" \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 100 \
    --icon "BabelBrain.app" 200 190 \
    --app-drop-link 600 185 \
    "$DMG" \
    "$STAGE/" \
  || hdiutil create -volname "BabelBrain (unsigned)" -srcfolder "$STAGE" \
       -ov -format UDZO "$DMG"
else
  echo ">> create-dmg not found (brew install create-dmg) — using hdiutil"
  hdiutil create -volname "BabelBrain (unsigned)" -srcfolder "$STAGE" \
    -ov -format UDZO "$DMG"
fi

echo ""
echo "Done: $(pwd)/$DMG"
echo "Open it and drag BabelBrain.app to /Applications, or just run:"
echo "  dist/hub/BabelBrain.app/Contents/MacOS/BabelBrain"
