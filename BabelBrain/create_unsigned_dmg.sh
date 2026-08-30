#!/usr/bin/env bash
#
# Build the BabelBrain installer DMG locally, UNSIGNED — a fast alternative to
# waiting for the GitHub Actions release build. It mirrors the macOS steps in
# .github/workflows/build-release.yml, just without code signing / notarization.
#
# Two-app model: the PKG inside the DMG installs
#   * /Applications/BabelBrain.app                    (the launcher / main app)
#   * /Applications/BabelBrain-Version-Selector.app   (the picker)
# and seeds a default BabelBrain version into the shared versions store
#   * /Users/Shared/BabelBrain/versions/<build_id>/BabelBrain.app
# so the app works offline right after install, and records it as the default
# version (PKG postinstall -> /Users/Shared/BabelBrain/default_build.json).
#
# Run from anywhere with the babelbrain conda env active:
#     ./create_unsigned_dmg.sh                        # full build
#     ./create_unsigned_dmg.sh --skip-version-build   # reuse dist/version (fast Hub-only rebuild)
#
# Fastest inner loop (no DMG/PKG at all): run a built app directly, e.g.
#     dist/selector/BabelBrain-Version-Selector.app/Contents/MacOS/BabelBrain-Version-Selector
#     dist/launcher/BabelBrain.app/Contents/MacOS/BabelBrain
#
rm *.pkg
rm *.dmg
rm -rf dist
rm -rf build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$SCRIPT_DIR"

SKIP_VERSION_BUILD="no"
for arg in "$@"; do
  case "$arg" in
    --skip-version-build) SKIP_VERSION_BUILD="yes";;
    -h|--help) sed -n '2,32p' "$SCRIPT_PATH"; exit 0;;
    *) echo "Unknown argument: $arg" >&2; exit 1;;
  esac
done

# Clean previous outputs. This has to run AFTER argument parsing: it used to be
# the first thing in the script, which wiped dist/ before --skip-version-build
# was even read, silently making that flag a no-op that rebuilt everything.
rm -f ./*.pkg ./*.dmg
rm -rf dist/selector dist/launcher build/selector build/launcher
if [[ "$SKIP_VERSION_BUILD" != "yes" ]]; then
  rm -rf dist/version build/version
fi

command -v pyinstaller >/dev/null || { echo "pyinstaller not found — activate the babelbrain conda env first." >&2; exit 1; }

case "$(uname -m)" in
  arm64)  ARCHKEY="arm64";;
  x86_64) ARCHKEY="x64";;
  *)      ARCHKEY="$(uname -m)";;
esac
DMG="BabelBrain-macOS-${ARCHKEY}-unsigned.dmg"
PKG="BabelBrain-macOS-${ARCHKEY}.pkg"

VERSION_APP="dist/version/BabelBrain.app"
SELECTOR_APP="dist/selector/BabelBrain-Version-Selector.app"
LAUNCHER_APP="dist/launcher/BabelBrain.app"

# ---------------------------------------------------------------------------
# 1. Build the BabelBrain version (the heavy part) + stamp build_info.json.
# ---------------------------------------------------------------------------
if [[ "$SKIP_VERSION_BUILD" == "yes" && -d "$VERSION_APP" ]]; then
  echo ">> Reusing existing $VERSION_APP (--skip-version-build)"
else
  echo ">> Generating build_info.json (channel=dev)"
  python Hub/gen_build_info.py --channel dev --out build_info.json
  cat build_info.json
  echo ">> Building BabelBrain version"
  pyinstaller BabelBrain.spec --noconfirm --clean \
    --distpath dist/version --workpath build/version
  cp build_info.json "$VERSION_APP/Contents/Resources/build_info.json"
fi
[[ -d "$VERSION_APP" ]] || { echo "error: $VERSION_APP missing." >&2; exit 1; }

# build_id (version+shortcommit) names the version's folder in the store.
BUILD_INFO="$VERSION_APP/Contents/Resources/build_info.json"
BUILD_ID="$(python -c "import json;d=json.load(open('$BUILD_INFO'));c=(d.get('git_commit') or '');print(d['version']+('+'+c[:7] if c else ''))")"
echo ">> build_id: $BUILD_ID"

# ---------------------------------------------------------------------------
# 2. Build the two launcher apps (fast).
# ---------------------------------------------------------------------------
echo ">> Building Version Selector app"
pyinstaller BabelBrainHub.spec --noconfirm --clean \
  --distpath dist/selector --workpath build/selector
echo ">> Building BabelBrain launcher app"
pyinstaller BabelBrainLauncher.spec --noconfirm --clean \
  --distpath dist/launcher --workpath build/launcher

# ---------------------------------------------------------------------------
# 3. Stage the PKG payload: both apps in /Applications, the version seeded in
#    the shared store.
# ---------------------------------------------------------------------------
echo ">> Staging PKG payload"
STAGE="$(mktemp -d -t bbpkg)"
# ${var:-} so the trap is safe under `set -u` before the later dirs are created.
trap 'rm -rf "$STAGE" "${STAGE_DMG:-}" "${PKG_SCRIPTS:-}" "${PKG_COMPONENT_DIR:-}" 2>/dev/null || true' EXIT
mkdir -p "$STAGE/Applications" "$STAGE/Users/Shared/BabelBrain/versions/$BUILD_ID"
/usr/bin/ditto "$LAUNCHER_APP" "$STAGE/Applications/BabelBrain.app"
/usr/bin/ditto "$SELECTOR_APP" "$STAGE/Applications/BabelBrain-Version-Selector.app"
/usr/bin/ditto "$VERSION_APP" "$STAGE/Users/Shared/BabelBrain/versions/$BUILD_ID/BabelBrain.app"

# ---------------------------------------------------------------------------
# 4. Build the (unsigned) PKG.
# ---------------------------------------------------------------------------
echo ">> Building unsigned PKG: $PKG"
VERSION_STR="$(cat version.txt)"
[[ -f "$PKG" ]] && rm -f "$PKG"
# postinstall records the seeded build so the Hub adopts it as the default;
# without it the new version installs but the previous selection keeps running.
#
# It MUST be attached to the *component* package via pkgbuild. productbuild's
# own --scripts only adds distribution-level scripts, which are never executed
# for a --root synthesized product: the component's PackageInfo comes out with
# no <scripts> element at all, so the postinstall ships but never runs.
# pkgbuild also gives the component a stable identifier, instead of the random
# per-build "bbpkg.XXXXXXXX" productbuild --root invents.
PKG_SCRIPTS="$(mktemp -d -t bbscripts)"
./Hub/make_pkg_scripts.sh "$BUILD_ID" "$PKG_SCRIPTS"
PKG_COMPONENT_DIR="$(mktemp -d -t bbcomp)"
pkgbuild \
  --root "$STAGE" \
  --scripts "$PKG_SCRIPTS" \
  --identifier com.ucalgary.babelbrain.pkg \
  --version "$VERSION_STR" \
  --install-location / \
  "$PKG_COMPONENT_DIR/component.pkg"
productbuild \
  --identifier com.ucalgary.babelbrain.pkg \
  --version "$VERSION_STR" \
  --package "$PKG_COMPONENT_DIR/component.pkg" \
  "$PKG"

# Fail loudly rather than ship a PKG that seeds a version without making it the
# default: that regression is invisible until someone runs the installer.
./Hub/verify_pkg_scripts.sh "$PKG"

# ---------------------------------------------------------------------------
# 5. Wrap the PKG in a DMG (unsigned).
# ---------------------------------------------------------------------------
echo ">> Building DMG: $DMG"
[[ -f "$DMG" ]] && rm -f "$DMG"
STAGE_DMG="$(mktemp -d -t bbdmg)"
cp "$PKG" "$STAGE_DMG/BabelBrain.pkg"
cp ../LICENSE "$STAGE_DMG/" 2>/dev/null || true

if command -v create-dmg >/dev/null; then
  create-dmg \
    --volname "BabelBrain Installer (unsigned)" \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 100 \
    --icon "BabelBrain.pkg" 200 190 \
    "$DMG" \
    "$STAGE_DMG/" \
  || hdiutil create -volname "BabelBrain (unsigned)" -srcfolder "$STAGE_DMG" \
       -ov -format UDZO "$DMG"
else
  echo ">> create-dmg not found (brew install create-dmg) — using hdiutil"
  hdiutil create -volname "BabelBrain (unsigned)" -srcfolder "$STAGE_DMG" \
    -ov -format UDZO "$DMG"
fi

echo ""
echo "Done: $(pwd)/$DMG   (contains $PKG)"
echo "Install: open the DMG and double-click BabelBrain.pkg (needs your password)."
echo "It installs both apps to /Applications and seeds version $BUILD_ID into"
echo "/Users/Shared/BabelBrain/versions. For a fast loop you can instead run:"
echo "  dist/selector/BabelBrain-Version-Selector.app/Contents/MacOS/BabelBrain-Version-Selector"
