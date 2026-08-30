'''
Filesystem locations and the platform identity key used by the Hub.

Two kinds of location:

* **Small state** (remembered choice, cached manifest) lives in the same
  ``~/.config/BabelBrain`` directory BabelBrain already uses for
  ``lastselection.yaml`` / ``installation.id`` — kept cross-platform on purpose
  so it sits next to the app's existing state.

* **Version bundles** are multi-GB, so they go in the platform-idiomatic
  writable data directories, with a per-user root and a shared (all-users) root.
  The Hub scans BOTH roots and never writes bundles inside its own signed
  ``.app`` (that would break the code signature).
'''
import os
import platform
import sys
from pathlib import Path

_SYSTEM = platform.system()          # 'Darwin' | 'Windows' | 'Linux'
IS_MAC = _SYSTEM == 'Darwin'
IS_WINDOWS = _SYSTEM == 'Windows'
IS_LINUX = _SYSTEM == 'Linux'


# ---------------------------------------------------------------------------
# Small state (matches BabelBrain's existing ~/.config/BabelBrain convention)
# ---------------------------------------------------------------------------

def config_dir() -> Path:
    '''Directory for small Hub state; created on demand by callers that write.'''
    return Path.home() / '.config' / 'BabelBrain'


def state_file() -> Path:
    '''YAML holding the remembered version choice and Hub preferences.'''
    return config_dir() / 'hub.yaml'


def manifest_cache_file() -> Path:
    '''Last successfully fetched release manifest, used when offline.'''
    return config_dir() / 'manifest_cache.json'


# ---------------------------------------------------------------------------
# Version bundle roots (per-user and shared/all-users)
# ---------------------------------------------------------------------------

def user_versions_root() -> Path:
    '''Per-user, always-writable location for downloaded version bundles.'''
    if IS_MAC:
        return Path.home() / 'Library' / 'Application Support' / 'BabelBrain' / 'versions'
    if IS_WINDOWS:
        base = os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local')
        return Path(base) / 'BabelBrain' / 'versions'
    # Linux / other: XDG data home
    base = os.environ.get('XDG_DATA_HOME') or (Path.home() / '.local' / 'share')
    return Path(base) / 'BabelBrain' / 'versions'


def shared_versions_root() -> Path:
    '''Shared, all-users location. Writing here may require elevation
    (Windows %ProgramData%, Linux /opt). On macOS /Users/Shared itself is
    writable without admin, but the PKG installer seeds versions there as root,
    so an existing store is root-owned and both installing and removing need
    elevation — see installer._place_elevated_macos / uninstall().'''
    if IS_MAC:
        return Path('/Users/Shared/BabelBrain/versions')
    if IS_WINDOWS:
        base = os.environ.get('PROGRAMDATA') or r'C:\ProgramData'
        return Path(base) / 'BabelBrain' / 'versions'
    return Path('/opt/BabelBrain/versions')


def versions_roots() -> list[tuple[Path, str]]:
    '''(root, scope) pairs to scan, scope in {"user", "shared"}.'''
    return [(user_versions_root(), 'user'), (shared_versions_root(), 'shared')]


def default_build_markers() -> list[Path]:
    '''Marker files a platform installer writes to name the build it just
    seeded, so the Hub can adopt it as the default version.

    One marker per bundle root, stored *beside* the root (not inside it, which
    would be scanned as a bundle):

        macOS  : /Users/Shared/BabelBrain/default_build.json      (PKG)
        Windows: %LOCALAPPDATA%\\BabelBrain\\default_build.json     (Inno Setup)

    Written by the installer running with whatever privileges it has; read by
    every user, each of whom adopts a given marker at most once.
    '''
    return [root.parent / 'default_build.json' for root, _ in versions_roots()]


def root_for_scope(scope: str) -> Path:
    if scope == 'user':
        return user_versions_root()
    if scope == 'shared':
        return shared_versions_root()
    raise ValueError(f'unknown scope {scope!r}')


# ---------------------------------------------------------------------------
# Platform identity used to pick the right download asset from the manifest
# ---------------------------------------------------------------------------

def platform_key() -> str:
    '''Matches the per-platform asset keys published in releases.json.'''
    machine = platform.machine().lower()
    if IS_MAC:
        # arm64 / aarch64 -> apple silicon; anything else -> intel build.
        if machine in ('arm64', 'aarch64'):
            return 'macos-arm64'
        return 'macos-x64'
    if IS_WINDOWS:
        return 'windows-x64'
    return 'linux-x64'


# ---------------------------------------------------------------------------
# Locating the read-only version bundled inside the Hub itself
# ---------------------------------------------------------------------------

def bundled_version_dir() -> Path | None:
    '''Path to the read-only BabelBrain version shipped *inside* the Hub, or
    ``None`` when running from source (no nested bundle).

    Resolved *relative to the Hub executable* rather than PyInstaller's
    ``sys._MEIPASS``, so the location is deterministic and CI can place the
    version there with the correct, signature-preserving tooling regardless of
    PyInstaller's internal data layout:

        macOS  : <Hub.app>/Contents/Resources/bundled/   (holds BabelBrain.app)
        Windows: <hub dir>/bundled/                       (holds BabelBrain.exe …)
    '''
    if not getattr(sys, 'frozen', False):
        return None
    exe = Path(sys.executable).resolve()
    if IS_MAC:
        # .../BabelBrain.app/Contents/MacOS/BabelBrain -> Contents/Resources/bundled
        candidate = exe.parents[1] / 'Resources' / 'bundled'
    else:
        candidate = exe.parent / 'bundled'
    return candidate if candidate.is_dir() else None
