'''
The version model and discovery.

A *version* is one complete, self-contained BabelBrain build. Each build carries
a ``build_info.json`` (written at CI build time) so the Hub can tell builds
apart by ``(version, git_commit)`` rather than by version string alone — this is
what lets a development ``0.8.8`` bundled inside the Hub coexist as a distinct
entry from a future released ``0.8.8``.

Discovery sources, in order:

* **builtin** — the read-only build shipped inside the frozen Hub, or, when
  running from source, this very repository (launched via the Python
  interpreter). Always present so the user can always run *something*.
* **user** / **shared** — bundles previously downloaded into the writable roots.
'''
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import paths


def _short(commit: str | None) -> str | None:
    return commit[:7] if commit else None


@dataclass(frozen=True)
class VersionInfo:
    version: str                       # e.g. "0.8.8"
    channel: str                       # "stable" | "prerelease" | "dev"
    scope: str                         # "builtin" | "user" | "shared"
    location: Path                     # the bundle directory (or repo root for source)
    git_commit: str | None = None
    built: str | None = None           # ISO timestamp
    tag: str | None = None             # git tag this build came from, if any
    from_source: bool = False          # launch via the Python interpreter

    # -- identity -----------------------------------------------------------
    @property
    def build_id(self) -> str:
        '''Stable identity used to remember a choice. Distinguishes two builds
        that share a version string but differ in commit.'''
        sc = _short(self.git_commit)
        return f'{self.version}+{sc}' if sc else self.version

    @property
    def display_name(self) -> str:
        sc = _short(self.git_commit)
        if self.channel == 'dev':
            tag = 'source' if self.from_source else 'bundled'
            return f'{self.version} (dev · {tag}{" · " + sc if sc else ""})'
        if self.channel == 'prerelease':
            return f'{self.version} (pre-release{" · " + sc if sc else ""})'
        return self.version

    @property
    def scope_label(self) -> str:
        return {
            'builtin': 'shipped with launcher',
            'user': 'installed for you',
            'shared': 'installed for all users',
        }.get(self.scope, self.scope)

    # -- launching ----------------------------------------------------------
    def launch_argv(self) -> list[str]:
        '''The argv prefix that starts this version; forwarded app args are
        appended by the launcher.'''
        if self.from_source:
            # Run the source tree with the current interpreter.
            return [sys.executable, str(self.location / 'BabelBrain.py')]
        return [str(self._executable())]

    def _executable(self) -> Path:
        if paths.IS_MAC:
            # location is either a BabelBrain.app or a dir containing one.
            app = self.location if self.location.suffix == '.app' else self.location / 'BabelBrain.app'
            return app / 'Contents' / 'MacOS' / 'BabelBrain'
        if paths.IS_WINDOWS:
            return self.location / 'BabelBrain.exe'
        return self.location / 'BabelBrain'

    def is_runnable(self) -> bool:
        if self.from_source:
            return (self.location / 'BabelBrain.py').is_file()
        return self._executable().is_file()


# ---------------------------------------------------------------------------
# build_info.json
# ---------------------------------------------------------------------------

def read_build_info(bundle_dir: Path) -> dict | None:
    '''Load ``build_info.json`` from a bundle directory (searching the common
    macOS ``Contents/Resources`` nesting too). Returns ``None`` if absent.'''
    candidates = [
        bundle_dir / 'build_info.json',
        bundle_dir / 'BabelBrain.app' / 'Contents' / 'Resources' / 'build_info.json',
        bundle_dir / 'Contents' / 'Resources' / 'build_info.json',
    ]
    for c in candidates:
        if c.is_file():
            try:
                with open(c) as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None
    return None


def _version_from_build_info(info: dict, scope: str, location: Path) -> VersionInfo:
    return VersionInfo(
        version=str(info.get('version', '0.0.0')),
        channel=str(info.get('channel', 'stable')),
        scope=scope,
        location=location,
        git_commit=info.get('git_commit'),
        built=info.get('built'),
        tag=info.get('tag'),
    )


# ---------------------------------------------------------------------------
# Source / builtin discovery
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ['git', '-C', str(repo), *args],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _source_version() -> VersionInfo | None:
    '''When running from source, offer this repository as a runnable "dev"
    version, with metadata derived from git and version.txt.'''
    here = Path(__file__).resolve().parent.parent   # .../BabelBrain (the package dir)
    if not (here / 'BabelBrain.py').is_file():
        return None
    version = '0.0.0'
    vfile = here / 'version.txt'
    if vfile.is_file():
        try:
            version = vfile.read_text().strip() or version
        except OSError:
            pass
    repo = here.parent
    commit = _git(repo, 'rev-parse', 'HEAD')
    tag = _git(repo, 'describe', '--tags', '--exact-match')
    return VersionInfo(
        version=version,
        channel='dev',
        scope='builtin',
        location=here,
        git_commit=commit,
        tag=tag,
        from_source=True,
    )


def builtin_version() -> VersionInfo | None:
    '''The always-available version: the nested bundle in a frozen Hub, else
    the source tree when running from source.'''
    bundled = paths.bundled_version_dir()
    if bundled is not None:
        info = read_build_info(bundled) or {}
        info.setdefault('channel', 'dev')
        return _version_from_build_info(info, 'builtin', bundled)
    return _source_version()


# ---------------------------------------------------------------------------
# Installed-bundle discovery
# ---------------------------------------------------------------------------

def _scan_root(root: Path, scope: str) -> list[VersionInfo]:
    found: list[VersionInfo] = []
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        info = read_build_info(entry)
        if info is None:
            # Not a recognizable bundle; skip rather than guess.
            continue
        vi = _version_from_build_info(info, scope, entry)
        if vi.is_runnable():
            found.append(vi)
    return found


def discover() -> list[VersionInfo]:
    '''All runnable versions, builtin first, then user, then shared. Duplicate
    build_ids from lower-priority scopes are dropped.'''
    result: list[VersionInfo] = []
    seen: set[str] = set()

    def add(vi: VersionInfo | None):
        if vi is None or not vi.is_runnable():
            return
        if vi.build_id in seen:
            return
        seen.add(vi.build_id)
        result.append(vi)

    add(builtin_version())
    for root, scope in paths.versions_roots():
        for vi in _scan_root(root, scope):
            add(vi)
    return result


def find_by_selector(versions: list[VersionInfo], selector: str) -> VersionInfo | None:
    '''Resolve a ``--version`` selector, matching either an exact build_id
    (``0.8.8+0799425``) or a plain version string (first match wins).'''
    for vi in versions:
        if vi.build_id == selector:
            return vi
    for vi in versions:
        if vi.version == selector:
            return vi
    return None
