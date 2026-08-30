'''
Persistent Hub preferences: the remembered version choice and a few toggles.

Stored as YAML in ``~/.config/BabelBrain/hub.yaml`` alongside BabelBrain's own
state. All reads are tolerant of a missing/corrupt file (return defaults); a
write failure is reported to the caller but never fatal.

Also implements *installer adoption*: the platform installers drop a
``default_build.json`` marker naming the build they just seeded, and the first
Hub run after that install promotes it to the current selection. The state is
per-user while the marker is per-machine, so each user adopts a given install
exactly once and their own later choices are never overwritten.
'''
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from . import paths


@dataclass
class HubState:
    current_build_id: str | None = None       # the version BabelBrain.app runs
    show_prereleases: bool = False            # reveal prerelease entries in the picker
    preferred_scope: str = 'user'             # default install target: 'user' | 'shared'
    adopted_default: str | None = None        # installer marker already adopted ("<build_id>@<stamp>")


def load() -> HubState:
    f = paths.state_file()
    if not f.is_file():
        return HubState()
    try:
        with open(f) as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return HubState()
    if not isinstance(data, dict):
        return HubState()
    known = {k: data[k] for k in HubState().__dict__ if k in data}
    return HubState(**known)


def save(state: HubState) -> bool:
    '''Persist state. Returns False (without raising) if it could not be written
    so callers can surface the problem instead of failing silently.'''
    f = paths.state_file()
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, 'w') as fh:
            yaml.safe_dump(asdict(state), fh)
        return True
    except (OSError, yaml.YAMLError):
        return False


# ---------------------------------------------------------------------------
# Installer adoption
# ---------------------------------------------------------------------------

def _read_marker(path: Path) -> tuple[str, str] | None:
    '''``(build_id, stamp)`` from one marker file, or None if unreadable.

    ``stamp`` is opaque: it only has to differ between two installs of the same
    build_id, so no date parsing is done on it.
    '''
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    build_id = data.get('build_id')
    if not build_id:
        return None
    return str(build_id), str(data.get('installed_at') or '')


def pending_default() -> tuple[str, str] | None:
    '''The most recently written installer marker as ``(build_id, stamp)``.

    Both roots are checked and the newest file wins, so a fresh per-user install
    takes precedence over an older shared one and vice versa.
    '''
    best: tuple[float, tuple[str, str]] | None = None
    for marker in paths.default_build_markers():
        try:
            mtime = marker.stat().st_mtime
        except OSError:
            continue
        parsed = _read_marker(marker)
        if parsed is None:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, parsed)
    return best[1] if best else None


def adopt_installer_default(state: HubState, installed_build_ids) -> bool:
    '''Promote a newly installed build to the current selection, once.

    Called on every Hub start. Returns True when the selection changed (and was
    persisted). A marker whose bundle is not actually present is left
    un-adopted, so a partially-removed install does not strand the selection on
    a version that cannot run.
    '''
    pending = pending_default()
    if pending is None:
        return False
    build_id, stamp = pending
    key = f'{build_id}@{stamp}'
    if state.adopted_default == key:
        return False                       # this user already adopted this install
    if build_id not in set(installed_build_ids):
        return False                       # marker without a runnable bundle
    state.adopted_default = key
    state.current_build_id = build_id
    save(state)
    return True
