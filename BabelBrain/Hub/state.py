'''
Persistent Hub preferences: the remembered version choice and a few toggles.

Stored as YAML in ``~/.config/BabelBrain/hub.yaml`` alongside BabelBrain's own
state. All reads are tolerant of a missing/corrupt file (return defaults); a
write failure is reported to the caller but never fatal.
'''
from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml

from . import paths


@dataclass
class HubState:
    remembered_build_id: str | None = None   # last version the user chose
    dont_ask: bool = False                    # skip the picker and launch remembered
    show_prereleases: bool = False            # reveal prerelease entries in the picker
    preferred_scope: str = 'user'             # default install target: 'user' | 'shared'


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
