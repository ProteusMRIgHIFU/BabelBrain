'''
BabelBrain Hub
==============

A small launcher that lets users of the *frozen* BabelBrain app pick which
BabelBrain version to run and swap between versions without uninstalling.

The Hub is a separate app from BabelBrain itself: it discovers installed
versions (a read-only build bundled inside the Hub, plus versions downloaded
into per-user and shared locations), optionally downloads more from a curated
release manifest, then hands control to the selected version's real binary by
forwarding all remaining CLI arguments unchanged.

Developers running from source do not need the Hub — they keep launching
``python BabelBrain/BabelBrain.py`` and swap versions with git. The Hub is only
relevant to the packaged distribution.

See the design note ``babelbrain-hub-launcher`` for the full rationale.
'''

HUB_VERSION = "1.0.0"
'''Version of the Hub launcher itself, independent of any BabelBrain version.
Compared against the manifest ``hub`` channel to offer a launcher update.'''
