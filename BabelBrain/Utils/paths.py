import os
import sys
from pathlib import Path

_IS_MAC = sys.platform == 'darwin'

def bundle_root(anchor: str | Path) -> Path:
    """Directory that holds BabelBrain's shared bundled resources.

    Frozen: the bundle root (``sys._MEIPASS``). From source: the parent of the
    calling module's folder, i.e. ``BabelBrain/`` for ``Options/Options.py`` or
    the repo root for ``TranscranialModeling/babel_integration/*.py``.

    Use this instead of ``resource_path(__file__).parent``: PyInstaller flattens
    packages when it resolves ``__file__``, so that expression only landed on the
    bundle root by accident and required an unrelated placeholder folder to be
    mapped in the .spec just so the existence check would pass.

    Args:
        anchor: Pass __file__ from the calling module.
    """

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)

    return Path(anchor).parent.parent


def resource_path(anchor: str | Path) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller.
    
    Args:
        anchor: Pass __file__ from the calling module.
    """
    
    anchor = Path(anchor)
    subdir = anchor.parent.name

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        root = Path(sys._MEIPASS)

        # Top-level modules (BabelBrain.py, CTZTEProcessing.py, ...) have their
        # frozen __file__ directly in the bundle root, so `subdir` is the name of
        # the bundle root itself ("Frameworks" inside a macOS .app, "_internal"
        # on Windows/Linux) rather than a folder mapped by the .spec.
        if anchor.parent == root or subdir == root.name:
            return root

        bundle_dir = root / subdir
        if not bundle_dir.exists():
            raise RuntimeError(
                f"Expected bundle subdirectory not found: {bundle_dir}\n"
                f"Check that '{subdir}' is correctly mapped in your .spec datas."
            )
        return bundle_dir

    return anchor.parent
