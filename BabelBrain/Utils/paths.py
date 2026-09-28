import os
import sys
from pathlib import Path

_IS_MAC = sys.platform == 'darwin'

def resource_path(anchor: str | Path) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller.
    
    Args:
        anchor: Pass __file__ from the calling module.
    """
    
    anchor = Path(anchor)
    subdir = anchor.parent.name

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / subdir
        if not bundle_dir.exists():
            raise RuntimeError(
                f"Expected bundle subdirectory not found: {bundle_dir}\n"
                f"Check that '{subdir}' is correctly mapped in your .spec datas."
            )
        return bundle_dir

    return anchor.parent