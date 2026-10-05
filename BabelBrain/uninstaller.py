# This Python file uses the following encoding: utf-8
'''
BabelBrain Uninstaller — entry point for BabelBrain-Uninstaller.app.

macOS has no uninstall hook: dragging an app to the Trash never runs code and a
PKG has no counterpart to its ``postinstall``. Without this app, removing
BabelBrain by hand would leave the version store — potentially tens of
gigabytes under /Users/Shared and ~/Library/Application Support — behind. The
installer therefore places this next to BabelBrain.app so it is found exactly
when someone goes looking for a way to remove the app.

On Windows the Inno uninstaller does the same work by calling
``BabelBrain-Version-Selector.exe --purge-user-data``; this app is macOS-only.

Developers running from source have nothing to uninstall — just delete the
clone and the conda environment.
'''
import multiprocessing
import os
import sys

# Keep imports resolvable both from source and when frozen.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Hub.cli import main  # noqa: E402

if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main(mode='uninstaller'))
