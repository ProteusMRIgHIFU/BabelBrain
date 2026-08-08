# This Python file uses the following encoding: utf-8
'''
BabelBrain Version Selector — entry point for BabelBrain-Version-Selector.app.

The picker: choose which BabelBrain version to run, download more, and switch
between them. Launching a version here records it as the current selection, so
BabelBrain.app (see babelbrain_launcher.py) then opens that version directly.

Developers running from source do not need this — launch BabelBrain directly
with ``python BabelBrain/BabelBrain.py`` and swap versions using git.
'''
import multiprocessing
import os
import sys

# Keep imports resolvable both from source and when frozen.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Hub.cli import main  # noqa: E402

if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main(mode='selector'))
