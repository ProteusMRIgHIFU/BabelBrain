# This Python file uses the following encoding: utf-8
'''
BabelBrain Hub — launcher entry point for the frozen distribution.

This is the app users double-click (and the one Brainsight invokes). It lets
them pick which BabelBrain version to run, download more, and swap between them,
then hands off to the chosen version's real binary with all remaining arguments
forwarded unchanged.

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
    sys.exit(main())
