# This Python file uses the following encoding: utf-8
'''
BabelBrain.app entry point — the main app users double-click.

It opens BabelBrain directly by running the currently-selected version (chosen in
BabelBrain-Version-Selector.app), forwarding any arguments — so it behaves "as
before" and Brainsight can invoke it with ``-bInUseWithBrainsight``. It never
shows a picker; use the Version Selector to change or download versions.

Developers running from source do not need this — launch BabelBrain directly
with ``python BabelBrain/BabelBrain.py``.
'''
import multiprocessing
import os
import sys

# Keep imports resolvable both from source and when frozen.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Hub.cli import main  # noqa: E402

if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main(mode='launcher'))
