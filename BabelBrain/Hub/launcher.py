'''
Hand control from the Hub to a chosen BabelBrain version.

The Hub deliberately does not interpret the forwarded arguments: whatever the
user (or Brainsight) passed for BabelBrain is appended verbatim after the
version's own launch argv. This keeps the Hub agnostic to a version's evolving
CLI — a 0.8.2 build and a 0.8.8 build can accept different flags and the Hub
never needs to know.
'''
from __future__ import annotations

import os
import subprocess
import sys

from .versions import VersionInfo


def launch(version: VersionInfo, forwarded_args: list[str]) -> int:
    '''Start ``version`` with ``forwarded_args`` appended.

    On POSIX the Hub process is *replaced* by the target (``execv``) so the
    exit code and any controlling caller (e.g. Brainsight) see the version's
    process directly. On Windows ``execv`` has awkward console/quoting
    semantics, so we spawn a child and propagate its return code.
    '''
    argv = version.launch_argv() + list(forwarded_args)
    if not version.is_runnable():
        raise FileNotFoundError(f'BabelBrain executable missing for {version.display_name}')

    # Let a launched version know which Hub started it (useful for a future
    # in-app "reopen with another version" that calls back into the Hub).
    env = dict(os.environ)
    env['BABELBRAIN_HUB'] = '1'
    env['BABELBRAIN_LAUNCHED_BUILD_ID'] = version.build_id

    if os.name == 'posix':
        os.execve(argv[0], argv, env)
        # execve does not return on success.
        raise RuntimeError('execve returned unexpectedly')

    completed = subprocess.run(argv, env=env)
    return completed.returncode
