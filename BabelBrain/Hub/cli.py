'''
Hub command-line entry.

The Hub owns only a small set of flags; everything else is forwarded, unchanged,
to the selected BabelBrain version. Two ways to pass BabelBrain arguments:

* after a ``--`` separator: ``BabelBrain --version 0.8.2 -- --serve`` , or
* directly (anything the Hub does not recognise is forwarded), so Brainsight's
  ``BabelBrain -bInUseWithBrainsight`` works without changes.

Hub flags::

    --version SELECTOR     launch this version (build_id or version string), no picker
    --list-versions        print installed versions and exit
    --no-picker            launch the remembered/default version without the picker
    --show-prereleases     include pre-releases when the picker opens
    --install-worker FILE  internal: elevated helper that finishes a shared install
'''
from __future__ import annotations

import argparse
import sys

from . import installer, state as state_mod, versions as versions_mod
from .launcher import launch


def _split_forwarded(argv: list[str]) -> tuple[list[str], list[str]]:
    '''Split argv at the first ``--``; everything after is forwarded verbatim.'''
    if '--' in argv:
        i = argv.index('--')
        return argv[:i], argv[i + 1:]
    return argv, []


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='BabelBrain', add_help=True,
        description='BabelBrain launcher — choose and run a BabelBrain version.')
    p.add_argument('--version', dest='selector', default=None,
                   help='Launch this version (build id or version string) without the picker.')
    p.add_argument('--list-versions', action='store_true',
                   help='List installed versions and exit.')
    p.add_argument('--no-picker', action='store_true',
                   help='Launch the remembered/default version without showing the picker.')
    p.add_argument('--show-prereleases', action='store_true',
                   help='Include pre-releases when the picker opens.')
    p.add_argument('--install-worker', dest='install_worker', default=None,
                   help=argparse.SUPPRESS)   # internal elevated helper
    return p


def _default_version(versions: list[versions_mod.VersionInfo],
                     st: state_mod.HubState) -> versions_mod.VersionInfo | None:
    '''Resolve the version to run without a picker: remembered first, else the
    highest-versioned installed build, else the builtin.'''
    if st.remembered_build_id:
        vi = versions_mod.find_by_selector(versions, st.remembered_build_id)
        if vi is not None:
            return vi
    if not versions:
        return None

    def key(vi):
        return tuple(int(''.join(c for c in part if c.isdigit()) or 0)
                     for part in vi.version.split('.'))
    return max(versions, key=key)


def _print_versions(versions: list[versions_mod.VersionInfo]):
    if not versions:
        print('No BabelBrain versions are installed.')
        return
    for vi in versions:
        print(f'{vi.build_id:24s}  {vi.display_name:32s}  [{vi.scope_label}]')


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hub_argv, after_sep = _split_forwarded(argv)

    parser = _build_parser()
    args, unknown = parser.parse_known_args(hub_argv)

    # Elevated worker: do only the privileged move, no GUI.
    if args.install_worker:
        return installer.run_install_worker(args.install_worker)

    # Unknown hub args + everything after '--' are forwarded to BabelBrain.
    forwarded = unknown + after_sep

    st = state_mod.load()
    if args.show_prereleases:
        st.show_prereleases = True
    versions = versions_mod.discover()

    if args.list_versions:
        _print_versions(versions)
        return 0

    # Direct selection: skip the picker entirely.
    if args.selector:
        vi = versions_mod.find_by_selector(versions, args.selector)
        if vi is None:
            sys.stderr.write(
                f"error: no installed version matches '{args.selector}'. "
                f"Run 'BabelBrain --list-versions' to see what is installed.\n")
            return 3
        return launch(vi, forwarded)

    # No picker: launch remembered/default.
    if args.no_picker or st.dont_ask:
        vi = _default_version(versions, st)
        if vi is not None:
            return launch(vi, forwarded)
        # Nothing resolvable — fall through to the picker.

    return _run_picker(st, versions, forwarded)


def _run_picker(st: state_mod.HubState, versions, forwarded: list[str]) -> int:
    # Import Qt lazily so headless uses (--list-versions, worker) need no display.
    from PySide6.QtWidgets import QApplication
    from .ui import HubWindow

    app = QApplication.instance() or QApplication([])
    win = HubWindow(st)
    ret = win.exec()
    if ret != HubWindow.Accepted:
        return 0                       # user quit the launcher
    vi = win.selected_version()
    if vi is None:
        return 0
    return launch(vi, forwarded)
