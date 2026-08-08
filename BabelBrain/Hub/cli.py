'''
Command-line entry shared by the two installed apps:

* **BabelBrain.app** (mode='launcher') — opens BabelBrain directly by running the
  currently-selected version, no picker. This is the "as before" main app.
* **BabelBrain-Version-Selector.app** (mode='selector') — the picker: choose,
  download, and switch versions. Selecting a version here records it as the
  current selection, so BabelBrain.app then runs it.

**launcher mode is fully transparent**: it intercepts NOTHING and forwards every
argument to the selected version, so BabelBrain.app behaves exactly like running
BabelBrain.py directly — ``BabelBrain --help`` shows BabelBrain's own help,
``BabelBrain --serve …`` / ``BabelBrain -bInUseWithBrainsight`` all pass straight
through. Version choice is the Version Selector's job, not this app's.

**selector mode** owns a small set of flags and forwards the rest (unknown flags,
and anything after a ``--`` separator) to the version it launches::

    --version SELECTOR     run this version (build id or version string) directly
    --list-versions        print installed versions and exit
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
        description='BabelBrain launcher — run and switch between BabelBrain versions.')
    p.add_argument('--version', dest='selector', default=None,
                   help='Run this version (build id or version string) directly.')
    p.add_argument('--list-versions', action='store_true',
                   help='List installed versions and exit.')
    p.add_argument('--show-prereleases', action='store_true',
                   help='Include pre-releases when the picker opens.')
    p.add_argument('--install-worker', dest='install_worker', default=None,
                   help=argparse.SUPPRESS)   # internal elevated helper
    return p


def _default_version(versions: list[versions_mod.VersionInfo],
                     st: state_mod.HubState) -> versions_mod.VersionInfo | None:
    '''The version to run without a picker: the current selection first, else the
    highest-versioned installed build.'''
    if st.current_build_id:
        vi = versions_mod.find_by_selector(versions, st.current_build_id)
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


def _no_version_dialog():
    '''Tell the user (of BabelBrain.app) how to get a version when none exist.'''
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication([])  # noqa: F841
        QMessageBox.warning(
            None, 'BabelBrain',
            'No BabelBrain version is installed yet.\n\n'
            'Open "BabelBrain Version Selector" to download one.')
    except Exception:  # noqa: BLE001 - headless: fall back to stderr
        sys.stderr.write('No BabelBrain version is installed. Open the '
                         'BabelBrain Version Selector to download one.\n')


def main(argv: list[str] | None = None, mode: str = 'selector') -> int:
    '''Entry point. ``mode`` is 'launcher' for BabelBrain.app (run current
    version, no picker) or 'selector' for the Version Selector (show the picker).'''
    argv = list(sys.argv[1:] if argv is None else argv)

    # BabelBrain.app: fully transparent. Run the current version and forward
    # EVERY argument to it, intercepting nothing — so it behaves exactly like
    # launching BabelBrain.py directly (--help, --serve, -bInUseWithBrainsight …).
    if mode == 'launcher':
        st = state_mod.load()
        versions = versions_mod.discover()
        vi = _default_version(versions, st)
        if vi is None:
            _no_version_dialog()
            return 3
        return launch(vi, argv)

    # Version Selector: parse the small set of hub flags, forward the rest.
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

    # Explicit selection (advanced / scripts): run it directly.
    if args.selector:
        vi = versions_mod.find_by_selector(versions, args.selector)
        if vi is None:
            sys.stderr.write(
                f"error: no installed version matches '{args.selector}'. "
                f"Run 'BabelBrain --list-versions' to see what is installed.\n")
            return 3
        return launch(vi, forwarded)

    # Version Selector: always show the picker.
    return _run_picker(st, versions, forwarded)


def _run_picker(st: state_mod.HubState, versions, forwarded: list[str]) -> int:
    # Import Qt lazily so headless uses (--list-versions, worker) need no display.
    from PySide6.QtWidgets import QApplication
    from .ui import HubWindow

    app = QApplication.instance() or QApplication([])  # noqa: F841
    win = HubWindow(st)
    ret = win.exec()
    if ret != HubWindow.Accepted:
        return 0                       # user closed the selector
    vi = win.selected_version()
    if vi is None:
        return 0
    return launch(vi, forwarded)
