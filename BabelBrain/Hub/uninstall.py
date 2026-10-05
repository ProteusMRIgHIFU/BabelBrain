'''
Complete removal of everything BabelBrain installs, so a user who wants the app
gone is not left with tens of gigabytes of version bundles behind.

Why this module exists
----------------------
Neither platform's "normal" uninstall reaches the version store:

* **macOS** has no uninstall hook at all — dragging an app to the Trash never
  runs code, and a PKG has no counterpart to ``postinstall``. Everything the
  installer seeded into ``/Users/Shared/BabelBrain`` and everything the Version
  Selector downloaded into ``~/Library/Application Support/BabelBrain`` would
  simply stay. Hence a separate **BabelBrain Uninstaller.app**, installed next
  to the other two apps so it is found exactly when someone goes looking.
* **Windows** does have a hook, but the Inno uninstaller only removes ``{app}``;
  the store lives in ``%LOCALAPPDATA%\\BabelBrain``, deliberately outside it. The
  uninstaller therefore calls into this module (see ``BabelBrain.iss``).

What is removed, and what is not
--------------------------------
Three categories, see :data:`CATEGORIES`:

``versions``  the version stores — always removed, this is the whole point.
``apps``      the installed launcher apps (macOS only; on Windows the Inno
              uninstaller owns ``{app}`` and is running from it).
``settings``  ``~/.config/BabelBrain`` (preferences, install id, **custom
              transducers**), ``~/.babelbrain`` and ``~/.BabelBrainSync``.
              **Opt-in, off by default**: it is a tiny footprint, developers and
              users who reinstall usually want it kept, and the custom
              transducers in there are the user's own work. The UI asks for it
              separately so a full wipe is still one click away.

The user's *study data* — input images, the ``.ini`` files BabelBrain writes
next to a dataset, and simulation outputs — is never touched: it lives in
folders the user chose and losing it would be far worse than leaving a cache
behind.

Elevation
---------
The macOS PKG installs as root, so ``/Applications/*.app`` and
``/Users/Shared/BabelBrain`` are root-owned. Everything that can be removed
unprivileged is removed first; whatever is left is then done in a **single**
elevated batch (one password prompt), following the same no-silent-fallback
rule as :mod:`Hub.installer` — a declined prompt is reported, never worked
around.
'''
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

#: macOS package identifier of the PKG installer, whose receipt is forgotten at
#: the end of an uninstall so the system no longer believes BabelBrain is there.
PKG_IDENTIFIER = 'com.ucalgary.babelbrain.pkg'

#: Bundle identifiers of the three installed apps, as set in their .spec files.
#: macOS creates preference/state/cache files named after these behind the
#: app's back, which are part of "leaving nothing behind".
BUNDLE_IDS = (
    'com.ucalgary.babelbrain',             # the version apps (and pre-0.8.9 launcher)
    'com.ucalgary.babelbrain.launcher',
    'com.ucalgary.babelbrain.selector',
    'com.ucalgary.babelbrain.uninstaller',
)

#: Category -> (title, user-facing description, removed by default?)
CATEGORIES: dict[str, tuple[str, str, bool]] = {
    'versions': ('BabelBrain versions',
                 'Every installed BabelBrain version and the version store.',
                 True),
    'apps': ('Applications',
             'The BabelBrain app, the Version Selector, this uninstaller, and '
             'the preference/cache files macOS generates for them.',
             True),
    'settings': ('Settings and custom transducers',
                 'Preferences, the remembered version choice, the installation '
                 'id and any transducers you created.',
                 False),
}


@dataclass(frozen=True)
class FootprintItem:
    '''One thing on disk that belongs to BabelBrain.'''
    path: Path
    category: str
    label: str
    size: int = 0                  # bytes, 0 when absent
    is_self: bool = False          # the running uninstaller app itself

    @property
    def human_size(self) -> str:
        return human_bytes(self.size)


@dataclass
class PurgeReport:
    removed: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    elevation_denied: bool = False
    receipt_forgotten: bool = False
    self_delete_scheduled: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.elevation_denied


def human_bytes(n: int) -> str:
    if n <= 0:
        return '—'
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f'{value:.0f} {unit}' if unit in ('B', 'KB') else f'{value:.1f} {unit}'
        value /= 1024
    return f'{value:.1f} TB'


def dir_size(path: Path) -> int:
    '''Bytes used by a file or directory tree. Unreadable entries count as 0 —
    an approximate size must never stop an uninstall.'''
    try:
        if path.is_symlink() or path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += st.st_size
    return total


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def _store_dirs() -> list[tuple[Path, str]]:
    '''The per-user and shared store directories, i.e. the parent of each
    versions root — it also holds the installer's ``default_build.json``
    marker, which must go with it.'''
    return [(root.parent, scope) for root, scope in paths.versions_roots()]


def _app_dirs() -> list[Path]:
    '''Installed launcher apps (macOS). Both the system-wide ``/Applications``
    and the per-user ``~/Applications`` are checked, since a user may have moved
    or installed them either way.'''
    if not paths.IS_MAC:
        return []
    names = ('BabelBrain.app',
             'BabelBrain-Version-Selector.app',
             'BabelBrain-Uninstaller.app')
    bases = (Path('/Applications'), Path.home() / 'Applications')
    return [base / name for base in bases for name in names]


def _macos_crumbs() -> list[Path]:
    """Per-user files macOS writes for an app bundle without the app asking:
    the preferences plist, the saved window state and the cache directory.

    They are generated, not authored, so they go with the apps rather than with
    the user's settings — nobody wants to preserve a window position across an
    uninstall. All are user-owned, so no elevation is needed.
    """
    if not paths.IS_MAC:
        return []
    home = Path.home()
    out: list[Path] = []
    for bundle_id in BUNDLE_IDS:
        out.append(home / 'Library' / 'Preferences' / f'{bundle_id}.plist')
        out.append(home / 'Library' / 'Saved Application State' / f'{bundle_id}.savedState')
        out.append(home / 'Library' / 'Caches' / bundle_id)
    return out


def _settings_paths() -> list[Path]:
    '''Small state outside the stores. Kept unless the user opts in.'''
    home = Path.home()
    return [
        paths.config_dir(),            # ~/.config/BabelBrain (incl. Transducers/)
        home / '.babelbrain',          # remote-server definitions (RemoteServers.py)
        home / '.BabelBrainSync',      # Brainsight sync directory (BabelBrain.py)
    ]


def self_app() -> Path | None:
    '''The ``.app`` bundle this uninstaller is running from, if any.

    It has to be deleted *last* and from outside this process (see
    :func:`_schedule_self_delete`): removing a running PyInstaller bundle pulls
    dylibs out from under it that may not be loaded yet.
    '''
    if not paths.IS_MAC or not getattr(sys, 'frozen', False):
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == '.app':
            return parent
    return None


def footprint(include_settings: bool = False,
              existing_only: bool = True) -> list[FootprintItem]:
    '''Everything BabelBrain owns on this machine, with sizes.

    ``include_settings`` adds the opt-in ``settings`` category. Sizes are
    computed by walking the trees, which is why the UI shows a wait cursor.
    '''
    me = self_app()
    items: list[FootprintItem] = []

    def add(path: Path, category: str, label: str):
        exists = path.exists() or path.is_symlink()
        if existing_only and not exists:
            return
        items.append(FootprintItem(
            path=path, category=category, label=label,
            size=dir_size(path) if exists else 0,
            is_self=(me is not None and path == me)))

    for store, scope in _store_dirs():
        label = ('Versions installed for you' if scope == 'user'
                 else 'Versions installed for all users')
        add(store, 'versions', label)
    for app in _app_dirs():
        add(app, 'apps', app.name)
    for crumb in _macos_crumbs():
        add(crumb, 'apps', crumb.name)
    if include_settings:
        for p in _settings_paths():
            add(p, 'settings', p.name)
    return items


def has_pkg_receipt() -> bool:
    '''True when the macOS installer receipt is still registered.'''
    if not paths.IS_MAC:
        return False
    try:
        out = subprocess.run(['/usr/sbin/pkgutil', '--pkgs=' + PKG_IDENTIFIER],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def is_removable(path: Path) -> bool:
    '''Guard for every deletion, including the ones performed as root.

    A path is removable only if it is one of the exact locations this module
    inventories. Nothing is derived from user input, so a bug (or a tampered
    elevated job spec) cannot turn the uninstaller into ``rm -rf`` on an
    arbitrary directory.
    '''
    try:
        candidate = Path(path).expanduser()
    except (OSError, RuntimeError):
        return False
    known = {p for p, _ in _store_dirs()}
    known.update(_app_dirs())
    known.update(_macos_crumbs())
    known.update(_settings_paths())
    me = self_app()
    if me is not None:
        known.add(me)
    for allowed in known:
        try:
            if candidate == allowed or str(candidate) == str(allowed):
                return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def _remove(path: Path) -> tuple[bool, str]:
    '''Remove one path unprivileged. Returns (done, error message).'''
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True, ''
    except (OSError, PermissionError) as e:
        return False, str(e)


def _schedule_self_delete(app: Path) -> str:
    '''Shell snippet that waits for this process to exit, then deletes the
    uninstaller bundle. Detached and with its output redirected so that, run
    through ``osascript``, it does not keep the elevated call waiting.'''
    return (f'/usr/bin/nohup /bin/sh -c '
            f'{shlex.quote(f"while /bin/kill -0 {os.getpid()} 2>/dev/null; do /bin/sleep 0.5; done; /bin/rm -rf {shlex.quote(str(app))}")} '
            f'>/dev/null 2>&1 &')


def _elevated_macos(targets: list[Path], forget_receipt: bool,
                    me: Path | None) -> None:
    '''One authorization prompt for every root-owned leftover.'''
    from . import installer           # local import: installer imports us lazily too

    parts = [f'/bin/rm -rf {shlex.quote(str(p))}' for p in targets]
    if forget_receipt:
        # A missing receipt must not fail the batch.
        parts.append(f'/usr/sbin/pkgutil --forget {shlex.quote(PKG_IDENTIFIER)} || true')
    if me is not None:
        parts.append(_schedule_self_delete(me))
    installer._run_elevated_macos(
        ' ; '.join(parts),
        'BabelBrain needs administrator privileges to finish removing itself.',
        targets[0] if targets else Path('/Applications'), 'removed')


def _elevated_windows(targets: list[Path]) -> None:
    from . import installer

    installer._run_elevated_windows(
        {'action': 'purge', 'paths': [str(p) for p in targets]},
        targets[0], 'removed')


def _run_elevated(report: PurgeReport, pending: list[Path], job) -> None:
    '''Run one elevated batch and fold its outcome into ``report``.

    A declined prompt is recorded as such rather than as a failure, so the UI
    can say "you cancelled" instead of "it broke".
    '''
    from . import installer

    try:
        job()
    except installer.ElevationDenied:
        report.elevation_denied = True
        report.failed.extend((p, 'administrator privileges declined') for p in pending)
        return
    except Exception as e:                      # noqa: BLE001 - surface anything
        report.failed.extend((p, str(e)) for p in pending)
        return
    for p in pending:
        if p.exists():
            report.failed.append((p, 'could not be removed'))
        else:
            report.removed.append(p)


def purge(include_settings: bool = False,
          allow_elevation: bool = True,
          forget_receipt: bool = True) -> PurgeReport:
    '''Remove everything in the selected categories.

    Unprivileged removals happen first; whatever is left (root-owned stores and
    apps on macOS, a machine-wide store on Windows) is done in one elevated
    batch. The running uninstaller bundle is always deleted last, from a
    detached watcher process.

    ``allow_elevation=False`` is used by the Windows uninstaller, which must
    never pop a UAC prompt of its own: it reports what it could not remove
    instead.
    '''
    report = PurgeReport()
    me = self_app()
    items = footprint(include_settings=include_settings)

    pending: list[Path] = []
    for item in items:
        if item.is_self:
            continue                      # handled at the very end
        if not is_removable(item.path):   # cannot happen; cheap insurance
            report.failed.append((item.path, 'not a BabelBrain-owned location'))
            continue
        done, err = _remove(item.path)
        if done:
            report.removed.append(item.path)
        else:
            pending.append(item.path)

    need_receipt = forget_receipt and has_pkg_receipt()
    need_self = me is not None and me.exists()

    if paths.IS_MAC:
        # Does deleting the uninstaller itself need root? /Applications is
        # root-owned after a PKG install; a hand-copied ~/Applications is not.
        self_needs_root = need_self and not os.access(str(me.parent), os.W_OK)
        if pending or need_receipt or self_needs_root:
            if not allow_elevation:
                report.failed.extend(
                    (p, 'administrator privileges required') for p in pending)
            else:
                _run_elevated(report, pending,
                              lambda: _elevated_macos(
                                  pending, need_receipt,
                                  me if self_needs_root else None))
                if report.ok:
                    report.receipt_forgotten = need_receipt
                    report.self_delete_scheduled = self_needs_root
        if need_self and not report.self_delete_scheduled:
            # Unprivileged case: same watcher, just not behind a prompt.
            subprocess.Popen(['/bin/sh', '-c', _schedule_self_delete(me)],
                             start_new_session=True)
            report.self_delete_scheduled = True
    elif pending:
        if not allow_elevation:
            report.failed.extend(
                (p, 'administrator privileges required') for p in pending)
        else:
            _run_elevated(report, pending, lambda: _elevated_windows(pending))

    return report


# ---------------------------------------------------------------------------
# Text reporting (used by --list-footprint and the quiet Windows path)
# ---------------------------------------------------------------------------

def format_footprint(items: list[FootprintItem]) -> str:
    if not items:
        return 'Nothing installed by BabelBrain was found on this machine.'
    lines = []
    total = 0
    for cat, (title, _desc, _default) in CATEGORIES.items():
        rows = [i for i in items if i.category == cat]
        if not rows:
            continue
        lines.append(f'{title}:')
        for i in rows:
            lines.append(f'  {i.human_size:>9}  {i.path}')
            total += i.size
    lines.append(f'{"total":>11}: {human_bytes(total)}')
    return '\n'.join(lines)
