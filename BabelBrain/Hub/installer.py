'''
Downloading and installing version bundles.

Flow for a normal install:

1. Download the platform asset to a *user-writable* temp file and verify its
   sha256 (so the network step never needs elevation).
2. Extract it to a user-writable staging directory. On macOS we use ``ditto`` so
   the ``.app`` code signature and symlinks survive; elsewhere we use zipfile
   with unix-mode/symlink restoration.
3. Move the staged bundle into the chosen root.

Only step 3 can require elevation (installing to a shared/all-users root). If it
does and elevation is denied or fails, we raise — the caller shows the real
error and returns the user to the local/global choice. Nothing is ever silently
redirected to the user root.

Removal follows the same rule. The macOS PKG seeds the shared store as root, so
its bundles are root-owned and an unprivileged ``rmtree`` fails; both platforms
therefore have an elevated path (macOS: an authorization prompt via
``osascript``; Windows: a UAC-elevated worker process), and a declined prompt
raises :class:`ElevationDenied` rather than reporting a bare failure.
'''
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from . import netutil, paths
from .manifest import Asset, CatalogEntry

ProgressCb = Callable[[int, int], None]   # (bytes_done, bytes_total)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InstallError(Exception):
    pass


class ChecksumError(InstallError):
    pass


class ElevationRequired(InstallError):
    '''Writing to a shared root needs elevated privileges. Carries the scope so
    the UI can send the user back to the local/global choice.'''
    def __init__(self, scope: str, target: Path):
        super().__init__(f'Administrator privileges are required to install for '
                         f'all users at {target}.')
        self.scope = scope
        self.target = target


class ElevationDenied(InstallError):
    '''An elevation prompt was shown and cancelled/denied by the user.'''
    def __init__(self, target: Path, action: str = 'installed'):
        super().__init__(f'The request for administrator privileges was declined; '
                         f'nothing was {action} at {target}.')
        self.target = target


# ---------------------------------------------------------------------------
# Download + verify
# ---------------------------------------------------------------------------

def download_and_verify(asset: Asset, dest: Path, progress: ProgressCb | None = None) -> Path:
    req = urllib.request.Request(asset.url, headers={'User-Agent': 'BabelBrain-Hub'})
    sha = hashlib.sha256()
    with urllib.request.urlopen(req, context=netutil.ssl_context()) as resp:
        total = int(resp.headers.get('Content-Length', 0) or 0)
        done = 0
        with open(dest, 'wb') as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if asset.sha256:
        got = sha.hexdigest()
        if got.lower() != asset.sha256.lower():
            dest.unlink(missing_ok=True)
            raise ChecksumError(
                f'Downloaded file failed its integrity check '
                f'(expected {asset.sha256[:12]}…, got {got[:12]}…).')
    return dest


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_zipfile_posix(archive: Path, dest: Path):
    '''zipfile extraction that restores unix permissions and symlinks, which
    plain ``extractall`` drops (breaking exec bits and .app symlinks).'''
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = dest / info.filename
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                link_target = zf.read(info).decode()
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                os.symlink(link_target, target)
            elif info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out)
                if mode:
                    os.chmod(target, mode)


def extract(archive: Path, dest: Path):
    '''Extract ``archive`` into ``dest`` (created fresh). Uses the best tool for
    the platform so code signatures / permissions survive.'''
    dest.mkdir(parents=True, exist_ok=True)
    if paths.IS_MAC:
        # ditto preserves resource forks, symlinks and the .app signature.
        result = subprocess.run(['/usr/bin/ditto', '-x', '-k', str(archive), str(dest)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise InstallError(f'Failed to extract bundle: {result.stderr.strip()}')
    elif paths.IS_WINDOWS:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        _extract_zipfile_posix(archive, dest)


def _ensure_build_info(container: Path, entry: CatalogEntry):
    '''Guarantee a build_info.json exists so discovery can identify the bundle,
    synthesizing one from the manifest entry if the archive lacked it.'''
    from .versions import read_build_info
    if read_build_info(container) is not None:
        return
    info = {
        'version': entry.version,
        'git_commit': entry.git_commit,
        'channel': entry.channel,
        'built': None,
        'tag': None,
    }
    # Always write at the container top level, NEVER inside the .app: modifying
    # a signed/notarized bundle would invalidate its signature and trip
    # Gatekeeper. read_build_info() checks the container top first, so this is
    # picked up. (Bundles built by our own pipeline already carry build_info.json
    # inside the app — signed in — so this synthesis only runs for bundles that
    # shipped without one, e.g. repackaged legacy releases.)
    target = container / 'build_info.json'
    try:
        target.write_text(json.dumps(info, indent=2))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Placing the bundle (the only step that may need elevation)
# ---------------------------------------------------------------------------

def _can_write(root: Path) -> bool:
    '''True if we can create ``root`` and write inside it without elevation.'''
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / '.write_probe'
        probe.write_text('')
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _place(bundle_root: Path, target_dir: Path):
    '''Move a staged bundle to its final location (assumes write access).'''
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(bundle_root), str(target_dir))


def _hub_relaunch_argv() -> list[str]:
    '''argv prefix to re-run *this Hub*, whether frozen or from source.'''
    if getattr(sys, 'frozen', False):
        return [sys.executable]
    hub_py = Path(__file__).resolve().parent.parent / 'hub.py'
    return [sys.executable, str(hub_py)]


def _applescript_string(value: str) -> str:
    '''Quote a Python string as an AppleScript string literal.'''
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _run_elevated_macos(shell_cmd: str, prompt: str, target: Path, action: str):
    '''Run one shell command as root behind the macOS authorization dialog.

    ``osascript`` is used rather than a helper tool because the Hub ships as a
    plain signed app with no privileged helper installed: this gives the user
    the standard system prompt, and a cancelled prompt (AppleScript error -128)
    is reported as :class:`ElevationDenied` instead of a generic failure.
    '''
    script = (f'do shell script {_applescript_string(shell_cmd)} '
              f'with prompt {_applescript_string(prompt)} '
              f'with administrator privileges')
    result = subprocess.run(['/usr/bin/osascript', '-e', script],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return
    err = (result.stderr or '').strip()
    if 'User canceled' in err or 'User cancelled' in err or '(-128)' in err:
        raise ElevationDenied(target, action)
    raise InstallError(err or 'The elevated operation failed.')


def _place_elevated_macos(bundle_root: Path, target_dir: Path):
    '''macOS: move the staged bundle into a root-owned shared store.

    The bundle is left owned by root and world-readable, matching what the PKG
    installer produces, so every user on the machine can run it.
    '''
    src = shlex.quote(str(bundle_root))
    dst = shlex.quote(str(target_dir))
    parent = shlex.quote(str(target_dir.parent))
    _run_elevated_macos(
        f'/bin/rm -rf {dst} && /bin/mkdir -p {parent} && /bin/mv {src} {dst} && '
        f'/usr/sbin/chown -R root:wheel {dst} && /bin/chmod -R go+rX {dst}',
        'BabelBrain needs administrator privileges to install this version for all users.',
        target_dir, 'installed')


def _run_elevated_windows(spec: dict, target: Path, action: str):
    '''Windows: run one privileged job by relaunching the Hub under a UAC prompt
    with a hidden worker subcommand. ``spec['action']`` selects the job.'''
    import ctypes
    import time

    spec_file = Path(tempfile.mkdtemp(prefix='bbhub_')) / 'install_job.json'
    result_file = spec_file.with_suffix('.result')
    spec = dict(spec, result=str(result_file))
    spec_file.write_text(json.dumps(spec))

    argv = _hub_relaunch_argv()
    file = argv[0]
    params = subprocess.list2cmdline(argv[1:] + ['--install-worker', str(spec_file)])

    # ShellExecuteW returns >32 on success; SE_ERR_ACCESSDENIED (5) when the
    # user cancels the UAC prompt.
    rc = ctypes.windll.shell32.ShellExecuteW(None, 'runas', file, params, None, 0)
    if rc <= 32:
        raise ElevationDenied(target, action)

    # Wait for the elevated worker to finish and report through result_file.
    for _ in range(600):            # up to ~60s
        if result_file.is_file():
            break
        time.sleep(0.1)
    if not result_file.is_file():
        raise InstallError('The elevated operation did not report completion.')
    status = json.loads(result_file.read_text())
    if not status.get('ok'):
        raise InstallError(status.get('error', 'The elevated operation failed.'))


def _place_elevated_windows(bundle_root: Path, target_dir: Path):
    _run_elevated_windows({'action': 'place', 'src': str(bundle_root),
                           'dst': str(target_dir)}, target_dir, 'installed')


def _remove_elevated_windows(location: Path):
    _run_elevated_windows({'action': 'remove', 'path': str(location)},
                          location, 'removed')


def run_install_worker(spec_file: str) -> int:
    '''Entry point for the elevated worker process (invoked via
    ``--install-worker``). Performs only the privileged filesystem change and
    reports back through the result file named in the spec.'''
    spec = json.loads(Path(spec_file).read_text())
    result_file = Path(spec['result'])
    try:
        if spec.get('action') == 'remove':
            location = Path(spec['path'])
            if not _is_managed_bundle(location):
                raise InstallError(f'Refusing to remove {location}: not a '
                                   f'BabelBrain version bundle.')
            shutil.rmtree(location)
        else:
            _place(Path(spec['src']), Path(spec['dst']))
        result_file.write_text(json.dumps({'ok': True}))
        return 0
    except Exception as e:                       # report any failure to the parent
        result_file.write_text(json.dumps({'ok': False, 'error': str(e)}))
        return 1


# ---------------------------------------------------------------------------
# Top-level install
# ---------------------------------------------------------------------------

def install(entry: CatalogEntry, scope: str, progress: ProgressCb | None = None,
            allow_elevation: bool = True) -> Path:
    '''Download, verify, and install ``entry`` into ``scope``'s root.

    Returns the installed bundle directory. Raises :class:`ElevationRequired`
    (no elevation attempted or possible) or :class:`ElevationDenied` (prompt
    cancelled) so the caller can put the user back in control — never a silent
    fallback to another scope.
    '''
    if entry.asset is None:
        raise InstallError(f'No download is available for this platform '
                           f'({paths.platform_key()}).')

    target_root = paths.root_for_scope(scope)
    target_dir = target_root / entry.build_id
    if target_dir.exists():
        return target_dir                         # already installed

    work = Path(tempfile.mkdtemp(prefix='bbhub_dl_'))
    try:
        archive = download_and_verify(entry.asset, work / 'bundle.zip', progress)
        # Extract straight into a staging *container* that mirrors the final
        # layout: the container holds BabelBrain.app (macOS) or BabelBrain.exe
        # and its support files (Windows) directly. The container is later moved
        # to <root>/<build_id>, so the assets must be packaged with those items
        # at the top level of the zip (mac: ditto --keepParent on the .app;
        # windows: zip the onedir *contents*, not the enclosing folder).
        container = work / 'container'
        extract(archive, container)
        _ensure_build_info(container, entry)

        if _can_write(target_root):
            _place(container, target_dir)
            return target_dir

        # Needs elevation.
        if not allow_elevation:
            raise ElevationRequired(scope, target_root)
        if paths.IS_WINDOWS:
            _place_elevated_windows(container, target_dir)
            return target_dir
        if paths.IS_MAC:
            # A shared store seeded by the PKG is root-owned, so this is the
            # normal path for an all-users install, not an edge case.
            _place_elevated_macos(container, target_dir)
            return target_dir
        # Linux /opt would need root; we have no prompt there, and we do not
        # silently fall back to the user root.
        raise ElevationRequired(scope, target_root)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def _is_managed_bundle(location: Path) -> bool:
    '''True only for a direct child of one of the version roots, so an elevated
    ``rm -rf`` can never be pointed at an arbitrary path.'''
    try:
        loc = Path(location).resolve()
    except OSError:
        return False
    for root, _ in paths.versions_roots():
        try:
            if loc.parent == root.resolve() and loc != root.resolve():
                return True
        except OSError:
            continue
    return False


def uninstall(location: Path, allow_elevation: bool = True) -> bool:
    '''Remove an installed bundle, elevating if the store is not user-writable.

    Bundles seeded by the macOS PKG (and by a system-wide Windows install) are
    owned by root/Administrators, so a plain ``rmtree`` raises ``PermissionError``;
    we then re-run just the removal behind the platform's authorization prompt.

    Returns True when the bundle is gone. Raises :class:`ElevationDenied` if the
    user declines the prompt, and returns False for a removal that failed for
    any other reason (so the caller can distinguish "you said no" from "it did
    not work").
    '''
    if not Path(location).exists():
        return True
    try:
        shutil.rmtree(location)
        return True
    except (OSError, PermissionError):
        pass

    if not allow_elevation or not _is_managed_bundle(location):
        return False
    try:
        if paths.IS_WINDOWS:
            _remove_elevated_windows(Path(location))
        elif paths.IS_MAC:
            _run_elevated_macos(
                f'/bin/rm -rf {shlex.quote(str(location))}',
                'BabelBrain needs administrator privileges to remove this version.',
                Path(location), 'removed')
        else:
            return False
    except ElevationDenied:
        raise
    except InstallError:
        return False
    return not Path(location).exists()
