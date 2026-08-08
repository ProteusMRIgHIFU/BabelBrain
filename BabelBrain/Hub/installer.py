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
'''
from __future__ import annotations

import hashlib
import json
import os
import platform
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
    def __init__(self, target: Path):
        super().__init__(f'The request for administrator privileges was declined; '
                         f'nothing was installed at {target}.')
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


def _place_elevated_windows(bundle_root: Path, target_dir: Path):
    '''Windows: perform the final move under a UAC elevation prompt by
    relaunching the Hub with a hidden worker subcommand.'''
    import ctypes

    spec = {'src': str(bundle_root), 'dst': str(target_dir)}
    spec_file = Path(tempfile.mkdtemp(prefix='bbhub_')) / 'install_job.json'
    result_file = spec_file.with_suffix('.result')
    spec['result'] = str(result_file)
    spec_file.write_text(json.dumps(spec))

    argv = _hub_relaunch_argv()
    file = argv[0]
    params = subprocess.list2cmdline(argv[1:] + ['--install-worker', str(spec_file)])

    # ShellExecuteW returns >32 on success; SE_ERR_ACCESSDENIED (5) when the
    # user cancels the UAC prompt.
    rc = ctypes.windll.shell32.ShellExecuteW(None, 'runas', file, params, None, 0)
    if rc <= 32:
        raise ElevationDenied(target_dir)

    # Wait for the elevated worker to finish and report through result_file.
    import time
    for _ in range(600):            # up to ~60s
        if result_file.is_file():
            break
        time.sleep(0.1)
    if not result_file.is_file():
        raise InstallError('The elevated install did not report completion.')
    status = json.loads(result_file.read_text())
    if not status.get('ok'):
        raise InstallError(status.get('error', 'The elevated install failed.'))


def run_install_worker(spec_file: str) -> int:
    '''Entry point for the elevated worker process (invoked via
    ``--install-worker``). Performs only the privileged move and reports back.'''
    spec = json.loads(Path(spec_file).read_text())
    result_file = Path(spec['result'])
    try:
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
        # macOS /Users/Shared is normally writable (so we should not get here);
        # Linux /opt would need root. We do not silently fall back.
        raise ElevationRequired(scope, target_root)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def uninstall(location: Path) -> bool:
    '''Remove an installed bundle. Returns False on permission failure so the
    caller can surface it (e.g. a shared bundle needing elevation).'''
    try:
        shutil.rmtree(location)
        return True
    except (OSError, PermissionError):
        return False
