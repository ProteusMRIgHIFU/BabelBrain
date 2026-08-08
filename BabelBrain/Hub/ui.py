'''
The version picker dialog.

Shows installed versions (builtin + downloaded, tagged by scope) and, from the
curated manifest, versions available to download. The user launches an installed
version, or downloads one first. Install target (just-me vs all-users) is always
an explicit choice, and any elevation failure returns the user to that choice
rather than silently redirecting — see the install handler.
'''
from __future__ import annotations

from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import HUB_VERSION, installer, manifest as manifest_mod, state as state_mod, versions as versions_mod

# Roles for stashing data on list rows.
_ROLE_KIND = Qt.UserRole            # 'installed' | 'available'
_ROLE_PAYLOAD = Qt.UserRole + 1     # VersionInfo | CatalogEntry


def _version_tuple(v: str | None):
    if not v:
        return ()
    parts = []
    for chunk in v.split('.'):
        num = ''.join(ch for ch in chunk if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def hub_update_available(hub: manifest_mod.HubUpdate) -> bool:
    return bool(hub.latest) and _version_tuple(hub.latest) > _version_tuple(HUB_VERSION)


class _DownloadWorker(QThread):
    progress = Signal(int, int)          # done, total
    finished_ok = Signal(str)            # installed path
    failed = Signal(str, str)            # error kind, message

    def __init__(self, entry, scope, allow_elevation=True):
        super().__init__()
        self._entry = entry
        self._scope = scope
        self._allow_elevation = allow_elevation

    def run(self):
        try:
            path = installer.install(
                self._entry, self._scope,
                progress=lambda d, t: self.progress.emit(d, t),
                allow_elevation=self._allow_elevation,
            )
            self.finished_ok.emit(str(path))
        except installer.ElevationRequired as e:
            self.failed.emit('elevation_required', str(e))
        except installer.ElevationDenied as e:
            self.failed.emit('elevation_denied', str(e))
        except Exception as e:  # noqa: BLE001 - surface any failure to the user
            self.failed.emit('error', str(e))


class HubWindow(QDialog):
    def __init__(self, state: state_mod.HubState, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle('BabelBrain Version Selector')
        self.resize(560, 460)
        self._state = state
        self._manifest: manifest_mod.Manifest | None = None
        self._manifest_error: str | None = None
        self._selected_version: versions_mod.VersionInfo | None = None
        self._launch_requested = False

        layout = QVBoxLayout(self)

        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        self._banner.setStyleSheet('padding:6px; background:#664; color:white; border-radius:4px;')
        layout.addWidget(self._banner)

        layout.addWidget(QLabel('Installed versions and available downloads:'))
        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(lambda *_: self._set_default())
        layout.addWidget(self._list, 1)

        self._prerelease_cb = QCheckBox('Show pre-releases')
        self._prerelease_cb.setChecked(state.show_prereleases)
        self._prerelease_cb.toggled.connect(self._on_prerelease_toggled)
        layout.addWidget(self._prerelease_cb)

        buttons = QHBoxLayout()
        self._setdefaul_btn = QPushButton('Selected as default')
        self._setdefaul_btn.setDefault(True)
        self._setdefaul_btn.clicked.connect(self._set_default)
        self._launch_btn = QPushButton('Selected as default and launch')
        self._launch_btn.clicked.connect(self._on_launch)
        self._download_btn = QPushButton('Download && install')
        self._download_btn.clicked.connect(self._on_download)
        self._uninstall_btn = QPushButton('Remove')
        self._uninstall_btn.clicked.connect(self._on_uninstall)
        self._quit_btn = QPushButton('Quit')
        self._quit_btn.clicked.connect(self.reject)
        buttons.addWidget(self._setdefaul_btn)
        buttons.addWidget(self._launch_btn)
        buttons.addWidget(self._download_btn)
        buttons.addWidget(self._uninstall_btn)
        buttons.addStretch(1)
        buttons.addWidget(self._quit_btn)
        layout.addLayout(buttons)

        self._refresh_all()

    # -- data / refresh -----------------------------------------------------
    def _load_manifest(self):
        try:
            self._manifest = manifest_mod.fetch()
            self._manifest_error = None
        except Exception as e:  # noqa: BLE001 - offline is fine, just no downloads
            self._manifest = None
            self._manifest_error = str(e)
            # Also print so `--` terminal launches leave a diagnosable trace.
            print(f'BabelBrain Hub: could not load version catalog: {e}')

    def _refresh_all(self):
        if self._manifest is None:
            self._load_manifest()
        self._populate()
        self._update_banner()

    def _populate(self):
        self._list.clear()
        installed = versions_mod.discover()
        installed_ids = {vi.build_id for vi in installed}

        for vi in installed:
            item = QListWidgetItem(f'{vi.display_name}   —   {vi.scope_label}')
            item.setData(_ROLE_KIND, 'installed')
            item.setData(_ROLE_PAYLOAD, vi)
            self._list.addItem(item)

        if self._manifest is not None:
            avail = manifest_mod.visible_entries(self._manifest, self._prerelease_cb.isChecked())
            for entry in avail:
                if entry.build_id in installed_ids:
                    continue
                tag = 'recommended' if entry.recommended else entry.channel
                item = QListWidgetItem(f'{entry.version}   —   available to download ({tag})')
                item.setData(_ROLE_KIND, 'available')
                item.setData(_ROLE_PAYLOAD, entry)
                self._list.addItem(item)

        # Preselect the current selection, else the first row.
        target_row = 0
        if self._state.current_build_id:
            for row in range(self._list.count()):
                it = self._list.item(row)
                if it.data(_ROLE_KIND) == 'installed' and \
                        it.data(_ROLE_PAYLOAD).build_id == self._state.current_build_id:
                    target_row = row
                    break
        if self._list.count():
            self._list.setCurrentRow(target_row)

    def _update_banner(self):
        if self._manifest is None:
            # Catalog unreachable: say so instead of showing an empty list with
            # no explanation. Installed versions are still runnable.
            msg = ('Could not load the online version catalog, so no downloads '
                   'are listed — only installed versions are available.')
            if self._manifest_error:
                msg += f'\n({self._manifest_error})'
            self._banner.setText(msg)
            self._banner.setVisible(True)
        elif hub_update_available(self._manifest.hub):
            self._banner.setText(
                f'A newer BabelBrain launcher ({self._manifest.hub.latest}) is available. '
                f'Update the launcher to unlock the latest BabelBrain versions.')
            self._banner.setVisible(True)
        else:
            self._banner.setVisible(False)

    # -- selection state ----------------------------------------------------
    def _current(self):
        items = self._list.selectedItems()
        if not items:
            return None, None
        it = items[0]
        return it.data(_ROLE_KIND), it.data(_ROLE_PAYLOAD)

    def _on_selection_changed(self):
        kind, _ = self._current()
        self._launch_btn.setEnabled(kind == 'installed')
        self._download_btn.setEnabled(kind == 'available')
        self._uninstall_btn.setEnabled(
            kind == 'installed' and self._current()[1].scope in ('user', 'shared'))

    def _on_prerelease_toggled(self, checked: bool):
        self._state.show_prereleases = checked
        self._populate()

    # -- actions ------------------------------------------------------------
    def _on_launch(self):
        kind, payload = self._current()
        if kind != 'installed':
            return
        self._selected_version = payload
        self._launch_requested = True
        self._persist_choice(payload.build_id)
        self.accept()

    def _set_default(self):
        kind, payload = self._current()
        if kind != 'installed':
            return
        self._selected_version = payload
        self._launch_requested = True
        self._persist_choice(payload.build_id)

    def _persist_choice(self, build_id: str):
        # Launching a version makes it the current selection, so BabelBrain.app
        # then opens it directly.
        self._state.current_build_id = build_id
        self._state.show_prereleases = self._prerelease_cb.isChecked()
        if not state_mod.save(self._state):
            QMessageBox.warning(self, 'BabelBrain',
                                'Your selection could not be saved, so BabelBrain '
                                'may open a different version next time.')

    def _choose_scope(self) -> str | None:
        '''Explicit local/global choice. Returns 'user', 'shared', or None.'''
        box = QMessageBox(self)
        box.setWindowTitle('Install location')
        box.setText('Where should this version be installed?')
        box.setInformativeText(
            'Just for me — installs in your user profile, no administrator '
            'privileges needed.\n\n'
            'For all users — installs system-wide and may require administrator '
            'privileges.')
        me_btn = box.addButton('Just for me', QMessageBox.AcceptRole)
        all_btn = box.addButton('For all users', QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is me_btn:
            return 'user'
        if clicked is all_btn:
            return 'shared'
        return None

    def _on_download(self):
        kind, entry = self._current()
        if kind != 'available':
            return
        # Loop so an elevation failure returns the user to the location choice
        # instead of silently falling back to a different scope.
        while True:
            scope = self._choose_scope()
            if scope is None:
                return
            outcome = self._run_download(entry, scope)
            if outcome == 'retry':
                continue
            return

    def _run_download(self, entry, scope) -> str:
        '''Returns 'done' or 'retry' (user should re-pick the location).'''
        progress = QProgressDialog('Downloading…', 'Cancel', 0, 100, self)
        progress.setWindowTitle(f'Installing BabelBrain {entry.version}')
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)

        worker = _DownloadWorker(entry, scope)
        result = {'status': None, 'kind': None, 'msg': None}
        canceled = {'value': False}
        # Drive the worker with a nested event loop that quits on QThread.finished.
        # finished_ok/failed are emitted from run() BEFORE finished, so they are
        # always delivered before we inspect `result` — avoiding the race where a
        # busy-wait exits (thread no longer "running") before the success signal
        # is processed, which produced a spurious "Unknown error".
        loop = QEventLoop()

        def on_progress(done, total):
            if total > 0:
                progress.setMaximum(total)
                progress.setValue(done)
                progress.setLabelText(f'Downloading… {done // (1024*1024)} / {total // (1024*1024)} MB')
            else:
                progress.setMaximum(0)  # indeterminate

        def on_ok(_path):
            result['status'] = 'ok'

        def on_fail(kind, msg):
            result['status'] = 'fail'
            result['kind'] = kind
            result['msg'] = msg

        def on_cancel():
            canceled['value'] = True
            worker.terminate()

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        worker.finished.connect(loop.quit)
        progress.canceled.connect(on_cancel)
        worker.start()
        loop.exec()
        worker.wait()          # ensure the thread has fully finished
        progress.close()

        if canceled['value'] and result['status'] is None:
            return 'done'      # user cancelled; not an error

        if result['status'] == 'ok':
            self._populate()
            return 'done'

        # Failure — surface the real error, never a silent redirect.
        if result['kind'] in ('elevation_required', 'elevation_denied'):
            QMessageBox.warning(self, 'Administrator privileges needed', result['msg'] or '')
            return 'retry'          # send the user back to the location choice
        QMessageBox.critical(self, 'Install failed', result['msg'] or 'Unknown error.')
        return 'done'

    def _on_uninstall(self):
        kind, vi = self._current()
        if kind != 'installed' or vi.scope not in ('user', 'shared'):
            return
        if QMessageBox.question(
                self, 'Remove version',
                f'Remove {vi.display_name} ({vi.scope_label})?') != QMessageBox.Yes:
            return
        if not installer.uninstall(vi.location):
            QMessageBox.warning(
                self, 'Could not remove',
                'This version could not be removed — it may be installed for all '
                'users and require administrator privileges.')
        self._populate()

    # -- result -------------------------------------------------------------
    def selected_version(self) -> versions_mod.VersionInfo | None:
        return self._selected_version if self._launch_requested else None
