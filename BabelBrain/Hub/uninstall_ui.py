'''
The uninstaller dialog.

Shows exactly what is on disk, with real sizes, before anything is removed —
the point of the whole feature is that the user can see there is nothing left
behind, so the UI never hides a path.

Two deliberate choices:

* **Settings are opt-in.** ``~/.config/BabelBrain`` is tiny, holds the user's
  own custom transducers, and is usually worth keeping across a reinstall. It
  gets its own checkbox (off by default) and its own line in the confirmation,
  so a genuinely complete wipe is still one click away.
* **Study data is never listed, because it is never touched.** The dialog says
  so explicitly: input images, the ``.ini`` files written next to a dataset and
  simulation outputs live in folders the user chose.
'''
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import paths, uninstall as uninstall_mod

_STUDY_DATA_NOTE = (
    'Your study data is never touched: input images, the .ini files BabelBrain '
    'writes next to a dataset, and simulation outputs stay where they are.')


class _ScanWorker(QThread):
    '''Walking multi-GB version stores to size them must not freeze the UI.'''
    done = Signal(object)            # list[FootprintItem]

    def run(self):
        try:
            items = uninstall_mod.footprint(include_settings=True)
        except Exception:            # noqa: BLE001 - an unreadable tree is not fatal
            items = []
        self.done.emit(items)


class _PurgeWorker(QThread):
    done = Signal(object)            # PurgeReport

    def __init__(self, include_settings: bool):
        super().__init__()
        self._include_settings = include_settings

    def run(self):
        try:
            report = uninstall_mod.purge(include_settings=self._include_settings)
        except Exception as e:       # noqa: BLE001 - report instead of crashing
            report = uninstall_mod.PurgeReport()
            report.failed.append((Path('.'), str(e)))
        self.done.emit(report)


class UninstallDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle('Uninstall BabelBrain')
        self.resize(640, 460)
        self._items: list[uninstall_mod.FootprintItem] = []
        self._completed = False

        layout = QVBoxLayout(self)

        intro = QLabel('This removes BabelBrain and every installed version '
                       'from this computer.')
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(['What', 'Size'])
        self._tree.setColumnWidth(0, 430)
        self._tree.setRootIsDecorated(True)
        layout.addWidget(self._tree, 1)

        self._settings_cb = QCheckBox('Also remove my settings and custom transducers')
        self._settings_cb.setChecked(False)          # keep by default: small, and the user's own work
        self._settings_cb.toggled.connect(self._refresh_tree)
        layout.addWidget(self._settings_cb)

        self._settings_hint = QLabel()
        self._settings_hint.setWordWrap(True)
        self._settings_hint.setStyleSheet('color: gray;')
        layout.addWidget(self._settings_hint)

        note = QLabel(_STUDY_DATA_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet('color: gray;')
        layout.addWidget(note)

        buttons = QHBoxLayout()
        self._uninstall_btn = QPushButton('Uninstall')
        self._uninstall_btn.clicked.connect(self._on_uninstall)
        self._uninstall_btn.setEnabled(False)        # until the scan finishes
        self._cancel_btn = QPushButton('Cancel')
        self._cancel_btn.setDefault(True)            # destructive action is never the default
        self._cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self._uninstall_btn)
        buttons.addWidget(self._cancel_btn)
        layout.addLayout(buttons)

        self._start_scan()

    # -- scanning -----------------------------------------------------------
    def _start_scan(self):
        placeholder = QTreeWidgetItem(['Looking for installed files…', ''])
        self._tree.addTopLevelItem(placeholder)
        self._scan = _ScanWorker()
        self._scan.done.connect(self._on_scanned)
        self._scan.start()

    def _on_scanned(self, items):
        self._items = items
        self._refresh_tree()
        self._uninstall_btn.setEnabled(True)

    def _refresh_tree(self):
        self._tree.clear()
        include_settings = self._settings_cb.isChecked()
        total = 0
        for cat, (title, desc, _default) in uninstall_mod.CATEGORIES.items():
            rows = [i for i in self._items if i.category == cat]
            kept = cat == 'settings' and not include_settings
            if not rows:
                continue
            size = sum(i.size for i in rows)
            if not kept:
                total += size
            head = QTreeWidgetItem([title + ('  — kept' if kept else ''),
                                    uninstall_mod.human_bytes(size)])
            head.setToolTip(0, desc)
            if kept:
                head.setForeground(0, Qt.gray)
            for i in rows:
                child = QTreeWidgetItem([str(i.path), i.human_size])
                if kept:
                    child.setForeground(0, Qt.gray)
                head.addChild(child)
            self._tree.addTopLevelItem(head)
            head.setExpanded(True)

        if self._tree.topLevelItemCount() == 0:
            self._tree.addTopLevelItem(QTreeWidgetItem(
                ['Nothing installed by BabelBrain was found.', '']))
            self._uninstall_btn.setEnabled(False)

        self._settings_hint.setText(
            'Settings will be kept, so a later reinstall finds your preferences '
            'and custom transducers. Tick the box above for a completely clean '
            'removal.' if not include_settings else
            'Settings, the remembered version choice and any transducers you '
            'created will be deleted.')
        self._uninstall_btn.setText(
            f'Uninstall  ({uninstall_mod.human_bytes(total)})' if total else 'Uninstall')

    # -- removal ------------------------------------------------------------
    def _confirm(self) -> bool:
        include_settings = self._settings_cb.isChecked()
        lines = ['Remove BabelBrain and all installed versions?', '']
        for cat, (title, _desc, _default) in uninstall_mod.CATEGORIES.items():
            rows = [i for i in self._items if i.category == cat]
            if not rows:
                continue
            if cat == 'settings' and not include_settings:
                lines.append(f'KEPT: {title}')
            else:
                lines.append(f'Remove: {title} '
                             f'({uninstall_mod.human_bytes(sum(i.size for i in rows))})')
        lines += ['', _STUDY_DATA_NOTE]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle('Confirm uninstall')
        box.setText('\n'.join(lines))
        if include_settings:
            # Second, explicit confirmation for the irreversible part: custom
            # transducers are user-authored content that lives nowhere else.
            box.setInformativeText(
                'Custom transducers you created will be deleted and cannot be '
                'recovered:\n' +
                '\n'.join(f'  {i.path}' for i in self._items if i.category == 'settings'))
        box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
        box.setDefaultButton(QMessageBox.Cancel)
        box.button(QMessageBox.Yes).setText('Uninstall')
        return box.exec() == QMessageBox.Yes

    def _on_uninstall(self):
        if not self._confirm():
            return
        progress = QProgressDialog('Removing BabelBrain…', '', 0, 0, self)
        progress.setWindowTitle('Uninstalling')
        progress.setCancelButton(None)          # no half-done uninstall
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Same nested-event-loop pattern as the download in ui.py: `done` is
        # emitted from run() before `finished`, so the report is always in hand
        # by the time the loop quits.
        worker = _PurgeWorker(self._settings_cb.isChecked())
        result: dict = {}
        loop = QEventLoop()
        worker.done.connect(lambda r: result.__setitem__('report', r))
        worker.finished.connect(loop.quit)
        worker.start()
        loop.exec()
        worker.wait()
        progress.close()
        self._show_result(result.get('report') or uninstall_mod.PurgeReport())

    def _show_result(self, report: uninstall_mod.PurgeReport):
        if report.elevation_denied:
            QMessageBox.warning(
                self, 'Uninstall incomplete',
                'Administrator privileges were declined, so the files installed '
                'for all users are still on this computer.\n\n'
                + '\n'.join(str(p) for p, _ in report.failed))
            self._rescan()
            return
        if report.failed:
            QMessageBox.warning(
                self, 'Uninstall incomplete',
                'Some items could not be removed. If BabelBrain is still '
                'running, quit it and try again.\n\n'
                + '\n'.join(f'{p} — {why}' for p, why in report.failed))
            self._rescan()
            return

        msg = 'BabelBrain has been removed from this computer.'
        if not self._settings_cb.isChecked():
            msg += ('\n\nYour settings and custom transducers were kept in '
                    f'{paths.config_dir()}.')
        if report.self_delete_scheduled:
            msg += '\n\nThis uninstaller deletes itself when you close this window.'
        QMessageBox.information(self, 'Uninstall complete', msg)
        self._completed = True
        self.accept()

    def _rescan(self):
        self._uninstall_btn.setEnabled(False)
        self._start_scan()

    @property
    def completed(self) -> bool:
        return self._completed
