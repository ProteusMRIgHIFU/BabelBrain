"""
Dialogs to manage BabelBrain remote-server configurations (client/server mode).

RemoteServerManagerDialog lists the saved servers and lets the user add, edit,
remove and test them; RemoteServerEditDialog is the per-server form. Persistence
lives in RemoteServers.py. Built programmatically (no .ui) so it stays small.
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                               QListWidgetItem, QPushButton, QLabel, QLineEdit,
                               QSpinBox, QFormLayout, QMessageBox, QDialogButtonBox)
from PySide6.QtCore import Qt

import RemoteServers


def _style(dlg):
    try:
        from GUIComponents.AppStyle import app_qss, apply_native_spinbox_style
        dlg.setStyleSheet(app_qss(dlg))
        apply_native_spinbox_style(dlg)
    except Exception:
        pass


class RemoteServerEditDialog(QDialog):
    """Add/edit one server: name, host, port, optional bearer token."""

    def __init__(self, parent=None, server=None):
        super().__init__(parent)
        self.setWindowTitle("Remote server")
        server = server or {}
        form = QFormLayout()
        self.nameEdit = QLineEdit(server.get('name', ''))
        self.hostEdit = QLineEdit(server.get('host', '127.0.0.1'))
        self.portSpin = QSpinBox()
        self.portSpin.setRange(1, 65535)
        self.portSpin.setValue(int(server.get('port', 8760) or 8760))
        self.tokenEdit = QLineEdit(server.get('token') or '')
        self.tokenEdit.setEchoMode(QLineEdit.Password)
        self.tokenEdit.setPlaceholderText("(leave empty if the server needs no token)")
        form.addRow("Name:", self.nameEdit)
        form.addRow("Host / IP:", self.hostEdit)
        form.addRow("Port:", self.portSpin)
        form.addRow("Token:", self.tokenEdit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        _style(self)

    def _accept(self):
        if not self.nameEdit.text().strip():
            QMessageBox.warning(self, "Missing name", "Please give the server a name.")
            return
        if not self.hostEdit.text().strip():
            QMessageBox.warning(self, "Missing host", "Please provide a host or IP.")
            return
        self.accept()

    def server(self):
        return {'name': self.nameEdit.text().strip(),
                'host': self.hostEdit.text().strip(),
                'port': int(self.portSpin.value()),
                'token': self.tokenEdit.text().strip() or None}


class RemoteServerManagerDialog(QDialog):
    """List + add/edit/remove/test saved remote servers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add / remove remote server")
        self.resize(460, 320)

        self.listWidget = QListWidget()
        self.listWidget.itemDoubleClicked.connect(lambda *_: self._edit())

        addBtn = QPushButton("Add…")
        editBtn = QPushButton("Edit…")
        removeBtn = QPushButton("Remove")
        testBtn = QPushButton("Test")
        addBtn.clicked.connect(self._add)
        editBtn.clicked.connect(self._edit)
        removeBtn.clicked.connect(self._remove)
        testBtn.clicked.connect(self._test)

        btnCol = QVBoxLayout()
        for b in (addBtn, editBtn, removeBtn, testBtn):
            btnCol.addWidget(b)
        btnCol.addStretch(1)

        row = QHBoxLayout()
        row.addWidget(self.listWidget, 1)
        row.addLayout(btnCol)

        closeBox = QDialogButtonBox(QDialogButtonBox.Close)
        closeBox.rejected.connect(self.accept)
        closeBox.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Servers that can run BabelBrain simulations for this client:"))
        layout.addLayout(row)
        layout.addWidget(closeBox)
        _style(self)
        self._reload()

    def _reload(self, select_name=None):
        self.listWidget.clear()
        for s in RemoteServers.load_servers():
            item = QListWidgetItem("%s   —   %s:%d" % (s['name'], s['host'], s['port']))
            item.setData(Qt.UserRole, s)
            self.listWidget.addItem(item)
            if select_name and s['name'] == select_name:
                self.listWidget.setCurrentItem(item)
        if select_name is None and self.listWidget.count():
            self.listWidget.setCurrentRow(0)

    def _selected(self):
        item = self.listWidget.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _add(self):
        dlg = RemoteServerEditDialog(self)
        if dlg.exec() == QDialog.Accepted:
            srv = dlg.server()
            RemoteServers.add_or_update(srv)
            self._reload(select_name=srv['name'])

    def _edit(self):
        srv = self._selected()
        if srv is None:
            return
        dlg = RemoteServerEditDialog(self, server=srv)
        if dlg.exec() == QDialog.Accepted:
            new = dlg.server()
            if new['name'] != srv['name']:
                RemoteServers.remove(srv['name'])     # renamed -> drop the old entry
            RemoteServers.add_or_update(new)
            self._reload(select_name=new['name'])

    def _remove(self):
        srv = self._selected()
        if srv is None:
            return
        if QMessageBox.question(self, "Remove server",
                                "Remove '%s'?" % srv['name']) == QMessageBox.Yes:
            RemoteServers.remove(srv['name'])
            self._reload()

    def _test(self):
        srv = self._selected()
        if srv is None:
            return
        ok, info = RemoteServers.test_connection(srv)
        if ok:
            caps = info.get('capabilities', {})
            txs = caps.get('transducers', [])
            feats = caps.get('features', [])
            QMessageBox.information(
                self, "Connection OK",
                "Connected to '%s' (%s).\n\nServer version: %s\nTransducers: %d\nFeatures: %s"
                % (srv['name'], RemoteServers.base_url(srv),
                   caps.get('server_version', '?'), len(txs),
                   ", ".join(feats) or "(none)"))
        else:
            QMessageBox.critical(self, "Connection failed",
                                 "Could not connect to '%s':\n\n%s" % (srv['name'], info))
