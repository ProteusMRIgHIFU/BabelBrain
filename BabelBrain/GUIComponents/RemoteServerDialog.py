"""
Dialogs to manage BabelBrain remote-server configurations (client/server mode).

RemoteServerManagerDialog lists the saved servers and lets the user add, edit,
remove and test them; RemoteServerEditDialog is the per-server form. Persistence
lives in RemoteServers.py. Built programmatically (no .ui) so it stays small.
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                               QListWidgetItem, QPushButton, QLabel, QLineEdit,
                               QSpinBox, QFormLayout, QMessageBox, QDialogButtonBox,
                               QCheckBox, QFileDialog, QWidget)
from PySide6.QtCore import QCoreApplication, Qt

import RemoteServers


def _style(dlg):
    try:
        from GUIComponents.AppStyle import app_qss, apply_native_spinbox_style
        dlg.setStyleSheet(app_qss(dlg))
        apply_native_spinbox_style(dlg)
    except Exception:
        pass


def _file_row(edit):
    """A line-edit + Browse button for choosing a PEM file path."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit)
    browse = QPushButton(QCoreApplication.translate("BabelBrain", "Browse…"))

    def _pick():
        path, _ = QFileDialog.getOpenFileName(
            row, "Select PEM file", edit.text().strip() or "",
            "Certificates (*.pem *.crt *.cer *.key);;All files (*)")
        if path:
            edit.setText(path)
    browse.clicked.connect(_pick)
    h.addWidget(browse)
    return row


class RemoteServerEditDialog(QDialog):
    """Add/edit one server: name, host, port, optional bearer token, and optional
    TLS (HTTPS) settings — CA bundle for a private/self-signed cert, client cert +
    key for mutual TLS, and an insecure (skip-verification) escape hatch."""

    def __init__(self, parent=None, server=None):
        super().__init__(parent)
        self.setWindowTitle(QCoreApplication.translate("BabelBrain", "Remote server"))
        server = server or {}
        form = QFormLayout()
        self.nameEdit = QLineEdit(server.get('name', ''))
        self.hostEdit = QLineEdit(server.get('host', '127.0.0.1'))
        self.portSpin = QSpinBox()
        self.portSpin.setRange(1, 65535)
        self.portSpin.setValue(int(server.get('port', 8760) or 8760))
        self.tokenEdit = QLineEdit(server.get('token') or '')
        self.tokenEdit.setEchoMode(QLineEdit.Password)
        self.tokenEdit.setPlaceholderText(QCoreApplication.translate("BabelBrain", "(leave empty if the server needs no token)"))
        form.addRow("Name:", self.nameEdit)
        form.addRow("Host / IP:", self.hostEdit)
        form.addRow("Port:", self.portSpin)
        form.addRow("Token:", self.tokenEdit)

        # -- TLS / HTTPS --
        self.httpsCheck = QCheckBox(QCoreApplication.translate("BabelBrain", "Use HTTPS (TLS)"))
        self.httpsCheck.setChecked(bool(server.get('https', False)))
        self.cafileEdit = QLineEdit(server.get('cafile') or '')
        self.cafileEdit.setPlaceholderText(QCoreApplication.translate("BabelBrain", "(CA bundle — only for a private / self-signed cert)"))
        self.clientCertEdit = QLineEdit(server.get('client_cert') or '')
        self.clientCertEdit.setPlaceholderText(QCoreApplication.translate("BabelBrain", "(client cert PEM — only for mutual TLS)"))
        self.clientKeyEdit = QLineEdit(server.get('client_key') or '')
        self.clientKeyEdit.setPlaceholderText(QCoreApplication.translate("BabelBrain", "(client key PEM — only for mutual TLS)"))
        self.insecureCheck = QCheckBox(QCoreApplication.translate("BabelBrain", "Skip certificate verification (testing only — unsafe)"))
        self.insecureCheck.setChecked(bool(server.get('insecure', False)))
        form.addRow(self.httpsCheck)
        form.addRow("CA file:", _file_row(self.cafileEdit))
        form.addRow("Client cert:", _file_row(self.clientCertEdit))
        form.addRow("Client key:", _file_row(self.clientKeyEdit))
        form.addRow(self.insecureCheck)

        # Enable the TLS fields only when HTTPS is on.
        self._tls_rows = (self.cafileEdit, self.clientCertEdit,
                          self.clientKeyEdit, self.insecureCheck)
        self.httpsCheck.toggled.connect(self._sync_tls_enabled)
        self._sync_tls_enabled(self.httpsCheck.isChecked())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        _style(self)

    def _sync_tls_enabled(self, on):
        for w in self._tls_rows:
            w.setEnabled(bool(on))

    def _accept(self):
        if not self.nameEdit.text().strip():
            QMessageBox.warning(self, "Missing name", "Please give the server a name.")
            return
        if not self.hostEdit.text().strip():
            QMessageBox.warning(self, "Missing host", "Please provide a host or IP.")
            return
        if (self.httpsCheck.isChecked() and self.clientCertEdit.text().strip()
                and not self.clientKeyEdit.text().strip()):
            QMessageBox.warning(self, "Missing client key",
                                "Mutual TLS needs both a client cert and a client key.")
            return
        self.accept()

    def server(self):
        https = self.httpsCheck.isChecked()
        return {'name': self.nameEdit.text().strip(),
                'host': self.hostEdit.text().strip(),
                'port': int(self.portSpin.value()),
                'token': self.tokenEdit.text().strip() or None,
                'https': https,
                'cafile': (self.cafileEdit.text().strip() or None) if https else None,
                'client_cert': (self.clientCertEdit.text().strip() or None) if https else None,
                'client_key': (self.clientKeyEdit.text().strip() or None) if https else None,
                'insecure': self.insecureCheck.isChecked() if https else False}


class RemoteServerManagerDialog(QDialog):
    """List + add/edit/remove/test saved remote servers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(QCoreApplication.translate("BabelBrain", "Add / remove remote server"))
        self.resize(460, 320)

        self.listWidget = QListWidget()
        self.listWidget.itemDoubleClicked.connect(lambda *_: self._edit())

        addBtn = QPushButton(QCoreApplication.translate("BabelBrain", "Add…"))
        editBtn = QPushButton(QCoreApplication.translate("BabelBrain", "Edit…"))
        removeBtn = QPushButton(QCoreApplication.translate("BabelBrain", "Remove"))
        testBtn = QPushButton(QCoreApplication.translate("BabelBrain", "Test"))
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
        layout.addWidget(QLabel(QCoreApplication.translate("BabelBrain", "Servers that can run BabelBrain simulations for this client:")))
        layout.addLayout(row)
        layout.addWidget(closeBox)
        _style(self)
        self._reload()

    def _reload(self, select_name=None):
        self.listWidget.clear()
        for s in RemoteServers.load_servers():
            scheme = "https" if s.get('https') else "http"
            item = QListWidgetItem("%s   —   %s://%s:%d"
                                   % (s['name'], scheme, s['host'], s['port']))
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
