import shutil

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QListWidget, QMessageBox, QPushButton, QVBoxLayout)

from CreateTransducers.transducer_creator import CUSTOM_TRANSDUCERS_FOLDER
from GUIComponents.custom_transducer_dialog import (CUSTOM_TRANSDUCER_OPTION,
                                                    CUSTOM_TRANSDUCER_PREFIX,
                                                    custom_transducer_display_name)


class CustomTransducerManagerDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_dialog = parent
        self.setWindowTitle("Manage Custom Transducers")
        self.resize(460, 320)

        self.TransducerListWidget = QListWidget(self)
        self.TransducerListWidget.setAlternatingRowColors(True)

        self.AddpushButton = QPushButton("Add…", self)
        self.DeletepushButton = QPushButton("Remove", self)
        self.MoveUppushButton = QPushButton("Move up", self)
        self.MoveDownpushButton = QPushButton("Move down", self)

        self.ButtonLayout = QVBoxLayout()
        for button in (self.AddpushButton, self.DeletepushButton,
                       self.MoveUppushButton, self.MoveDownpushButton):
            self.ButtonLayout.addWidget(button)
        self.ButtonLayout.addStretch(1)

        self.RowLayout = QHBoxLayout()
        self.RowLayout.addWidget(self.TransducerListWidget, 1)
        self.RowLayout.addLayout(self.ButtonLayout)

        self.CloseButtonBox = QDialogButtonBox(QDialogButtonBox.Close)
        self.CloseButtonBox.rejected.connect(self.accept)
        self.CloseButtonBox.accepted.connect(self.accept)

        self.MainLayout = QVBoxLayout(self)
        self.MainLayout.addWidget(QLabel("Custom transducers available in BabelBrain:"))
        self.MainLayout.addLayout(self.RowLayout)
        self.MainLayout.addWidget(self.CloseButtonBox)

        try:
            from GUIComponents.AppStyle import app_qss, apply_native_spinbox_style
            self.setStyleSheet(app_qss(self))
            apply_native_spinbox_style(self)
        except Exception:
            pass

        self.AddpushButton.clicked.connect(self.AddTransducer)
        self.DeletepushButton.clicked.connect(self.DeleteTransducer)
        self.MoveUppushButton.clicked.connect(lambda: self.MoveTransducer(-1))
        self.MoveDownpushButton.clicked.connect(lambda: self.MoveTransducer(1))
        self.TransducerListWidget.itemSelectionChanged.connect(self.UpdateButtons)

        self.RefreshTransducerList()

    def RefreshTransducerList(self, selected_tx=None):
        self.TransducerListWidget.clear()

        for index in range(self.parent_dialog.ui.TransducerTypecomboBox.count()):
            item_text = self.parent_dialog.ui.TransducerTypecomboBox.itemText(index)

            if item_text.startswith(CUSTOM_TRANSDUCER_PREFIX):
                tx_name = item_text.removeprefix(CUSTOM_TRANSDUCER_PREFIX)
                self.TransducerListWidget.addItem(tx_name)

        if self.TransducerListWidget.count() > 0:
            selected_row = 0

            if selected_tx is not None:
                matching_items = self.TransducerListWidget.findItems(
                    selected_tx,
                    Qt.MatchFlag.MatchExactly,
                )

                if matching_items:
                    selected_row = self.TransducerListWidget.row(matching_items[0])

            self.TransducerListWidget.setCurrentRow(selected_row)

        self.UpdateButtons()

    @Slot()
    def AddTransducer(self):
        add_tx_index = self.parent_dialog.ui.TransducerTypecomboBox.findText(
            CUSTOM_TRANSDUCER_OPTION
        )

        if add_tx_index >= 0:
            self.parent_dialog.ui.TransducerTypecomboBox.setCurrentIndex(add_tx_index)

        current_tx = self.parent_dialog.ui.TransducerTypecomboBox.currentText()

        if current_tx.startswith(CUSTOM_TRANSDUCER_PREFIX):
            current_tx = current_tx.removeprefix(CUSTOM_TRANSDUCER_PREFIX)
        else:
            current_tx = None

        self.RefreshTransducerList(current_tx)

    @Slot()
    def DeleteTransducer(self):
        selected_item = self.TransducerListWidget.currentItem()

        if selected_item is None:
            return

        tx_name = selected_item.text()
        response = QMessageBox.question(
            self,
            "Delete Custom Transducer",
            f"Are you sure you want to delete {tx_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if response != QMessageBox.StandardButton.Yes:
            return

        tx_folder = CUSTOM_TRANSDUCERS_FOLDER / f"Babel_{tx_name}"
        current_tx = self.parent_dialog.ui.TransducerTypecomboBox.currentText()

        try:
            shutil.rmtree(tx_folder)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Unable to delete transducer",
                str(error),
            )
            return

        self.parent_dialog.AddCustomTransducersToList()

        current_tx_index = self.parent_dialog.ui.TransducerTypecomboBox.findText(current_tx)

        if current_tx_index >= 0:
            self.parent_dialog.ui.TransducerTypecomboBox.setCurrentIndex(current_tx_index)
        else:
            self.parent_dialog.ui.TransducerTypecomboBox.setCurrentIndex(0)

        self.parent_dialog._previous_transducer_index = (
            self.parent_dialog.ui.TransducerTypecomboBox.currentIndex()
        )
        self.RefreshTransducerList()

    def MoveTransducer(self, offset):
        current_row = self.TransducerListWidget.currentRow()
        new_row = current_row + offset

        if current_row < 0 or new_row < 0 or new_row >= self.TransducerListWidget.count():
            return

        selected_item = self.TransducerListWidget.takeItem(current_row)
        self.TransducerListWidget.insertItem(new_row, selected_item)
        self.TransducerListWidget.setCurrentRow(new_row)
        self.ApplyTransducerOrder()

    def ApplyTransducerOrder(self):
        combo_box = self.parent_dialog.ui.TransducerTypecomboBox
        current_tx = combo_box.currentText()
        combo_box.blockSignals(True)

        try:
            for index in range(combo_box.count() - 1, -1, -1):
                if combo_box.itemText(index).startswith(CUSTOM_TRANSDUCER_PREFIX):
                    combo_box.removeItem(index)

            insert_index = combo_box.findText(CUSTOM_TRANSDUCER_OPTION)

            for index in range(self.TransducerListWidget.count()):
                tx_name = self.TransducerListWidget.item(index).text()
                combo_box.insertItem(
                    insert_index,
                    custom_transducer_display_name(tx_name),
                )
                insert_index += 1

            current_tx_index = combo_box.findText(current_tx)

            if current_tx_index >= 0:
                combo_box.setCurrentIndex(current_tx_index)
        finally:
            combo_box.blockSignals(False)

        self.parent_dialog._previous_transducer_index = combo_box.currentIndex()
        self.UpdateButtons()

    @Slot()
    def UpdateButtons(self):
        current_row = self.TransducerListWidget.currentRow()
        item_count = self.TransducerListWidget.count()
        self.DeletepushButton.setEnabled(current_row >= 0)
        self.MoveUppushButton.setEnabled(current_row > 0)
        self.MoveDownpushButton.setEnabled(
            current_row >= 0 and current_row < item_count - 1
        )