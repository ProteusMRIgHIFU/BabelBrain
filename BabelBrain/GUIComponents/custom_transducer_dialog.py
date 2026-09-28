# This Python file uses the following encoding: utf-8
import os

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

CUSTOM_TRANSDUCER_OPTION = 'Add Custom Transducer ...'
CUSTOM_TRANSDUCER_PREFIX = 'Custom: '


class CustomTransducerDialog(QDialog):
    def __init__(self, parent=None, config_file=''):
        super().__init__(parent)
        self.setWindowTitle('Add Custom Transducer')
        self.config_file = config_file

        details_label = QLabel('For details on transducer config file formatting, see ...')
        details_label.setWordWrap(True)

        config_label = QLabel('Config File')
        self.config_line_edit = QLineEdit(config_file)
        self.config_line_edit.setMinimumWidth(520)

        select_button = QPushButton('Select Config file')
        select_button.clicked.connect(self.SelectConfigFile)

        create_button = QPushButton('Create Transducer')
        create_button.clicked.connect(self.CreateTransducer)

        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)

        file_layout = QGridLayout()
        file_layout.addWidget(config_label, 0, 0)
        file_layout.addWidget(select_button, 0, 1)
        file_layout.addWidget(self.config_line_edit, 0, 2)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(create_button)
        button_layout.addWidget(cancel_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(details_label)
        main_layout.addLayout(file_layout)
        main_layout.addLayout(button_layout)
        self.resize(760, 170)

    @Slot()
    def SelectConfigFile(self):
        curfile = self.config_line_edit.text()
        bdir = os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        config_file = QFileDialog.getOpenFileName(
            self,
            'Select transducer config file',
            bdir,
            'YAML (*.yaml *.yml)',
        )[0]
        if len(config_file) > 0:
            self.config_line_edit.setText(config_file)
            self.config_line_edit.setCursorPosition(len(config_file))

    @Slot()
    def CreateTransducer(self):
        config_file = self.config_line_edit.text()
        if not os.path.isfile(config_file):
            msgBox = QMessageBox(self)
            msgBox.setText('Please indicate a valid transducer config file')
            msgBox.exec()
            return
        self.config_file = config_file
        self.accept()


def custom_transducer_display_name(config_file):
    return CUSTOM_TRANSDUCER_PREFIX + os.path.basename(config_file)


def is_custom_transducer_display_name(transducer_name):
    return transducer_name.startswith(CUSTOM_TRANSDUCER_PREFIX)
