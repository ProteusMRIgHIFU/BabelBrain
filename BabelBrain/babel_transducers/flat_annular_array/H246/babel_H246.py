# This Python file uses the following encoding: utf-8

import os
import sys

from PySide6.QtWidgets import QApplication

from babel_transducers.flat_annular_array.H246.H246_form import H246Form
from babel_transducers.transducer_templates.babel_flat_annular_array_tx import FlatAnnularArrayTx
from Utils.paths import resource_path


class H246(FlatAnnularArrayTx):
    def __init__(self, parent=None, MainApp=None):
        config_file = os.path.join(resource_path(__file__), "default.yaml")
        super().__init__(parent,MainApp,config_file,H246Form)

if __name__ == "__main__":
    app = QApplication([])
    widget = H246()
    widget.show()
    sys.exit(app.exec_())
