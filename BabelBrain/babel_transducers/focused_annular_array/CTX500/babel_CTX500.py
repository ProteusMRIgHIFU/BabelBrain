# This Python file uses the following encoding: utf-8

import os

from babel_transducers.focused_annular_array.focused_annular_array_tx_form import FocusedAnnularArrayTxForm
from babel_transducers.transducer_templates.babel_focused_annular_array_tx import FocusedAnnularArrayTx
from Utils.paths import resource_path


class CTX500(FocusedAnnularArrayTx):
    def __init__(self, parent=None, MainApp=None):
        config_file = os.path.join(resource_path(__file__), "default.yaml")
        super().__init__(parent,MainApp,config_file,FocusedAnnularArrayTxForm)
