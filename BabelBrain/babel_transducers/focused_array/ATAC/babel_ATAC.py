# This Python file uses the following encoding: utf-8
import os

from babel_transducers.focused_array.focused_array_with_cone_form import FocusedArrayWithConeForm
from babel_transducers.transducer_templates.babel_focused_array_tx import FocusedArrayTx
from Utils.paths import resource_path


class ATAC(FocusedArrayTx):
    def __init__(self, parent=None, MainApp=None):
        config_file = os.path.join(resource_path(__file__), "default.yaml")
        super().__init__(parent,MainApp,config_file,FocusedArrayWithConeForm)
