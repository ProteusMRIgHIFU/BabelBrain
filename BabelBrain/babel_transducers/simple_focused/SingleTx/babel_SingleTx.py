import os

from babel_transducers.simple_focused.SingleTx.SingleTx_form import SingleTxForm
from babel_transducers.transducer_templates.babel_simple_focused_tx import SimpleFocusedTx
from Utils.paths import resource_path


class SingleTx(SimpleFocusedTx):
    def __init__(self, parent=None, MainApp=None):
        config_file = os.path.join(resource_path(__file__), "default.yaml")
        super().__init__(parent,MainApp,config_file,SingleTxForm)