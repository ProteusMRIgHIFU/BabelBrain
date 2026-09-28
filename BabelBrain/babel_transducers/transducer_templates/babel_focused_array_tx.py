from babel_transducers.transducer_templates.babel_base_phase_array_tx import BabelBasePhaseArray

class FocusedArrayTx(BabelBasePhaseArray):
    def __init__(self,parent=None,MainApp=None,tx_config_file=None,step_2_form=None):
        super().__init__(parent,MainApp,tx_config_file,step_2_form)