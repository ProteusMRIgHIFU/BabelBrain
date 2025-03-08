# This Python file uses the following encoding: utf-8

import os
from pathlib import Path
import sys
import yaml


from Babel_CTX500.Babel_CTX500 import CTX500

import platform
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_R15287'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class R15287(CTX500):
     def load_ui(self):
        super(R15287, self).load_ui()
        self.Widget.labelTPORange.setText('Range steering (mm)')
        self.Widget.labelTPODistance.setText('Steering from outplane (mm)')


     def DefaultConfig(self):
        #Specific parameters for the R15287 - to be configured later via a yaml
        with open(os.path.join(resource_path(),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)

        self.Config=config


if __name__ == "__main__":
    app = QApplication([])
    widget = R15287()
    widget.show()
    sys.exit(app.exec_())
