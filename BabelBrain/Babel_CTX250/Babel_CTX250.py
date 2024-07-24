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
        bundle_dir = Path(sys._MEIPASS) / 'Babel_CTX250'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class CTX250(CTX500):
     def DefaultConfig(self):
        #Specific parameters for the CTX250 - to be configured later via a yaml
        with open(os.path.join(resource_path(),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)

        self.Config=config


if __name__ == "__main__":
    app = QApplication([])
    widget = CTX250()
    widget.show()
    sys.exit(app.exec_())
