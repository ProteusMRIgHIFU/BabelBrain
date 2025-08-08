# This Python file uses the following encoding: utf-8

import os
from pathlib import Path
import sys
import yaml


from _Babel_RingTx.Babel_RingTx import RingTx

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

class CTX250(RingTx):
     def DefaultConfig(self):
        #Specific parameters for the CTX250 - to be configured later via a yaml
        with open(os.path.join(resource_path(),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)

        self.Config=config

