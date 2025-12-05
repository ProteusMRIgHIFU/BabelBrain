# This Python file uses the following encoding: utf-8
from multiprocessing import Process,Queue
import os
from pathlib import Path
import sys
import platform
import yaml
from Babel_H317.Babel_H317 import H317


_IS_MAC = platform.system() == 'Darwin'
def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_R15148'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class H301(H317):
    def DefaultConfig(self):
        #Specific parameters for the H301 - configured via a yaml
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("H301 configuration:")
        print(config)
        self.Config=config
