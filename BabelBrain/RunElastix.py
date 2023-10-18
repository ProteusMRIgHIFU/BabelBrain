import itk

import argparse
import sys
from pathlib import Path
import os
import platform

_IS_MAC = platform.system() == 'Darwin'


def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir


def RigidRegistration(reference,moving,finalname):
    print('Running elastix')
    parameter_object = itk.ParameterObject.New()
    rpath = os.path.join(resource_path(), "rigid.txt")
    parameter_object.AddParameterFile(rpath)
    print(parameter_object)
    fixed_image = itk.imread(reference)
    moving_image = itk.imread(moving)
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=True)
    itk.imwrite(result_image,finalname,compression=True)
    print('End Running elastix, file saved to ',finalname)


if __name__ == "__main__":

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    
    parser = MyParser(prog='RunElastix', usage='python %(prog)s.py [options]',description='Rigid coregistration with itk-elastix',  
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('reference', type=str, nargs='+',help='Input reference image')
    parser.add_argument('moving', type=str, nargs='+',help='Moving reference image')
    parser.add_argument('finalname', type=str, nargs='+',help='Filename of output image')
    
    args = parser.parse_args()

    reference=args.reference[0]
    moving=args.moving[0]
    finalname=args.finalname[0]
    RigidRegistration(reference,moving,finalname)
