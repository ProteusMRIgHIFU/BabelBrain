from simnibs import mesh_io
import nibabel
import numpy as np

import argparse
import sys

if __name__ == "__main__":

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(prog='MeshConv', usage='MeshConv refNifti input_mesh output_Nifti',
                      description='Transform SimNIBS mesh file (input_mesh) to reference system (refNifti)',  
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('refNifti', type=str,help='Reference Nifti filename')
    parser.add_argument('input_mesh', type=str,help='Input SimNIBS mesh filename')
    parser.add_argument('output_Nifti', type=str,help='Output Nifti filename')

    args = parser.parse_args()

    print(args)

    aRef = nibabel.load(args.refNifti)
    mesh = mesh_io.read_msh(args.input_mesh)
    mesh = mesh.crop_mesh(elm_type=4)
    ed = mesh.elm.tag1.astype(np.uint16)
    ed = mesh_io.ElementData(ed)
    ed.mesh = mesh
    ed.to_nifti(aRef.shape, aRef.affine, args.output_Nifti, method = 'assign')
    