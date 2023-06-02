'''
Samuel Pichardo,
May 2, 2023

Tool to normalize BabelBrain Nifti results on the brain region
'''
import argparse
import sys
import nibabel
import numpy as np

if __name__ == "__main__":

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    
    parser = MyParser(prog='ResaveBabelNormalized', usage='python %(prog)s.py [options]',description='Resave results in normalized conditions in brain region',  
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ResultsPath', type=str, nargs='+',help='Input Results file path to Nifti data (must finish with _Sub.nii.gz file ending)')
    parser.add_argument('MaskPath', type=str, nargs='+',help='Input Mask file path to Nifti data (must finish with _BabelViscoInput.nii.gz file ending)')
    
    args = parser.parse_args()
  
    RPath=args.ResultsPath[0]
    MaskPath=args.MaskPath[0]
    assert('FullElasticSolution_Sub.nii.gz' in RPath)
    assert('BabelViscoInput.nii.gz' in MaskPath)
    NRPath=RPath.replace('FullElasticSolution_Sub.nii.gz','FullElasticSolution_Sub_NORM.nii.gz')


    Results=nibabel.load(RPath)
    Mask=nibabel.load(MaskPath)

    ResultsData=Results.get_fdata()
    MaskData=Mask.get_fdata()
    ii,jj,kk=np.mgrid[0:ResultsData.shape[0],0:ResultsData.shape[1],0:ResultsData.shape[2]]

    Indexes=np.c_[(ii.flatten().T,jj.flatten().T,kk.flatten().T,np.ones((kk.size,1)))].T

    PosResults=Results.affine.dot(Indexes)

    IndexesMask=np.round(np.linalg.inv(Mask.affine).dot(PosResults)).astype(int)

    SubMask=MaskData[IndexesMask[0,:],IndexesMask[1,:],IndexesMask[2,:]].reshape(ResultsData.shape)
    ResultsData[SubMask<4]=0
    ResultsData/=ResultsData.max()
    NormalizedNifti=nibabel.Nifti1Image(ResultsData,Results.affine,header=Results.header)
    NormalizedNifti.to_filename(NRPath)
