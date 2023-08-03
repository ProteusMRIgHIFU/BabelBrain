'''
Tool to convert between Brainsight and 3D Slicer linear transform
'''
import pandas as pd
import numpy as np
import argparse
import sys

templateSlicer=\
'''#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: {m0n0:10.9f} {m0n1:10.9f} {m0n2:10.9f} {m1n0:10.9f} {m1n1:10.9f} {m1n2:10.9f} {m2n0:10.9f} {m2n1:10.9f} {m2n2:10.9f} {X:10.9f} {Y:10.9f} {Z:1.9f}
FixedParameters: 0 0 0
'''

templateBSight=\
'''# Version: 13
# Coordinate system: NIfTI:Aligned
# Created by: Brainsight 2.5
# Units: millimetres, degrees, milliseconds, and microvolts
# Encoding: UTF-8
# Notes: Each column is delimited by a tab. Each value within a column is delimited by a semicolon.
# Target Name	Loc. X	Loc. Y	Loc. Z	m0n0	m0n1	m0n2	m1n0	m1n1	m1n2	m2n0	m2n1	m2n2
{name}\t{X:10.9f}\t{Y:10.9f}\t{Z:10.9f}\t{m0n0:10.9f}\t{m0n1:10.9f}\t{m0n2:10.9f}\t{m1n0:10.9f}\t{m1n1:10.9f}\t{m1n2:10.9f}\t{m2n0:10.9f}\t{m2n1:10.9f}\t{m2n2:10.9f}
'''

import re

def read_itk_affine_transform(filename):
    with open(filename) as f:
        tfm_file_lines = f.readlines()
    # parse the transform parameters
    match = re.match("Transform: AffineTransform_[a-z]+_([0-9]+)_([0-9]+)", tfm_file_lines[2])
    if not match or match.group(1) != '3' or match.group(2) != '3':
        raise ValueError(f"{filename} is not an ITK 3D affine transform file")
    p = np.array( tfm_file_lines[3].split()[1:], dtype=np.float64 )
    # assemble 4x4 matrix from ITK transform parameters
    itk_transform = np.array([
        [p[0], p[1], p[2], p[9]],
        [p[3], p[4], p[5], p[10]],
        [p[6], p[7], p[8], p[11]],
        [0, 0, 0, 1]])
    return itk_transform

def itk_to_BSight(itk_transform):
    # ITK transform: from parent, using LPS coordinate system
    # Transform displayed in Slicer: to parent, using RAS coordinate system
    ras2lps = np.diag([-1, -1, 1, 1])
    transform_from_parent_RAS = np.linalg.inv(ras2lps @ itk_transform @ ras2lps)
    transform_from_parent_RAS[:3,:3]=np.diagflat([-1,-1,-1])@transform_from_parent_RAS[:3,:3]
    return transform_from_parent_RAS

def BSight_to_itk(BSight_transform):
    ras2lps = np.diag([-1, -1, 1, 1])
    in_trans=BSight_transform.copy()
    in_trans[:3,:3]=np.diagflat([-1,-1,-1])@in_trans[:3,:3]
    transform_to_LPS = ras2lps @  np.linalg.inv(in_trans) @ ras2lps

    return transform_to_LPS


def GetIDTrajectoryBrainsight(fname):
    names=['Target name', 
      'Loc. X','Loc. Y','Loc. Z',
      'm0n0','m0n1','m0n2',
      'm1n0','m1n1','m1n2',
      'm2n0','m2n1','m2n2']
    df=pd.read_csv(fname,comment='#',sep='\t',header=None,names=names,engine='python',usecols=names).iloc[0]  
    return df['Target name']

def ReadTrajectoryBrainsight(fname):
    names=['Target name', 
      'Loc. X','Loc. Y','Loc. Z',
      'm0n0','m0n1','m0n2',
      'm1n0','m1n1','m1n2',
      'm2n0','m2n1','m2n2']
    df=pd.read_csv(fname,comment='#',sep='\t',header=None,names=names,engine='python',usecols=names[1:]).iloc[0].to_numpy()
    print(df)
    Mat4=np.eye(4)
    Mat4[:3,3]=df[:3]
    Mat4[:3,0]=df[3:6]
    Mat4[:3,1]=df[6:9]
    Mat4[:3,2]=df[9:]
    
    return Mat4


if __name__ == "__main__":

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    
    parser = MyParser(prog='ConvMatTransform', usage='python %(prog)s.py [options]',description='Convert matrix trajectory (transform) between Brainsight and 3D Slicer (ITK) format',  
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('Infname', type=str, nargs='+',help='Input text file of trajectory/transform')
    parser.add_argument('--bFromSlicer2BSight',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--IdName', type=str, nargs='?',default='unknown',help='name of the target when exporting to BSight')
    

    args = parser.parse_args()

    print(args)

    Infname=args.Infname[0]
    lps2ras=np.diagflat([-1 ,-1 ,1, 1])
    ras2lps=np.diagflat([-1, -1, 1, 1])
   
    if args.bFromSlicer2BSight:
        outname=Infname.split('.txt')[0]+'_BSight.txt'

        inMat=read_itk_affine_transform(Infname)
         #we add this as in Brainsight the needle for trajectory starts at with a vector pointing 
         #to the feet direction , while in SlicerIGT it starts with a vector towards the head
        print(inMat)

        transform = itk_to_BSight(inMat)

        print(transform)

        outString=templateBSight.format(m0n0=transform[0,0],
                                m0n1=transform[1,0],
                                m0n2=transform[2,0],
                                m1n0=transform[0,1],
                                m1n1=transform[1,1],
                                m1n2=transform[2,1],
                                m2n0=transform[0,2],
                                m2n1=transform[1,2],
                                m2n2=transform[2,2],
                                X=transform[0,3],
                                Y=transform[1,3],
                                Z=transform[2,3],
                                name=args.IdName)
    else:
        inMat=ReadTrajectoryBrainsight(Infname)
        print(inMat)
        transform = BSight_to_itk(inMat)
        print(transform)
        outname=Infname.split('.txt')[0]+'_Slicer.txt'

        outString=templateSlicer.format(m0n0=transform[0,0],
                                        m0n1=transform[1,0],
                                        m0n2=transform[2,0],
                                        m1n0=transform[0,1],
                                        m1n1=transform[1,1],
                                        m1n2=transform[2,1],
                                        m2n0=transform[0,2],
                                        m2n1=transform[1,2],
                                        m2n2=transform[2,2],
                                        X=transform[0,3],
                                        Y=transform[1,3],
                                        Z=transform[2,3])
        
    
    print(outString)
    with open(outname,'w') as f:
        f.write(outString)