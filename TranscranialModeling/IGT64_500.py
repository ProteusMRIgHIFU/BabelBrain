'''
Class for Tx definition
Stanford U - IGT - Imasonic IGT64_500 Tx
64 elements, focal=75mm, diameter=65mm , elem diameter=6 mm, freq=500 kHz
ABOUT:
    author        - Samuel Pichardo
    date          - Aug 17, 2025
'''

import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import GenerateFocusTx,SpeedofSoundWater
import pandas as pd
import os

extlay={}
TemperatureWater=37.0
extlay['c']=SpeedofSoundWater(TemperatureWater)

def computeIGT64_500Geometry():
    df= pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),'IGT64_500.csv'),sep=',')
    transxyz = np.zeros((64,3))
    transxyz[:,0]=np.array(df['X'])
    transxyz[:,1]=np.array(df['Y'])
    transxyz[:,2]=75.0-np.array(df['Z'])
    return transxyz
        
def IGT64_500Locations():
    temp_positions = computeIGT64_500Geometry()
    transLoc = temp_positions/1000
    return transLoc

def GenerateIGT64_500Tx(Frequency=0.5e6,RotationZ=0,FactorEnlarge=1):

    f=Frequency
    Foc=75e-3*FactorEnlarge
    Diameter=6.0e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    transLoc = IGT64_500Locations()
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxIGT64_500={}
    TxIGT64_500['center'] = np.zeros((0,3))
    TxIGT64_500['elemcenter'] = np.zeros((len(theta),3))
    TxIGT64_500['ds'] = np.zeros((0,1))
    TxIGT64_500['normal'] = np.zeros((0,3))
    TxIGT64_500['elemdims']=TxElem['ds'].size
    TxIGT64_500['NumberElems']=len(theta)
    TxIGT64_500['VertDisplay'] = np.zeros((0,3))
    TxIGT64_500['FaceDisplay'] = np.zeros((0,4),int)

    for n in range(len(theta)):
        prevFaceLength=TxIGT64_500['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxIGT64_500['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxIGT64_500['center']=np.vstack((TxIGT64_500['center'],center))
        TxIGT64_500['ds'] =np.vstack((TxIGT64_500['ds'],TxElem['ds']))
        TxIGT64_500['normal'] =np.vstack((TxIGT64_500['normal'],normal))
        TxIGT64_500['VertDisplay'] =np.vstack((TxIGT64_500['VertDisplay'],VertDisplay))
        TxIGT64_500['FaceDisplay']=np.vstack((TxIGT64_500['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxIGT64_500['VertDisplay'][:,2]+=Foc
    TxIGT64_500['center'][:,2]+=Foc
    TxIGT64_500['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxIGT64_500['VertDisplay'][:,0].max()-TxIGT64_500['VertDisplay'][:,0].min(),
                                        TxIGT64_500['VertDisplay'][:,1].max()-TxIGT64_500['VertDisplay'][:,1].min())

    TxIGT64_500['FocalLength']=Foc
    TxIGT64_500['Aperture']=np.max([TxIGT64_500['VertDisplay'][:,0].max()-TxIGT64_500['VertDisplay'][:,0].min(),
                                      TxIGT64_500['VertDisplay'][:,1].max()-TxIGT64_500['VertDisplay'][:,1].min()]);
    return TxIGT64_500
