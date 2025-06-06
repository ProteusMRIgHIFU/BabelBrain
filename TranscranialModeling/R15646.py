'''
Class for Tx definition
U Washington St Louis Imasonic R15646 Tx
64 elements, focal=65mm, diameter=65.95mm , elem diameter=6 mm, freq=650 kHz
ABOUT:
    author        - Samuel Pichardo
    date          - June 1, 2025
'''

import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import GenerateFocusTx,SpeedofSoundWater
import pandas as pd
import os

extlay={}
TemperatureWater=37.0
extlay['c']=SpeedofSoundWater(TemperatureWater)

def computeR15646Geometry():
    df= pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),'R15646.csv'),sep=' ')
    transxyz = np.zeros((64,3))
    transxyz[:,0]=np.array(df['X'])
    transxyz[:,1]=np.array(df['Y'])
    transxyz[:,2]=65.0-np.array(df['Z'])
    return transxyz

def R15646Locations():
    temp_positions = computeR15646Geometry()
    transLoc = temp_positions/1000
    return transLoc

def GenerateR15646Tx(Frequency=0.65e6,RotationZ=0,FactorEnlarge=1):

    f=Frequency
    Foc=65.0e-3*FactorEnlarge
    Diameter=6e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    transLoc = R15646Locations()
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxR15646={}
    TxR15646['center'] = np.zeros((0,3))
    TxR15646['elemcenter'] = np.zeros((len(theta),3))
    TxR15646['ds'] = np.zeros((0,1))
    TxR15646['normal'] = np.zeros((0,3))
    TxR15646['elemdims']=TxElem['ds'].size
    TxR15646['NumberElems']=len(theta)
    TxR15646['VertDisplay'] = np.zeros((0,3))
    TxR15646['FaceDisplay'] = np.zeros((0,4),np.int)

    for n in range(len(theta)):
        prevFaceLength=TxR15646['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxR15646['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxR15646['center']=np.vstack((TxR15646['center'],center))
        TxR15646['ds'] =np.vstack((TxR15646['ds'],TxElem['ds']))
        TxR15646['normal'] =np.vstack((TxR15646['normal'],normal))
        TxR15646['VertDisplay'] =np.vstack((TxR15646['VertDisplay'],VertDisplay))
        TxR15646['FaceDisplay']=np.vstack((TxR15646['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxR15646['VertDisplay'][:,2]+=Foc
    TxR15646['center'][:,2]+=Foc
    TxR15646['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxR15646['VertDisplay'][:,0].max()-TxR15646['VertDisplay'][:,0].min(),
                                        TxR15646['VertDisplay'][:,1].max()-TxR15646['VertDisplay'][:,1].min())

    TxR15646['FocalLength']=Foc
    TxR15646['Aperture']=np.max([TxR15646['VertDisplay'][:,0].max()-TxR15646['VertDisplay'][:,0].min(),
                                      TxR15646['VertDisplay'][:,1].max()-TxR15646['VertDisplay'][:,1].min()]);
    return TxR15646
