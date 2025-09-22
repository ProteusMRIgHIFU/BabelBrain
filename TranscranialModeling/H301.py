'''
Class for Tx definition
SonicConcepts H301 Tx
128 elements, focal=150mm, diameter=150mm , elem diameter=10.15 mm, freq=1100 kHz
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

def computeH301Geometry():
    df=pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),'H301.csv'),
                sep='\t',names=['theta','radii'])
    F=np.array([0,0,150])
    C=np.array([0,0,0])
    V1=(F-C)/150
    V2=np.cross(V1,[1,0,0])
    radii=df['radii'].values.reshape((128,1))
    theta=np.deg2rad(df['theta'].values.reshape((128,1)))
    a=150-np.sqrt(150**2-radii**2)
    TxCoords=C + V1*a + radii*V2*np.cos(theta) + np.sin(theta)*np.cross(V1,radii*V2)
    assert(np.allclose(np.linalg.norm(TxCoords-F,axis=1),150))
    return TxCoords
        
def H301Locations():
    temp_positions = computeH301Geometry()
    transLoc = temp_positions/1000
    return transLoc

def GenerateH301Tx(Frequency=1.1e6,RotationZ=0,FactorEnlarge=1):

    f=Frequency
    Foc=150e-3*FactorEnlarge
    Diameter=10.15e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    transLoc = H301Locations()
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxH301={}
    TxH301['center'] = np.zeros((0,3))
    TxH301['elemcenter'] = np.zeros((len(theta),3))
    TxH301['ds'] = np.zeros((0,1))
    TxH301['normal'] = np.zeros((0,3))
    TxH301['elemdims']=TxElem['ds'].size
    TxH301['NumberElems']=len(theta)
    TxH301['VertDisplay'] = np.zeros((0,3))
    TxH301['FaceDisplay'] = np.zeros((0,4),int)

    for n in range(len(theta)):
        prevFaceLength=TxH301['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxH301['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxH301['center']=np.vstack((TxH301['center'],center))
        TxH301['ds'] =np.vstack((TxH301['ds'],TxElem['ds']))
        TxH301['normal'] =np.vstack((TxH301['normal'],normal))
        TxH301['VertDisplay'] =np.vstack((TxH301['VertDisplay'],VertDisplay))
        TxH301['FaceDisplay']=np.vstack((TxH301['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxH301['VertDisplay'][:,2]+=Foc
    TxH301['center'][:,2]+=Foc
    TxH301['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxH301['VertDisplay'][:,0].max()-TxH301['VertDisplay'][:,0].min(),
                                        TxH301['VertDisplay'][:,1].max()-TxH301['VertDisplay'][:,1].min())

    TxH301['FocalLength']=Foc
    TxH301['Aperture']=np.max([TxH301['VertDisplay'][:,0].max()-TxH301['VertDisplay'][:,0].min(),
                                      TxH301['VertDisplay'][:,1].max()-TxH301['VertDisplay'][:,1].min()]);
    return TxH301
