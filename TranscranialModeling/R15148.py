'''
Class for Tx definition
U Washington St Louis Imasonic R15148 Tx
128 elements, focal=80mm, diameter=103mm , elem diameter=6.6 mm, freq=500 kHz
ABOUT:
    author        - Samuel Pichardo
    date          - Nov 22, 2024
'''

import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import GenerateFocusTx,SpeedofSoundWater
from scipy.io import loadmat
import os

extlay={}
TemperatureWater=37.0
extlay['c']=SpeedofSoundWater(TemperatureWater)

def computeR15148Geometry():
    transxyz = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)),'R15148_1001.mat'))['IGT128']
    assert(transxyz.shape[0]==128) #number of elements
    assert(transxyz.shape[1]==3) # X,Y,Z coordinates in mm      
    return transxyz

def R15148Locations():
    temp_positions = computeR15148Geometry()
    transLoc = temp_positions/1000
    return transLoc

def GenerateR15148Tx(Frequency=0.5e6,RotationZ=0,FactorEnlarge=1):

    f=Frequency
    Foc=80e-3*FactorEnlarge
    Diameter=6.6e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    transLoc = R15148Locations()
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxR15148={}
    TxR15148['center'] = np.zeros((0,3))
    TxR15148['elemcenter'] = np.zeros((len(theta),3))
    TxR15148['ds'] = np.zeros((0,1))
    TxR15148['normal'] = np.zeros((0,3))
    TxR15148['elemdims']=TxElem['ds'].size
    TxR15148['NumberElems']=len(theta)
    TxR15148['VertDisplay'] = np.zeros((0,3))
    TxR15148['FaceDisplay'] = np.zeros((0,4),int)

    for n in range(len(theta)):
        prevFaceLength=TxR15148['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxR15148['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxR15148['center']=np.vstack((TxR15148['center'],center))
        TxR15148['ds'] =np.vstack((TxR15148['ds'],TxElem['ds']))
        TxR15148['normal'] =np.vstack((TxR15148['normal'],normal))
        TxR15148['VertDisplay'] =np.vstack((TxR15148['VertDisplay'],VertDisplay))
        TxR15148['FaceDisplay']=np.vstack((TxR15148['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxR15148['VertDisplay'][:,2]+=Foc
    TxR15148['center'][:,2]+=Foc
    TxR15148['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxR15148['VertDisplay'][:,0].max()-TxR15148['VertDisplay'][:,0].min(),
                                        TxR15148['VertDisplay'][:,1].max()-TxR15148['VertDisplay'][:,1].min())

    TxR15148['FocalLength']=Foc
    TxR15148['Aperture']=np.max([TxR15148['VertDisplay'][:,0].max()-TxR15148['VertDisplay'][:,0].min(),
                                      TxR15148['VertDisplay'][:,1].max()-TxR15148['VertDisplay'][:,1].min()]);
    return TxR15148
