from re import S
import numpy as np
#from mkl_fft import fft
from numpy.fft import fft
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from  scipy.io import loadmat
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, InitCuda, InitOpenCL , GenerateFocusTx,SpeedofSoundWater
from BabelViscoFDTD.PropagationModel import PropagationModel
from BabelViscoFDTD.H5pySimple import SaveToH5py, ReadFromH5py
from pprint import pprint
from scipy.io import loadmat
from sys import platform
from time import time
import os

PModel=PropagationModel()
import gc
import time

extlay={}
TemperatureWater=37.0
extlay['c']=SpeedofSoundWater(TemperatureWater)



def computeH317Geometry(bDoRunDeepSanityTest=False,ToleranceDistance=9.5,FocalDistance=135):
    MininalInterElementDistanceInRadians=ToleranceDistance/FocalDistance # we can test using the angular distance from center n to center m
    print('*****\nMinimal angular distance\n*****', MininalInterElementDistanceInRadians)
    transxyz = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'H-317 XYZ Coordinates_revB update 1.18.22.csv'),delimiter=',',skiprows=1)
    assert(transxyz.shape[0]==128) #number of elements
    assert(transxyz.shape[1]==4) #element, X,Y,Z coordinates in IN
    transxyz[:,1:]*=25.4 # from in to mm
    transxyz=transxyz[:,1:] #we skip the Tx # 
    if bDoRunDeepSanityTest:
        # we will calculate the closest Tx element distance to each other
        MinimalDistances=np.zeros(128)
        for n in range(128):
            #we select all the other Tx elements
            Sel=np.hstack((np.arange(0,n),np.arange(n+1,128)))
            SelElem=transxyz[n,:].reshape((1,3))
            RestTx=transxyz[Sel,:]
            #we calculate the Euclidean distance 
            DistanceToOtherElements=np.linalg.norm(RestTx-np.repeat(SelElem,127,axis=0),axis=1)
            NormalizedDistance=DistanceToOtherElements/FocalDistance
            #the angular distance between the tx elements is straighforward
            AngularDistance=2*np.arcsin(NormalizedDistance/2.0)
            
            assert(AngularDistance.size==127)
            MinimalDistances[n]=AngularDistance.min()
            print('Closest element distance',n,MinimalDistances[n]) # just for nicer printing
            if MinimalDistances[n]<MininalInterElementDistanceInRadians:
                print(' ******** overlap of elem %i with %i' % (n,Sel[np.argmin(DistanceToOtherElements)]) )
        assert(np.all(MinimalDistances>=MininalInterElementDistanceInRadians))
        
            
    return transxyz

def H317Locations(Foc=135e-3):
    radiusMm = Foc*1e3
    temp_positions = computeH317Geometry()
    temp_positions[:,2]=radiusMm-temp_positions[:,2]
    transLoc = temp_positions/1000
    return transLoc

def GenerateH317Tx(Frequency=700e3,RotationZ=0,FactorEnlarge=1):


    f=Frequency;
    Foc=135e-3*FactorEnlarge
    Diameter=9.5e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    # fig = plt.figure()
    # ax = Axes3D(fig)

    # for n in range(TxElem['FaceDisplay'].shape[0]):
        # verts=TxElem['VertDisplay'][TxElem['FaceDisplay'][n,:],:]
        # ax.scatter3D(verts[:,0],verts[:,1],verts[:,2],color='r')
    # plt.show()


    transLoc = H317Locations(Foc=Foc)
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxH317={}
    TxH317['center'] = np.zeros((0,3))
    TxH317['elemcenter'] = np.zeros((len(theta),3))
    TxH317['ds'] = np.zeros((0,1))
    TxH317['normal'] = np.zeros((0,3))
    TxH317['elemdims']=TxElem['ds'].size
    TxH317['NumberElems']=len(theta)
    TxH317['VertDisplay'] = np.zeros((0,3))
    TxH317['FaceDisplay'] = np.zeros((0,4),np.int64)

    for n in range(len(theta)):
        prevFaceLength=TxH317['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxH317['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxH317['center']=np.vstack((TxH317['center'],center))
        TxH317['ds'] =np.vstack((TxH317['ds'],TxElem['ds']))
        TxH317['normal'] =np.vstack((TxH317['normal'],normal))
        TxH317['VertDisplay'] =np.vstack((TxH317['VertDisplay'],VertDisplay))
        TxH317['FaceDisplay']=np.vstack((TxH317['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxH317['VertDisplay'][:,2]+=Foc
    TxH317['center'][:,2]+=Foc
    TxH317['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxH317['VertDisplay'][:,0].max()-TxH317['VertDisplay'][:,0].min(),
                                        TxH317['VertDisplay'][:,1].max()-TxH317['VertDisplay'][:,1].min())

    # fig = plt.figure(figsize=(8,8))
    # ax = Axes3D(fig)

    # ax.scatter3D(TxH317['center'][::5,0],TxH317['center'][::5,1],TxH317['center'][::5,2],color='r')
    # plt.show()
    TxH317['FocalLength']=Foc
    TxH317['Aperture']=np.max([TxH317['VertDisplay'][:,0].max()-TxH317['VertDisplay'][:,0].min(),
                                      TxH317['VertDisplay'][:,1].max()-TxH317['VertDisplay'][:,1].min()]);
    return TxH317
