'''
Class for Tx definition
Vanderbilt IGT 12378 Tx
128 elements, focal=72mm, diameter=103mm , elem diameter=6.6 mm, freq=650 kHz
ABOUT:
    author        - Samuel Pichardo
    date          - March 15, 2023
    last update   - March 15, 2023
'''

import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import GenerateFocusTx,SpeedofSoundWater
import os

extlay={}
TemperatureWater=37.0
extlay['c']=SpeedofSoundWater(TemperatureWater)

def computeI12378Geometry(bDoRunDeepSanityTest=False,ToleranceDistance=6.6,FocalDistance=72):
    MininalInterElementDistanceInRadians=ToleranceDistance/FocalDistance # we can test using the angular distance from center n to center m
    print('*****\nMinimal angular distance\n*****', MininalInterElementDistanceInRadians)
    transxyz = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'I12378.csv'),delimiter=',',skiprows=0)
    assert(transxyz.shape[0]==128) #number of elements
    assert(transxyz.shape[1]==3) # X,Y,Z coordinates in mm
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

def I12378Locations(Foc=72e-3):
    radiusMm = Foc*1e3
    temp_positions = computeI12378Geometry()
    temp_positions[:,2]=radiusMm-temp_positions[:,2]
    transLoc = temp_positions/1000
    return transLoc

def GenerateI12378Tx(Frequency=650e3,RotationZ=0,FactorEnlarge=1):

    f=Frequency;
    Foc=72e-3*FactorEnlarge
    Diameter=6.6e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    transLoc = I12378Locations(Foc=Foc)
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxI12378={}
    TxI12378['center'] = np.zeros((0,3))
    TxI12378['elemcenter'] = np.zeros((len(theta),3))
    TxI12378['ds'] = np.zeros((0,1))
    TxI12378['normal'] = np.zeros((0,3))
    TxI12378['elemdims']=TxElem['ds'].size
    TxI12378['NumberElems']=len(theta)
    TxI12378['VertDisplay'] = np.zeros((0,3))
    TxI12378['FaceDisplay'] = np.zeros((0,4),np.int)

    for n in range(len(theta)):
        prevFaceLength=TxI12378['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxI12378['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxI12378['center']=np.vstack((TxI12378['center'],center))
        TxI12378['ds'] =np.vstack((TxI12378['ds'],TxElem['ds']))
        TxI12378['normal'] =np.vstack((TxI12378['normal'],normal))
        TxI12378['VertDisplay'] =np.vstack((TxI12378['VertDisplay'],VertDisplay))
        TxI12378['FaceDisplay']=np.vstack((TxI12378['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxI12378['VertDisplay'][:,2]+=Foc
    TxI12378['center'][:,2]+=Foc
    TxI12378['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxI12378['VertDisplay'][:,0].max()-TxI12378['VertDisplay'][:,0].min(),
                                        TxI12378['VertDisplay'][:,1].max()-TxI12378['VertDisplay'][:,1].min())

    TxI12378['FocalLength']=Foc
    TxI12378['Aperture']=np.max([TxI12378['VertDisplay'][:,0].max()-TxI12378['VertDisplay'][:,0].min(),
                                      TxI12378['VertDisplay'][:,1].max()-TxI12378['VertDisplay'][:,1].min()]);
    return TxI12378
