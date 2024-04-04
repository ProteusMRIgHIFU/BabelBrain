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

def computeATACGeometry(bDoRunDeepSanityTest=False,ToleranceDistance=6.6,FocalDistance=53.2):
    MininalInterElementDistanceInRadians=ToleranceDistance/FocalDistance # we can test using the angular distance from center n to center m
    print('*****\nMinimal angular distance\n*****', MininalInterElementDistanceInRadians)
    transxyz = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'ATACArray.csv'),delimiter=',',skiprows=0)
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

def ATACLocations(Foc=53.2e-3):
    radiusMm = Foc*1e3
    temp_positions = computeATACGeometry()
    transLoc = temp_positions/1000
    return transLoc

def GenerateATACTx(Frequency=1e6,RotationZ=0,FactorEnlarge=1):

    f=Frequency
    Foc=53.2e-3*FactorEnlarge
    Diameter=3.5e-3*FactorEnlarge

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    transLoc = ATACLocations(Foc=Foc)
  
    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxATAC={}
    TxATAC['center'] = np.zeros((0,3))
    TxATAC['elemcenter'] = np.zeros((len(theta),3))
    TxATAC['ds'] = np.zeros((0,1))
    TxATAC['normal'] = np.zeros((0,3))
    TxATAC['elemdims']=TxElem['ds'].size
    TxATAC['NumberElems']=len(theta)
    TxATAC['VertDisplay'] = np.zeros((0,3))
    TxATAC['FaceDisplay'] = np.zeros((0,4),np.int)

    for n in range(len(theta)):
        prevFaceLength=TxATAC['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxATAC['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxATAC['center']=np.vstack((TxATAC['center'],center))
        TxATAC['ds'] =np.vstack((TxATAC['ds'],TxElem['ds']))
        TxATAC['normal'] =np.vstack((TxATAC['normal'],normal))
        TxATAC['VertDisplay'] =np.vstack((TxATAC['VertDisplay'],VertDisplay))
        TxATAC['FaceDisplay']=np.vstack((TxATAC['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxATAC['VertDisplay'][:,2]+=Foc
    TxATAC['center'][:,2]+=Foc
    TxATAC['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxATAC['VertDisplay'][:,0].max()-TxATAC['VertDisplay'][:,0].min(),
                                        TxATAC['VertDisplay'][:,1].max()-TxATAC['VertDisplay'][:,1].min())

    TxATAC['FocalLength']=Foc
    TxATAC['Aperture']=np.max([TxATAC['VertDisplay'][:,0].max()-TxATAC['VertDisplay'][:,0].min(),
                                      TxATAC['VertDisplay'][:,1].max()-TxATAC['VertDisplay'][:,1].min()]);
    return TxATAC
