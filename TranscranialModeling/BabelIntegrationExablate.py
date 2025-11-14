'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

'''

from . import BabelIntegrationDOME_PHASEDARRAY  
from BabelViscoFDTD.tools.RayleighAndBHTE import GenerateFocusTx,SpeedofSoundWater
import numpy as np
import os

def computeExaNeuroGeometry():
    transxyz = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'ExaNeuroTransducerGeometry.csv'),delimiter=',',skiprows=0)
    assert(transxyz.shape[0]==1024) #number of elements
    assert(transxyz.shape[1]==4) #element, X,Y,Z coordinates in mm, and area in mm2
    transxyz=transxyz[:,:3] #we skip the Tx element coordinates only
    return transxyz*1e-3


def GenerateExaNeuroTx(Frequency=220e3,RotationZ=0,FactorEnlarge=1,PPWSurface=9):

    f=Frequency;
    Foc=150e-3*FactorEnlarge
    Diameter=9e-3*FactorEnlarge

    extlay={}
    TemperatureWater=37.0
    extlay['c']=SpeedofSoundWater(TemperatureWater)

    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'],PPWSurface=PPWSurface)

    transLoc = computeExaNeuroGeometry()

    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc
    NElems=transLoc.shape[0]
    nSubElems=len( TxElem['ds'])
   
    nSubElemsVert=TxElem['VertDisplay'].shape[0]

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)
    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])
    phi+=np.deg2rad(RotationZ)

    TxExaNeuro={}
    TxExaNeuro['center'] = np.zeros((nSubElems*NElems,3))
    TxExaNeuro['elemcenter'] = np.zeros((len(theta),3))
    TxExaNeuro['ds'] = np.zeros((nSubElems*NElems,1))
    TxExaNeuro['normal'] = np.zeros((nSubElems*NElems,3))
    TxExaNeuro['elemdims']=TxElem['ds'].size
    TxExaNeuro['NumberElems']=len(theta)
    TxExaNeuro['VertDisplay'] = np.zeros((nSubElemsVert*NElems,3))
    TxExaNeuro['FaceDisplay'] = np.zeros((nSubElems*NElems,4),np.int64)

    for n in range(len(theta)):
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxExaNeuro['elemcenter'][n,:]=np.mean(center,axis=0) # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T

        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxExaNeuro['center'][n*nSubElems:(n+1)*nSubElems,:]=center
        TxExaNeuro['ds'][n*nSubElems:(n+1)*nSubElems]=TxElem['ds']
        TxExaNeuro['normal'][n*nSubElems:(n+1)*nSubElems,:]=normal
        TxExaNeuro['VertDisplay'][n*nSubElemsVert:(n+1)*nSubElemsVert,:]=VertDisplay
        TxExaNeuro['FaceDisplay'][n*nSubElems:(n+1)*nSubElems,:]=TxElem['FaceDisplay']+n*nSubElemsVert

    TxExaNeuro['VertDisplay'][:,2]
    TxExaNeuro['center'][:,2]
    TxExaNeuro['elemcenter'][:,2]
    
    print('Aperture dimensions (x,y) =',TxExaNeuro['center'][:,0].max()-TxExaNeuro['center'][:,0].min(),
                                        TxExaNeuro['center'][:,1].max()-TxExaNeuro['center'][:,1].min())

    print('Aperture dimensions (x,y) =',TxExaNeuro['VertDisplay'][:,0].max()-TxExaNeuro['VertDisplay'][:,0].min(),
                                        TxExaNeuro['VertDisplay'][:,1].max()-TxExaNeuro['VertDisplay'][:,1].min())


    TxExaNeuro['FocalLength']=Foc
    TxExaNeuro['Aperture']=np.max([TxExaNeuro['VertDisplay'][:,0].max()-TxExaNeuro['VertDisplay'][:,0].min(),
                                      TxExaNeuro['VertDisplay'][:,1].max()-TxExaNeuro['VertDisplay'][:,1].min()]);
    
    #We use calibration per PPW
    TxExaNeuro['Amplitude1W']={6:215775.2400433752,
                                7:217102.38518294977,
                                8:219502.12978939048,
                                9:216841.80813398556,
                                10:214838.89980138713,
                                11:241676.32380101856,
                                12:239446.6339836397,
                                13:236778.14630625723,
                                14:235598.60482377192,
                                15:230939.00845080774,
                                16:228203.36642938704,
                                17:247228.00527528222,
                                18:226231.69718295365,
                                19:241637.3257215121,
                                20:225525.45465253628}
    225707.72534123383
    return TxExaNeuro


class RUN_SIM(BabelIntegrationDOME_PHASEDARRAY.RUN_SIM):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    **kargs)
    
##########################################

class BabelFTD_Simulations(BabelIntegrationDOME_PHASEDARRAY.BabelFTD_Simulations):
    #Meta class dealing with the specificis of each test based on the string name
    
    def CreateSimConditions(self,**kargs):
        return SimulationConditions(Aperture=300e-3,
                                    FocalLength=150e-3,
                                    XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    **kargs)

        

class SimulationConditions(BabelIntegrationDOME_PHASEDARRAY.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,Aperture=300e-3,
                      FocalLength=150e-3,
                      **kargs):
        super().__init__(Aperture=Aperture,FocalLength=FocalLength,**kargs)
        
    def GenTransducerGeom(self):

        self._Tx=GenerateExaNeuroTx(Frequency=self._Frequency,RotationZ=self._RotationZ,FactorEnlarge=self._FactorEnlarge)
        self._TxOrig=GenerateExaNeuroTx(Frequency=self._Frequency,RotationZ=self._RotationZ)
        self._TxHighRes=GenerateExaNeuroTx(Frequency=self._Frequency,RotationZ=self._RotationZ,PPWSurface=20)
        
        
