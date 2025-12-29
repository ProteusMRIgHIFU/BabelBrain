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

def computeDomeTxGeometry():
    transxyz = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'DomeTxTransducerGeometry.csv'),delimiter=',',skiprows=0)
    assert(transxyz.shape[0]==1024) #number of elements
    assert(transxyz.shape[1]==4) #element, X,Y,Z coordinates in mm, and area in mm2
    transxyz=transxyz[:,:3] #we skip the Tx element coordinates only
    return transxyz*1e-3


def GenerateDomeTx(Frequency=220e3,RotationZ=0,FactorEnlarge=1,PPWSurface=9):

    f=Frequency;
    Foc=150e-3*FactorEnlarge
    Diameter=9e-3*FactorEnlarge

    extlay={}
    TemperatureWater=37.0
    extlay['c']=SpeedofSoundWater(TemperatureWater)

    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'],PPWSurface=PPWSurface)

    transLoc = computeDomeTxGeometry()

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

    TxDomeTx={}
    TxDomeTx['center'] = np.zeros((nSubElems*NElems,3))
    TxDomeTx['elemcenter'] = np.zeros((len(theta),3))
    TxDomeTx['ds'] = np.zeros((nSubElems*NElems,1))
    TxDomeTx['normal'] = np.zeros((nSubElems*NElems,3))
    TxDomeTx['elemdims']=TxElem['ds'].size
    TxDomeTx['NumberElems']=len(theta)
    TxDomeTx['VertDisplay'] = np.zeros((nSubElemsVert*NElems,3))
    TxDomeTx['FaceDisplay'] = np.zeros((nSubElems*NElems,4),np.int64)

    for n in range(len(theta)):
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxDomeTx['elemcenter'][n,:]=np.mean(center,axis=0) # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T

        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxDomeTx['center'][n*nSubElems:(n+1)*nSubElems,:]=center
        TxDomeTx['ds'][n*nSubElems:(n+1)*nSubElems]=TxElem['ds']
        TxDomeTx['normal'][n*nSubElems:(n+1)*nSubElems,:]=normal
        TxDomeTx['VertDisplay'][n*nSubElemsVert:(n+1)*nSubElemsVert,:]=VertDisplay
        TxDomeTx['FaceDisplay'][n*nSubElems:(n+1)*nSubElems,:]=TxElem['FaceDisplay']+n*nSubElemsVert

    TxDomeTx['VertDisplay'][:,2]
    TxDomeTx['center'][:,2]
    TxDomeTx['elemcenter'][:,2]
    
    print('Aperture dimensions (x,y) =',TxDomeTx['center'][:,0].max()-TxDomeTx['center'][:,0].min(),
                                        TxDomeTx['center'][:,1].max()-TxDomeTx['center'][:,1].min())

    print('Aperture dimensions (x,y) =',TxDomeTx['VertDisplay'][:,0].max()-TxDomeTx['VertDisplay'][:,0].min(),
                                        TxDomeTx['VertDisplay'][:,1].max()-TxDomeTx['VertDisplay'][:,1].min())


    TxDomeTx['FocalLength']=Foc
    TxDomeTx['Aperture']=np.max([TxDomeTx['VertDisplay'][:,0].max()-TxDomeTx['VertDisplay'][:,0].min(),
                                      TxDomeTx['VertDisplay'][:,1].max()-TxDomeTx['VertDisplay'][:,1].min()]);
    
    #We use calibration per PPW to generate 1W per element
    TxDomeTx['Amplitude1W']={'Rayleigh':0.14475482330468514,
                            'Visco':{220000:{6:74065.04,
                                             7:79050.414,
                                             8:84021.836,
                                             9:88933.47,
                                            10:94068.0,
                                            11:91529.37,
                                            12:97344.266},
                                     670000:{6:166890.38}}}

    return TxDomeTx


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

        self._Tx=GenerateDomeTx(Frequency=self._Frequency,RotationZ=self._RotationZ,FactorEnlarge=self._FactorEnlarge)
        self._TxOrig=GenerateDomeTx(Frequency=self._Frequency,RotationZ=self._RotationZ)
        if self._Frequency == 220e3:
            self._TxHighRes=GenerateDomeTx(Frequency=self._Frequency,RotationZ=self._RotationZ,PPWSurface=20)
        else:
            self._TxHighRes=self._TxOrig
        
        
