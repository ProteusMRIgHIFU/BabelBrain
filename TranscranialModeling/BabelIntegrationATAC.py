'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - March 21, 2024

'''
from .ATAC import GenerateATACTx
from . import BabelIntegrationCONCAVE_PHASEDARRAY  

class RUN_SIM(BabelIntegrationCONCAVE_PHASEDARRAY.RUN_SIM):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    DistanceConeToFocus=self._DistanceConeToFocus,
                                    **kargs)
    
##########################################

class BabelFTD_Simulations(BabelIntegrationCONCAVE_PHASEDARRAY.BabelFTD_Simulations):
    #Meta class dealing with the specificis of each test based on the string name
    
    def CreateSimConditions(self,**kargs):
        return SimulationConditions(Aperture=58.0e-3,
                                    FocalLength=52.3e-3,
                                    XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    DistanceConeToFocus=self._DistanceConeToFocus,
                                    RotationZ=self._RotationZ,
                                    **kargs)

        

class SimulationConditions(BabelIntegrationCONCAVE_PHASEDARRAY.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,Aperture=58.0e-3,
                      FocalLength=52.3e-3,
                      **kargs):
        super().__init__(Aperture=Aperture,FocalLength=FocalLength,**kargs)
        
    def GenTransducerGeom(self):
        self._Tx=GenerateATACTx(Frequency=self._Frequency,RotationZ=self._RotationZ,FactorEnlarge=self._FactorEnlarge)
        self._TxOrig=GenerateATACTx(Frequency=self._Frequency,RotationZ=self._RotationZ)
        
        
