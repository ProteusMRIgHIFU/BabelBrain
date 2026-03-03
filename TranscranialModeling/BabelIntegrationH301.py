'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - Sep 19, 2025

'''
from .H301 import generate_h301_tx
from . import BabelIntegrationCONCAVE_PHASEDARRAY  

class RUN_SIM(BabelIntegrationCONCAVE_PHASEDARRAY.RUN_SIM):
    def create_sim_object(self,**kargs):
        return BabelFTD_Simulations(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    DistanceConeToFocus=self._DistanceConeToFocus,
                                    **kargs)
    
##########################################

class BabelFTD_Simulations(BabelIntegrationCONCAVE_PHASEDARRAY.BabelFTD_Simulations):
    #Meta class dealing with the specificis of each test based on the string name
    
    def create_sim_conditions(self,**kargs):
        return SimulationConditions(Aperture=150e-3,
                                    FocalLength=150e-3,
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
    def __init__(self,Aperture=150e-3,
                      FocalLength=150e-3,
                      **kargs):
        super().__init__(Aperture=Aperture,FocalLength=FocalLength,**kargs)
        
    def gen_transducer_geom(self):
        self._Tx=generate_h301_tx(Frequency=self._Frequency,RotationZ=self._RotationZ,FactorEnlarge=self._FactorEnlarge)
        self._TxOrig=generate_h301_tx(Frequency=self._Frequency,RotationZ=self._RotationZ)
        