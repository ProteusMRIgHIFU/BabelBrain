'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - Aug 21, 2024
     last update   - Aug 21, 2024

The DPX 500 is identical to the CTX 500, excepting the focal length of 150 mm. We are using the config of the CTX250 for the rings
'''
import numpy as np
from . import BabelIntegrationANNULAR_ARRAY

class RUN_SIM(BabelIntegrationANNULAR_ARRAY.RUN_SIM):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(ZSteering=self._ZSteering,
                                     **kargs)


class BabelFTD_Simulations(BabelIntegrationANNULAR_ARRAY.BabelFTD_Simulations):
    def CreateSimConditions(self,**kargs):  
        return SimulationConditions(ZSteering=self._ZSteering,
                                    Aperture=64e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                                    FocalLength=150e-3,
                                    **kargs)
class SimulationConditions(BabelIntegrationANNULAR_ARRAY.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,InDiameters= np.array([0.        , 0.03198638, 0.04561623, 0.05624559]), #inner diameter of rings
                      OutDiameters=np.array([0.03049454, 0.0441331 , 0.05477149, 0.063851  ]), #outer diameter of rings
                      **kargs): # steering
        
        super().__init__(InDiameters=InDiameters,
                         OutDiameters=OutDiameters,**kargs)
  