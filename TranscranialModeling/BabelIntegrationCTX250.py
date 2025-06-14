'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - July 24, 2024
     last update   - July 24, 2024

The CTX 250 is identical to the CTX 500, excepting the central frequency
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
                                    FocalLength=63.2e-3,
                                    **kargs)
class SimulationConditions(BabelIntegrationANNULAR_ARRAY.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,InDiameters= np.array([0.        , 0.03243857, 0.04582899, 0.05597536]), #inner diameter of rings
                      OutDiameters=np.array([0.0312153 , 0.04464872, 0.05483928, 0.06328742]), #outer diameter of rings
                      **kargs): # steering
        
        super().__init__(InDiameters=InDiameters,
                         OutDiameters=OutDiameters,**kargs)
  