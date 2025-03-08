'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - March 7, 2025
     last update   - March 7, 2025

The R15287 is a 10-ring Tx from Imasonic with focal length of 75 mm and diameter of 65 mm
'''
import numpy as np
from . import BabelIntegrationCTX500

class RUN_SIM(BabelIntegrationCTX500.RUN_SIM):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(ZSteering=self._ZSteering,
                                     **kargs)


class BabelFTD_Simulations(BabelIntegrationCTX500.BabelFTD_Simulations):
    def CreateSimConditions(self,**kargs):  
        return SimulationConditions(ZSteering=self._ZSteering,
                                    Aperture=65e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                                    FocalLength=75e-3,
                                    **kargs)
class SimulationConditions(BabelIntegrationCTX500.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,InDiameters= np.array([10e-3        ,
                                            22.3e-3, 
                                            30e-3, 
                                            36.3e-3,
                                            41.7e-3,
                                            46.5e-3,
                                            51e-3,
                                            55.1e-3,
                                            58.9e-3,
                                            62.5e-3]), #inner diameter of rings
                      OutDiameters=np.array([21.3e-3        ,
                                            29.1e-3, 
                                            35.3e-3, 
                                            40.7e-3,
                                            45.6e-3,
                                            50e-3,
                                            54.1e-3,
                                            58e-3,
                                            61.6e-3,
                                            65e-3]), #outer diameter of rings
                      **kargs): # steering
        
        super().__init__(InDiameters=InDiameters,
                         OutDiameters=OutDiameters,**kargs)
  