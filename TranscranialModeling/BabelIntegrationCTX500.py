'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - May 19, 2022

'''

import numpy as np

from . import BabelIntegrationANNULAR_ARRAY

class RUN_SIM(BabelIntegrationANNULAR_ARRAY.RUN_SIM):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(ZSteering=self._ZSteering,
                                     **kargs)


class BabelFTD_Simulations(BabelIntegrationANNULAR_ARRAY.BabelFTD_Simulations):
    #Meta class dealing with the specificis of each test based on the string name
    def CreateSimConditions(self,**kargs):     
        return SimulationConditions(ZSteering=self._ZSteering,
                                    Aperture=64e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                                    FocalLength=63.2e-3,
                                    **kargs)
class SimulationConditions(BabelIntegrationANNULAR_ARRAY.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,InDiameters= np.array([0.0    , 32.8e-3, 46e-3,   55.9e-3]), #inner diameter of rings
                      OutDiameters=np.array([32.8e-3, 46e-3,   55.9e-3, 64e-3]), #outer diameter of rings
                      **kargs): # steering
        super().__init__(InDiameters=InDiameters,
                         OutDiameters=OutDiameters,**kargs)
        
