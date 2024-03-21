'''
Pipeline to execute viscoleastic simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

'''
from .BabelIntegrationBASE import (RUN_SIM_BASE, 
                            BabelFTD_Simulations_BASE,
                            SimulationConditionsBASE,
                            Material)
import numpy as np
from sys import platform
import os
from stl import mesh
import scipy
from trimesh import creation 
import matplotlib.pyplot as plt
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple
from .I12378 import GenerateI12378Tx
import nibabel
from . import BabelIntegrationH317  

class RUN_SIM(BabelIntegrationH317.RUN_SIM):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(**kargs)
    
##########################################

class BabelFTD_Simulations(BabelIntegrationH317.BabelFTD_Simulations):
    #Meta class dealing with the specificis of each test based on the string name
    
    def CreateSimConditions(self,**kargs):
        return SimulationConditions(Aperture=103.0e-3,
                                    FocalLength=72.0e-3,
                                    **kargs)

        

class SimulationConditions(BabelIntegrationH317.SimulationConditions):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,Aperture=103.0e-3,
                      FocalLength=72.0e-3,
                      **kargs):
        super().__init__(Aperture=Aperture,FocalLength=FocalLength,**kargs)
        
    def GenTransducerGeom(self):
        self._Tx=GenerateI12378Tx(Frequency=self._Frequency,RotationZ=self._RotationZ,FactorEnlarge=self._FactorEnlarge)
        self._TxOrig=GenerateI12378Tx(Frequency=self._Frequency,RotationZ=self._RotationZ)
        
        
