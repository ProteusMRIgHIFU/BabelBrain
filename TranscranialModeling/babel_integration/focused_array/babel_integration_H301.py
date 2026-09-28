'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - Sep 19, 2025

'''
from BabelViscoFDTD.tools.RayleighAndBHTE import SpeedofSoundWater
import numpy as np

from TranscranialModeling.babel_integration.integration_templates import (
    babel_integration_focused_array,
)
from TranscranialModeling.tx_geometries import generate_focused_array_tx

def compute_H301_xyz_coords(radii, theta, focal_length):
        
    natural_focal_spot = np.array([0,0,focal_length])
    center = np.array([0,0,0])
    
    z_unit_vector = (natural_focal_spot-center)/focal_length
    y_unit_vector = np.cross(z_unit_vector,[1,0,0])

    height = np.sqrt(focal_length**2 - radii**2)

    tx_xyz = (
        center
        + z_unit_vector * height
        + radii * y_unit_vector * np.cos(theta)
        + np.sin(theta) * np.cross(z_unit_vector, radii * y_unit_vector)
    )
    assert np.allclose(np.linalg.norm(tx_xyz, axis=1), focal_length)
    return tx_xyz


class RUN_SIM(babel_integration_focused_array.RUN_SIM):
    pass


class BabelFTD_Simulations(babel_integration_focused_array.BabelFTD_Simulations):
    pass


class SimulationConditions(babel_integration_focused_array.SimulationConditions):
    '''
    # Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    # '''
    
    def GenTransducerGeom(self,PPWSurface=None):
        if PPWSurface is None:
            PPWSurface = self.PPW_SURFACE

        radii = np.array(self._elements["radii"]).reshape(self._num_elements,1)
        thetas = np.deg2rad(np.array(self._elements["thetas"]).reshape(self._num_elements,1))
        tx_xyz = compute_H301_xyz_coords(radii,thetas,self._FocalLength)

        self._Tx = generate_focused_array_tx(tx_xyz, self._num_elements, self._Frequency, self._FocalLength, self._element_size, validate_elements=False, sos=SpeedofSoundWater(20.0),rotation_z=self._RotationZ, coordinate_sys="cartesian",show_plot=False,ppw_surface=PPWSurface)
        self._TxOrig = generate_focused_array_tx(tx_xyz, self._num_elements, self._Frequency, self._OrigFocalLength, self._original_element_size, validate_elements=False, sos=SpeedofSoundWater(20.0),rotation_z=self._RotationZ, coordinate_sys="cartesian",show_plot=False,ppw_surface=PPWSurface)

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations