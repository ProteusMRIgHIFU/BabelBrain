"""
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

"""
import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple

from TranscranialModeling.babel_integration.integration_templates import (
    babel_integration_flat_array_2D,
)
from TranscranialModeling.babel_integration.integration_templates.babel_integration_base import Material

# Sector1 = first 128 elements, Sector2 = last 128 elements
REMOPD_SUBSETS = {
    "Sector1": np.arange(0, 128),
    "Sector2": np.arange(128, 256),
}

def DeviceFrameSteering(XSteering, YSteering, flip_y=False):
    """Map GUI electronic steering into REMOPD simulation-domain axes.

    Changed on remopd/feasible-traj (TW / Brainsight): hydrophone checks showed
    GUI +Y is opposite the device/domain +Y. Sam asked that this swap apply
    only when BabelBrain is launched from Brainsight, so Slicer and Localite
    keep the identity map until a shared convention exists. Mechanical X/Y
    are already domain coordinates and are not mapped here.
    """
    if flip_y:
        return XSteering, -YSteering
    return XSteering, YSteering

class RUN_SIM(babel_integration_flat_array_2D.RUN_SIM):
    def RunCases(self, TxSet="Total", bFlipSteeringY=False, **kargs):
        self._TxSet = TxSet
        self._bFlipSteeringY = bFlipSteeringY
        return super().RunCases(**kargs)

    def CreateSimObject(self, **kargs):
        return self._BabelFTDSimClass(
            XSteering=self._XSteering,
            YSteering=self._YSteering,
            ZSteering=self._ZSteering,
            RotationZ=self._RotationZ,
            Aperture=self._Aperture,
            TxSet=self._TxSet,
            bFlipSteeringY=self._bFlipSteeringY,
            **kargs,
        )


class BabelFTD_Simulations(babel_integration_flat_array_2D.BabelFTD_Simulations):
    def __init__(self, TxSet="Total", bFlipSteeringY=False, **kargs):
        self._TxSet = TxSet
        self._bFlipSteeringY = bFlipSteeringY
        super().__init__(**kargs)

    def CreateSimConditions(self, **kargs):
        return self._SimConditionsClass(
            XSteering=self._XSteering,
            YSteering=self._YSteering,
            ZSteering=self._ZSteering,
            RotationZ=self._RotationZ,
            FocalLength=0.0,
            Aperture=self._Aperture,
            elements=self._elements,
            num_elements=self._num_elements,
            element_size=self._element_size,
            distance_outplane=self._distance_outplane,
            TxSet=self._TxSet,
            bFlipSteeringY=self._bFlipSteeringY,
            **kargs,
        )

    def AddSaveDataSim(self, DataForSim):
        super().AddSaveDataSim(DataForSim)
        DataForSim["TxSet"] = self._TxSet


class SimulationConditions(babel_integration_flat_array_2D.SimulationConditions):
    """
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    """

    def __init__(self, TxSet="Total", bFlipSteeringY=False, **kargs):
        super().__init__(**kargs)
        self._TxSet = TxSet
        self._bFlipSteeringY = bFlipSteeringY

    def GenTransducerGeom(self):
        indices = REMOPD_SUBSETS.get(self._TxSet)  # None when TxSet == 'Total'
        
        # REMOPD specific z rotation
        phi = np.deg2rad(self._RotationZ)
        rotation_matrix_z = np.array([
            [np.cos(phi),-np.sin(phi),0],
            [np.sin(phi), np.cos(phi),0],
            [0,0,1]
        ])
        self._RotationZ = rotation_matrix_z
        
        return super().GenTransducerGeom(subset_indices=indices)

    def CalculateRayleighFieldsForward(self, deviceName="6800"):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        # first we generate the high res source of the tx elements
        # and we select the set based on input
        self._Tx = self.GenTransducerGeom()

        if self._TxMechanicalAdjustmentZ < 0:
            zCorrec = self._TxMechanicalAdjustmentZ
        else:
            zCorrec = 0.0

        for k in ["center", "elemcenter", "VertDisplay"]:
            self._Tx[k][:, 0] += self._TxMechanicalAdjustmentX
            self._Tx[k][:, 1] += self._TxMechanicalAdjustmentY
            self._Tx[k][:, 2] = (
                self._ZDim[self._ZSourceLocation]
                - self._SkullMaskNii.header.get_zooms()[2] / 1e3
                + zCorrec
            )

        Correction = 0.0
        while np.max(self._Tx["center"][:, 2]) >= self._ZDim[self._ZSourceLocation]:
            # at the most, we could be too deep only a fraction of a single voxel, in such case we just move the Tx back a single step
            for Tx in [self._Tx]:
                for k in ["center", "VertDisplay", "elemcenter"]:
                    Tx[k][:, 2] -= self._SkullMaskNii.header.get_zooms()[2] / 1e3
            Correction += self._SkullMaskNii.header.get_zooms()[2] / 1e3
        if Correction > 0:
            print("Warning: Need to apply correction to reposition Tx for", Correction)
        # if yet we are not there, we need to stop
        if np.max(self._Tx["center"][:, 2]) > self._ZDim[self._ZSourceLocation]:
            print(
                "np.max(self._Tx['center'][:,2]),self._ZDim[self._ZSourceLocation]",
                np.max(self._Tx["center"][:, 2]),
                self._ZDim[self._ZSourceLocation],
            )
            raise RuntimeError(
                "The Tx limit in Z is below the location of the layer for source location for forward propagation."
            )

        print("self._Tx['center'].min(axis=0)", self._Tx["center"].min(axis=0))
        print("self._Tx['elemcenter'].min(axis=0)", self._Tx["elemcenter"].min(axis=0))

        # we apply an homogeneous pressure

        cwvnb_extlay = np.array(
            2 * np.pi * self._Frequency / Material["Water"][1] + 1j * 0
        ).astype(np.complex64)

        # we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming = np.zeros(
            self._Tx["NumberElems"], np.complex64
        )
        self.BasePhasedArrayProgrammingRefocusing = np.zeros(
            self._Tx["NumberElems"], np.complex64
        )

        if self._XSteering != 0.0 or self._YSteering != 0.0 or self._ZSteering != 0.0:
            print("Running Steering")
            ds = np.ones((1)) * self._SpatialStep**2

            # we apply an homogeneous pressure
            u0 = np.zeros((1), np.complex64)
            u0[0] = 1 + 0j
            center = np.zeros((1, 3), np.float32)
            # GUI Y is stored unchanged in the H5; flip only the focus used
            # for phasing, and only when launched from Brainsight.
            steerX, steerY = DeviceFrameSteering(
                self._XSteering, self._YSteering, flip_y=self._bFlipSteeringY
            )
            center[0, 0] = (
                self._XDim[self._FocalSpotLocation[0]]
                + self._TxMechanicalAdjustmentX
                + steerX
            )
            center[0, 1] = (
                self._YDim[self._FocalSpotLocation[1]]
                + self._TxMechanicalAdjustmentY
                + steerY
            )
            center[0, 2] = self._ZDim[self._ZSourceLocation] + self._ZSteering + zCorrec

            print(
                "center",
                center,
                "device-frame XY",
                (steerX, steerY),
                np.mean(self._Tx["elemcenter"][:, 2]),
            )

            u2back = ForwardSimple(
                cwvnb_extlay,
                center,
                ds.astype(np.float32),
                u0,
                self._Tx["elemcenter"].astype(np.float32),
                deviceMetal=deviceName,
            )
            u0 = np.zeros((self._Tx["center"].shape[0], 1), np.complex64)
            nBase = 0
            for n in range(self._Tx["NumberElems"]):
                phi = np.angle(np.conjugate(u2back[n]))
                self.BasePhasedArrayProgramming[n] = np.conjugate(u2back[n])
                u0[nBase : nBase + self._Tx["elemdims"]] = (
                    self._SourceAmpPa * np.exp(1j * phi)
                ).astype(np.complex64)
                nBase += self._Tx["elemdims"]

        else:
            u0 = (
                np.ones((self._Tx["center"].shape[0], 1), np.float32)
                + 1j * np.zeros((self._Tx["center"].shape[0], 1), np.float32)
            ) * self._SourceAmpPa

        nxf = len(self._XDim)
        nyf = len(self._YDim)
        nzf = len(self._ZDim)
        xp, yp, zp = np.meshgrid(self._XDim, self._YDim, self._ZDim, indexing="ij")

        print("ZDim[self._ZSourceLocation]", self._ZDim[self._ZSourceLocation])

        rf = np.hstack(
            (
                np.reshape(xp, (nxf * nyf * nzf, 1)),
                np.reshape(yp, (nxf * nyf * nzf, 1)),
                np.reshape(zp, (nxf * nyf * nzf, 1)),
            )
        ).astype(np.float32)

        u0 *= self.AdjustWeightAmplitudes()

        u2 = ForwardSimple(
            cwvnb_extlay,
            self._Tx["center"].astype(np.float32),
            self._Tx["ds"].astype(np.float32),
            u0,
            rf,
            deviceMetal=deviceName,
        )
        u2 = np.reshape(u2, xp.shape)

        self._u2RayleighField = u2

        self._SourceMapRayleigh = u2[:, :, self._ZSourceLocation].copy()

        self._SourceMapRayleigh[: self._PMLThickness, :] = 0
        self._SourceMapRayleigh[-self._PMLThickness :, :] = 0
        self._SourceMapRayleigh[:, : self._PMLThickness] = 0
        self._SourceMapRayleigh[:, -self._PMLThickness :] = 0

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
