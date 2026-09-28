"""
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - May 19, 2022

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, SpeedofSoundWater
from scipy.io import loadmat
from stl import mesh
from trimesh import creation

from TranscranialModeling.babel_integration.integration_templates.babel_integration_base import (
    RUN_SIM_BASE,
    _rec_artifact,
    BabelFTD_Simulations_BASE,
    SimulationConditionsBASE,
    Material,
)
from TranscranialModeling.tx_geometries import generate_curved_element


class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self, **kargs):
        return self._BabelFTDSimClass(**kargs)

    def RunCases(self, **kargs):
        self._Aperture = kargs["Aperture"]
        self._FocalLength = kargs["FocalLength"]
        return super().RunCases(**kargs)


class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    # Meta class dealing with the specificis of each test based on the string name
    def __init__(self, Aperture=0.0, FocalLength=0.0, **kargs):
        self._Aperture = Aperture
        self._FocalLength = FocalLength
        super().__init__(**kargs)

    def CreateSimConditions(self, **kargs):
        return self._SimConditionsClass(
            Aperture=self._Aperture, FocalLength=self._FocalLength, **kargs
        )

    def GenerateSTLTx(self, prefix):
        n = 1
        LocSpot = np.array(
            np.where(self._SkullMask.get_fdata(dtype=np.float32) == 5.0)
        ).flatten()
        affine = self._SkullMask.affine
        VertDisplay = self._SIM_SETTINGS._Tx["VertDisplay"]
        FaceDisplay = self._SIM_SETTINGS._Tx["FaceDisplay"]

        # we also export the STL of the Tx for display in Brainsight or 3D slicer
        TxVert = VertDisplay.T.copy()
        TxVert[2, :] -= self._SIM_SETTINGS._ZDim[self._SIM_SETTINGS._PMLThickness]
        TxVert[2, :] = -TxVert[2, :]
        TxVert /= self._SIM_SETTINGS.SpatialStep
        TxVert[2, :] += self._SkullMask.shape[2] - 1
        TxVert = np.vstack([TxVert, np.ones((1, TxVert.shape[1]))])

        TxVert[0, :] += LocSpot[0]
        TxVert[1, :] += LocSpot[1]

        TxVert = np.dot(affine, TxVert)

        TxStl = mesh.Mesh(np.zeros(FaceDisplay.shape[0] * 2, dtype=mesh.Mesh.dtype))

        TxVert = TxVert.T[:, :3]
        self._TxElemCenters = np.mean(TxVert, axis=0).reshape((1, 3))
        for i, f in enumerate(FaceDisplay):
            TxStl.vectors[i * 2][0] = TxVert[f[0], :]
            TxStl.vectors[i * 2][1] = TxVert[f[1], :]
            TxStl.vectors[i * 2][2] = TxVert[f[3], :]

            TxStl.vectors[i * 2 + 1][0] = TxVert[f[1], :]
            TxStl.vectors[i * 2 + 1][1] = TxVert[f[2], :]
            TxStl.vectors[i * 2 + 1][2] = TxVert[f[3], :]

        bdir = os.path.dirname(self._MASKFNAME)
        TxStl.save(bdir + os.sep + prefix + "Tx.stl")
        _rec_artifact(bdir + os.sep + prefix + "Tx.stl")

        TransformationCone = np.eye(4)
        TransformationCone[2, 2] = -1
        OrientVec = np.array([0, 0, 1]).reshape((1, 3))
        TransformationCone[0, 3] = LocSpot[0]
        TransformationCone[1, 3] = LocSpot[1]
        RadCone = self._SIM_SETTINGS._OrigAperture / self._SIM_SETTINGS.SpatialStep / 2
        HeightCone = (
            self._SIM_SETTINGS._FocalLength
            / self._SIM_SETTINGS._FactorEnlarge
            / self._SIM_SETTINGS.SpatialStep
        )
        HeightCone = np.sqrt(HeightCone**2 - RadCone**2)
        TransformationCone[2, 3] = (
            LocSpot[2]
            + HeightCone
            - self._SIM_SETTINGS._TxMechanicalAdjustmentZ
            / self._SIM_SETTINGS.SpatialStep
        )
        Cone = creation.cone(RadCone, HeightCone, transform=TransformationCone)
        Cone.apply_transform(affine)
        # we save the final cone profile
        Cone.export(bdir + os.sep + prefix + "_Cone.stl")
        _rec_artifact(bdir + os.sep + prefix + "_Cone.stl")

    def AddSaveDataSim(self, DataForSim):
        super().AddSaveDataSim(DataForSim)
        DataForSim["TransducerType"] = "SingleElement"
        DataForSim["Aperture"] = self._Aperture
        DataForSim["FocalLength"] = self._FocalLength


########################################################
########################################################
class SimulationConditions(SimulationConditionsBASE):
    """
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    """

    PPW_SURFACE = 5  # default points-per-wavelength for meshing the tx surface; overrideable by subclasses

    def __init__(
        self,
        FactorEnlarge=1.0,  # putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
        Aperture=0.0,  # m, aperture of the Tx, used to calculated cross section area entering the domain
        FocalLength=0.0,
        **kargs,
    ):  # steering
        super().__init__(
            Aperture=Aperture * FactorEnlarge,
            FocalLength=FocalLength * FactorEnlarge,
            **kargs,
        )
        self._FactorEnlarge = FactorEnlarge
        self._OrigAperture = Aperture
        self._OrigFocalLength = FocalLength
        self._Aperture = Aperture * FactorEnlarge
        self._FocalLength = FocalLength * FactorEnlarge

    def GenTx(self, bOrigDimensions=False, PPWSurface=None):
        fScaling = 1.0
        if bOrigDimensions:
            fScaling = self._FactorEnlarge
        if PPWSurface is None:
            PPWSurface = self.PPW_SURFACE
        TxRC = generate_curved_element(
            self._Frequency,
            self._FocalLength / fScaling,
            self._Aperture / fScaling,
            SpeedofSoundWater(20.0),
            ppw_surface=PPWSurface,
        )
        TxRC["Aperture"] = self._Aperture / fScaling
        TxRC["center"][:, 2] += self._FocalLength / fScaling
        TxRC["elemcenter"][:, 2] += self._FocalLength / fScaling
        TxRC["VertDisplay"][:, 2] += self._FocalLength / fScaling
        return TxRC

    def CalculateRayleighFieldsForward(self, deviceName="6800"):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        # first we generate the high res source of the tx elements
        self._Tx = self.GenTx()

        ZDomainStart = self.CalculateDomainZReference()

        print("Init Location of back Tx in Z", self._Tx["center"][:, 2].min())

        for Tx in [self._Tx]:
            for k in ["center", "VertDisplay", "elemcenter"]:
                Tx[k][:, 0] += self._TxMechanicalAdjustmentX
                Tx[k][:, 1] += self._TxMechanicalAdjustmentY
                Tx[k][:, 2] += self._TxMechanicalAdjustmentZ + ZDomainStart
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

        # we apply an homogeneous pressure
        print("Location of back Tx in Z", self._Tx["center"][:, 2].min())
        print("Location of source layer Z", self._ZDim[self._ZSourceLocation])

        cwvnb_extlay = np.array(
            2 * np.pi * self._Frequency / Material["Water"][1] + 1j * 0
        ).astype(np.complex64)

        u0 = (
            np.ones((self._Tx["center"].shape[0], 1), np.float32)
            + 1j * np.zeros((self._Tx["center"].shape[0], 1), np.float32)
        ) * self._SourceAmpPa
        nxf = len(self._XDim)
        nyf = len(self._YDim)
        nzf = len(self._ZDim)
        yp, xp, zp = np.meshgrid(self._YDim, self._XDim, self._ZDim)

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

        if len(self._BenchmarkTestFile) > 0 and len(self._InputFocusStart) > 0:
            print("Loading input focus from", self._InputFocusStart)
            # we load the input focus from the file
            InputFocus = loadmat(self._InputFocusStart)
            self._SourceMapRayleigh[
                self._PMLThickness : -self._PMLThickness,
                self._PMLThickness : -self._PMLThickness,
            ] = InputFocus["sourceplane"]

    def CreateSources(self, ramp_length=4):
        # we create the list of functions sources taken from the Rayliegh incident field
        LengthSource = (
            np.floor(self._TimeSimulation / (1.0 / self._Frequency))
            * 1
            / self._Frequency
        )
        TimeVectorSource = np.arange(
            0, LengthSource + self._TemporalStep, self._TemporalStep
        )
        # we do as in k-wave to create a ramped signal

        ramp_length_points = int(
            np.round(ramp_length / self._Frequency / self._TemporalStep)
        )
        ramp_axis = np.arange(0, np.pi, np.pi / ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points = len(ramp)

        self._SourceMap = np.zeros((self._N1, self._N2, self._N3), np.uint32)
        LocZ = self._ZSourceLocation

        SourceMaskIND = np.where(np.abs(self._SourceMapRayleigh) > 0)
        SourceMask = np.zeros((self._N1, self._N2), np.uint32)

        RefI = (
            int((SourceMaskIND[0].max() - SourceMaskIND[0].min()) / 2)
            + SourceMaskIND[0].min()
        )
        RefJ = (
            int((SourceMaskIND[1].max() - SourceMaskIND[1].min()) / 2)
            + SourceMaskIND[1].min()
        )
        AngRef = np.angle(self._SourceMapRayleigh[RefI, RefJ])
        PulseSource = np.zeros(
            (np.sum(np.abs(self._SourceMapRayleigh) > 0), TimeVectorSource.shape[0])
        )
        nSource = 1
        for i, j in zip(SourceMaskIND[0], SourceMaskIND[1]):
            SourceMask[i, j] = nSource
            u0 = self._SourceMapRayleigh[i, j]
            # we recover amplitude and phase from Rayleigh field
            PulseSource[nSource - 1, :] = np.abs(u0) * np.sin(
                2 * np.pi * self._Frequency * TimeVectorSource + np.angle(u0)
            )
            PulseSource[nSource - 1, : int(ramp_length_points)] *= ramp
            nSource += 1
        self._SourceMap[:, :, LocZ] = SourceMask

        self._PulseSource = PulseSource

        if self._bDisplay:
            plt.figure(figsize=(6, 3))
            for n in range(1, 4):
                plt.plot(
                    TimeVectorSource * 1e6,
                    PulseSource[int(PulseSource.shape[0] / 4) * n, :],
                )
                plt.title("CW signal, example %i" % (n))

            plt.xlim(0, 50)

            plt.figure(figsize=(3, 2))
            plt.imshow(self._SourceMap[:, :, LocZ])
            plt.title("source map - source ids")

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
