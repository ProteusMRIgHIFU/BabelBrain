"""
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - May 19, 2022

"""

import os
from sys import platform

from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, SpeedofSoundWater
import matplotlib.pyplot as plt
import numpy as np
from stl import mesh
from trimesh import creation

from TranscranialModeling.babel_integration.integration_templates.babel_integration_base import (
    RUN_SIM_BASE,
    _rec_artifact,
    BabelFTD_Simulations_BASE,
    SimulationConditionsBASE,
    Material,
)
from TranscranialModeling.tx_geometries import generate_flat_annular_array_tx


class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self, **kargs):
        return self._BabelFTDSimClass(
            ZSteering=self._ZSteering,
            Aperture=self._Aperture,
            InDiameters=self._InDiameters,
            OutDiameters=self._OutDiameters,
            **kargs,
        )

    def RunCases(
        self,
        ZSteering=0.0,
        Aperture=0.0,
        InDiameters=[0.0],
        OutDiameters=[0.0],
        **kargs,
    ):
        self._ZSteering = ZSteering
        self._Aperture = Aperture
        self._InDiameters = InDiameters
        self._OutDiameters = OutDiameters
        return super().RunCases(**kargs)


class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    # Meta class dealing with the specificis of each test based on the string name
    def __init__(
        self,
        ZSteering=0.0,
        Aperture=0.0,
        InDiameters=[0.0],
        OutDiameters=[0.0],
        **kargs,
    ):
        self._ZSteering = ZSteering
        self._Aperture = Aperture
        self._InDiameters = InDiameters
        self._OutDiameters = OutDiameters
        super().__init__(**kargs)

    def CreateSimConditions(self, **kargs):
        return self._SimConditionsClass(
            ZSteering=self._ZSteering,
            Aperture=self._Aperture,
            InDiameters=self._InDiameters,
            OutDiameters=self._OutDiameters,
            FocalLength=0.0,
            **kargs,
        )

    def GenerateSTLTx(self, prefix):
        affine = self._SkullMask.affine
        LocSpot = np.array(
            np.where(self._SkullMask.get_fdata(dtype=np.float32) == 5.0)
        ).flatten()

        allMeshes = []
        TxElemCenters = []

        for VertDisplay, FaceDisplay in zip(
            self._SIM_SETTINGS._Tx["RingVertDisplay"],
            self._SIM_SETTINGS._Tx["RingFaceDisplay"],
        ):
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
            TxElemCenters.append(
                np.mean(TxVert, axis=0)
            )  # In ring Tx, we use the center of mass of all vertices
            for i, f in enumerate(FaceDisplay):
                TxStl.vectors[i * 2][0] = TxVert[f[0], :]
                TxStl.vectors[i * 2][1] = TxVert[f[1], :]
                TxStl.vectors[i * 2][2] = TxVert[f[3], :]

                TxStl.vectors[i * 2 + 1][0] = TxVert[f[1], :]
                TxStl.vectors[i * 2 + 1][1] = TxVert[f[2], :]
                TxStl.vectors[i * 2 + 1][2] = TxVert[f[3], :]

            allMeshes.append(TxStl.data)

        bdir = os.path.dirname(self._MASKFNAME)
        FinalMesh = mesh.Mesh(np.concatenate(allMeshes))
        FinalMesh.save(bdir + os.sep + prefix + "Tx.stl")
        _rec_artifact(bdir + os.sep + prefix + "Tx.stl")
        TransformationCone = np.eye(4)
        TransformationCone[2, 2] = -1
        OrientVec = np.array([0, 0, 1]).reshape((1, 3))
        TransformationCone[0, 3] = LocSpot[0] + int(
            np.round(self._TxMechanicalAdjustmentX / self._SIM_SETTINGS.SpatialStep)
        )
        TransformationCone[1, 3] = LocSpot[1] + int(
            np.round(self._TxMechanicalAdjustmentY / self._SIM_SETTINGS.SpatialStep)
        )
        RadCone = self._SIM_SETTINGS._OrigAperture / self._SIM_SETTINGS.SpatialStep / 2
        HeightCone = self._ZSteering / self._SIM_SETTINGS.SpatialStep
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
        self._TxElemCenters = np.array(TxElemCenters)

    def AddSaveDataSim(self, DataForSim):
        super().AddSaveDataSim(DataForSim)
        DataForSim["TransducerType"] = "FlatAnnularArray"
        DataForSim["ZSteering"] = self._ZSteering
        DataForSim["BasePhasedArrayProgramming"] = (
            self._SIM_SETTINGS.BasePhasedArrayProgramming
        )


########################################################
########################################################
class SimulationConditions(SimulationConditionsBASE):
    """
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    """

    PPW_SURFACE = 8  # default points-per-wavelength for meshing the tx surface; overrideable by subclasses

    def __init__(
        self,
        FactorEnlarge=1,  # putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
        Aperture=0.0,  # m, aperture of the Tx, used to calculated cross section area entering the domain
        FocalLength=0.0,
        ZSteering=0.0,
        InDiameters=np.array([0.0]),  # inner diameter of rings
        OutDiameters=np.array([0.0]),  # outer diameter of rings
        **kargs,
    ):  # steering
        super().__init__(Aperture=Aperture * FactorEnlarge, FocalLength=0, **kargs)
        self._FactorEnlarge = FactorEnlarge
        self._OrigAperture = Aperture
        self._OrigFocalLength = FocalLength
        self._OrigInDiameters = InDiameters
        self._OrigOutDiameters = OutDiameters
        self._Aperture = Aperture * FactorEnlarge
        self._FocalLength = FocalLength * FactorEnlarge
        self._InDiameters = InDiameters * FactorEnlarge
        self._OutDiameters = OutDiameters * FactorEnlarge
        self._ZSteering = ZSteering

    def GenTx(self, bOrigDimensions=False, PPWSurface=None):
        fScaling = 1.0
        if bOrigDimensions:
            fScaling = self._FactorEnlarge
        if PPWSurface is None:
            PPWSurface = self.PPW_SURFACE
        print(
            "self._InDiameters, self._OutDiameters,self._FocalLength",
            self._InDiameters / fScaling,
            self._OutDiameters / fScaling,
            self._FocalLength / fScaling,
        )
        FocalLengthFlat = 1e3
        TxRC = generate_flat_annular_array_tx(
            self._Frequency,
            self._Aperture / fScaling,
            FocalLengthFlat,
            self._InDiameters / fScaling,
            self._OutDiameters / fScaling,
            SpeedofSoundWater(20.0),
            ppw_surface=PPWSurface,
        )

        return TxRC

    def CalculateRayleighFieldsForward(self, deviceName="6800"):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        # first we generate the high res source of the tx elements
        self._Tx = self.GenTx()

        if self._TxMechanicalAdjustmentZ < 0:
            zCorrec = self._TxMechanicalAdjustmentZ
        else:
            zCorrec = 0.0

        for Tx in [self._Tx]:
            for k in ["center", "RingVertDisplay", "elemcenter"]:
                if k == "RingVertDisplay":
                    for n in range(len(Tx[k])):
                        Tx[k][n][:, 0] += self._TxMechanicalAdjustmentX
                        Tx[k][n][:, 1] += self._TxMechanicalAdjustmentY
                        Tx[k][n][:, 2] = (
                            self._ZDim[self._ZSourceLocation]
                            - self._SkullMaskNii.header.get_zooms()[2] / 1e3
                            + zCorrec
                        )
                else:
                    Tx[k][:, 0] += self._TxMechanicalAdjustmentX
                    Tx[k][:, 1] += self._TxMechanicalAdjustmentY
                    Tx[k][:, 2] = (
                        self._ZDim[self._ZSourceLocation]
                        - self._SkullMaskNii.header.get_zooms()[2] / 1e3
                        + zCorrec
                    )

        print(
            "self._Tx['center'].max()",
            self._Tx["center"][:, 2].max(),
            self._ZDim[self._ZSourceLocation],
        )
        # we apply an homogeneous pressure

        cwvnb_extlay = np.array(
            2 * np.pi * self._Frequency / Material["Water"][1] + 1j * 0
        ).astype(np.complex64)

        # we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming = np.zeros(
            self._Tx["NumberElems"], np.complex64
        )

        print("Running Steering")
        ds = np.ones((1)) * self._SpatialStep**2

        center = np.zeros((1, 3), np.float32)
        # to avoid adding an erroneous steering to the calculations, we need to discount the mechanical motion
        center[0, 0] = (
            self._XDim[self._FocalSpotLocation[0]] + self._TxMechanicalAdjustmentX
        )
        center[0, 1] = (
            self._YDim[self._FocalSpotLocation[1]] + self._TxMechanicalAdjustmentY
        )
        center[0, 2] = self._ZDim[self._ZSourceLocation] + self._ZSteering + zCorrec

        print(
            "center",
            center,
            self._Tx["center"][:, 2].max(),
            self._ZDim[self._ZSourceLocation],
            self._Tx["center"][:, 2].max() - center[0, 2],
        )
        print("self._ZSourceLocation", self._ZSourceLocation)
        u2back = np.zeros(self._Tx["NumberElems"], np.complex64)
        nBase = 0
        for n in range(self._Tx["NumberElems"]):
            u0 = np.ones(self._Tx["elemdims"][n][0], np.complex64)
            SelCenters = self._Tx["center"][
                nBase : nBase + self._Tx["elemdims"][n][0], :
            ].astype(np.float32)
            SelDs = self._Tx["ds"][
                nBase : nBase + self._Tx["elemdims"][n][0], :
            ].astype(np.float32)
            u2back[n] = ForwardSimple(
                cwvnb_extlay, SelCenters, SelDs, u0, center, deviceMetal=deviceName
            )[0]
            nBase += self._Tx["elemdims"][n][0]

        AllPhi = np.zeros(self._Tx["NumberElems"])
        for n in range(self._Tx["NumberElems"]):
            self.BasePhasedArrayProgramming[n] = np.exp(-1j * np.angle(u2back[n]))
            phi = -np.angle(u2back[n])
            AllPhi[n] = phi

        self.BasePhasedArrayProgramming = np.exp(1j * AllPhi)
        print("Phase for array: [", np.rad2deg(AllPhi).tolist(), "]")
        u0 = np.zeros((self._Tx["center"].shape[0], 1), np.complex64)
        nBase = 0
        for n in range(self._Tx["NumberElems"]):
            u0[nBase : nBase + self._Tx["elemdims"][n][0]] = (
                self._SourceAmpPa * np.exp(1j * AllPhi[n])
            ).astype(np.complex64)
            nBase += self._Tx["elemdims"][n][0]

        nxf = len(self._XDim)
        nyf = len(self._YDim)
        nzf = len(self._ZDim)

        xp, yp, zp = np.meshgrid(self._XDim, self._YDim, self._ZDim, indexing="ij")

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

        self._SourceMapRayleigh = u2[:, :, self._ZSourceLocation]

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
