"""
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

"""

import os

from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, SpeedofSoundWater
import matplotlib.pyplot as plt
import numpy as np
from stl import mesh

from TranscranialModeling.babel_integration.integration_templates.babel_integration_base import (
    RUN_SIM_BASE,
    _rec_artifact,
    BabelFTD_Simulations_BASE,
    SimulationConditionsBASE,
    Material,
)
from TranscranialModeling.tx_geometries import generate_flat_array_2d_tx


class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self, **kargs):
        return self._BabelFTDSimClass(
            XSteering=self._XSteering,
            YSteering=self._YSteering,
            ZSteering=self._ZSteering,
            RotationZ=self._RotationZ,
            Aperture=self._Aperture,
            **kargs,
        )

    def RunCases(
        self,
        XSteering=0.0,
        YSteering=0.0,
        ZSteering=0.0,
        RotationZ=0.0,
        Aperture=0.0,
        **kargs,
    ):
        self._RotationZ = RotationZ
        self._XSteering = XSteering
        self._YSteering = YSteering
        self._ZSteering = ZSteering
        self._Aperture = Aperture

        return super().RunCases(**kargs)


##########################################


class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    # Meta class dealing with the specificis of each test based on the string name
    def __init__(
        self,
        XSteering=0.0,
        YSteering=0.0,
        ZSteering=0.0,
        RotationZ=0.0,
        Aperture=0.0,
        elements=[],
        num_elements=0,
        element_size=0,
        distance_outplane=0,
        **kargs,
    ):

        self._XSteering = XSteering
        self._YSteering = YSteering
        self._ZSteering = ZSteering
        self._RotationZ = RotationZ
        self._Aperture = Aperture
        self._elements = elements
        self._num_elements = num_elements
        self._element_size = element_size
        self._distance_outplane = distance_outplane
        super().__init__(**kargs)

    def CreateSimConditions(self, **kargs):
        return self._SimConditionsClass(
            XSteering=self._XSteering,
            YSteering=self._YSteering,
            ZSteering=self._ZSteering,
            RotationZ=self._RotationZ,
            FocalLength=0.0,
            Aperture=self._Aperture,  # m, aperture of the Tx, used tof calculated cross section area entering the domain
            elements=self._elements,
            num_elements=self._num_elements,
            element_size=self._element_size,
            distance_outplane=self._distance_outplane,
            **kargs,
        )

    def GenerateSTLTx(self, prefix):
        # we also export the STL of the Tx for display in Brainsight or 3D slicer

        affine = self._SkullMask.affine
        LocSpot = np.array(
            np.where(self._SkullMask.get_fdata(dtype=np.float32) == 5.0)
        ).flatten()

        for nt, st in enumerate(["VertDisplay", "elemcenter"]):
            TxVert = self._SIM_SETTINGS._Tx[st].T.copy()
            TxVert[2, :] -= self._SIM_SETTINGS._ZDim[self._SIM_SETTINGS._PMLThickness]
            TxVert[2, :] = -TxVert[2, :]
            TxVert /= self._SIM_SETTINGS.SpatialStep
            TxVert[2, :] += self._SkullMask.shape[2] - 1
            TxVert = np.vstack([TxVert, np.ones((1, TxVert.shape[1]))])

            TxVert[0, :] += LocSpot[0]
            TxVert[1, :] += LocSpot[1]

            TxVert = np.dot(affine, TxVert)

            TxVert = TxVert.T[:, :3]

            if nt == 0:
                TxStl = mesh.Mesh(
                    np.zeros(
                        self._SIM_SETTINGS._Tx["FaceDisplay"].shape[0] * 2,
                        dtype=mesh.Mesh.dtype,
                    )
                )
                for i, f in enumerate(self._SIM_SETTINGS._Tx["FaceDisplay"]):
                    TxStl.vectors[i * 2][0] = TxVert[f[0], :]
                    TxStl.vectors[i * 2][1] = TxVert[f[1], :]
                    TxStl.vectors[i * 2][2] = TxVert[f[3], :]

                    TxStl.vectors[i * 2 + 1][0] = TxVert[f[1], :]
                    TxStl.vectors[i * 2 + 1][1] = TxVert[f[2], :]
                    TxStl.vectors[i * 2 + 1][2] = TxVert[f[3], :]

                bdir = os.path.dirname(self._MASKFNAME)
                TxStl.save(bdir + os.sep + prefix + "Tx.stl")
                _rec_artifact(bdir + os.sep + prefix + "Tx.stl")
            else:
                self._TxElemCenters = TxVert

    def AddSaveDataSim(self, DataForSim):
        super().AddSaveDataSim(DataForSim)
        DataForSim["TransducerType"] = "FlatPhasedArray"
        DataForSim["XSteering"] = self._XSteering
        DataForSim["YSteering"] = self._YSteering
        DataForSim["ZSteering"] = self._ZSteering
        DataForSim["RotationZ"] = self._RotationZ
        DataForSim["bDoRefocusing"] = self._bDoRefocusing
        DataForSim["BasePhasedArrayProgrammingRefocusing"] = (
            self._SIM_SETTINGS.BasePhasedArrayProgrammingRefocusing
        )
        DataForSim["BasePhasedArrayProgramming"] = (
            self._SIM_SETTINGS.BasePhasedArrayProgramming
        )


class SimulationConditions(SimulationConditionsBASE):
    """
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    """

    PPW_SURFACE = 12.0  # default points-per-wavelength for meshing the tx surface; overrideable by subclasses

    def __init__(
        self,
        Aperture=0.0,  # m, aperture of the Tx, used tof calculated cross section area entering the domain
        FocalLength=0.0,
        XSteering=0.0,  # lateral steering
        YSteering=0.0,
        ZSteering=0.0,
        RotationZ=0.0,  # rotation of Tx over Z axis
        elements=[],
        num_elements=0,
        element_size=0,
        distance_outplane=0.0,
        **kargs,
    ):
        super().__init__(
            Aperture=Aperture,
            FocalLength=FocalLength,
            ZTxCorrecton=distance_outplane,  # this will put the required water space in the simulation domain
            **kargs,
        )
        self._XSteering = XSteering
        self._YSteering = YSteering
        self._ZSteering = ZSteering
        self._RotationZ = RotationZ
        self._elements = elements
        self._num_elements = num_elements
        self._element_size = element_size
        self._Aperture = Aperture
        self._zdistance = -distance_outplane

    def GenTransducerGeom(self, subset_indices=None, PPWSurface=None):
        if PPWSurface is None:
            PPWSurface = self.PPW_SURFACE
        element_positions = np.column_stack(
            (self._elements["x"], self._elements["y"], self._elements["z"])
        )
        if subset_indices is not None:
            element_positions = element_positions[subset_indices]
            num_elements = len(subset_indices)
        else:
            num_elements = self._num_elements
        return generate_flat_array_2d_tx(
            element_positions,
            num_elements,
            self._element_size,
            deadspace=self._zdistance,
            rotation_z=self._RotationZ,
            frequency=self._Frequency,
            ppw_surface=PPWSurface,
            sos=SpeedofSoundWater(20.0)
        )

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
            center[0, 0] = (
                self._XDim[self._FocalSpotLocation[0]]
                + self._TxMechanicalAdjustmentX
                + self._XSteering
            )
            center[0, 1] = (
                self._YDim[self._FocalSpotLocation[1]]
                + self._TxMechanicalAdjustmentY
                + self._YSteering
            )
            center[0, 2] = self._ZDim[self._ZSourceLocation] + self._ZSteering + zCorrec

            print(
                "center",
                center,
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

        ## Now we create the sources for back propagation

        self._PunctualSource = np.sin(
            2 * np.pi * self._Frequency * TimeVectorSource
        ).reshape(1, len(TimeVectorSource))
        self._PunctualSource[0, : int(ramp_length_points)] *= ramp
        self._PunctualSource[0, -int(ramp_length_points) :] *= np.flip(ramp)
        self._SourceMapPunctual = np.zeros((self._N1, self._N2, self._N3), np.uint32)
        LocForRefocusing = self._FocalSpotLocation.copy()
        # LocForRefocusing[2]=0.0
        # LocForRefocusing[0]+=int(np.round(self._XSteering/self._SpatialStep))
        # LocForRefocusing[1]+=int(np.round(self._YSteering/self._SpatialStep))
        # LocForRefocusing[2]+=int(np.round(self._ZSteering/self._SpatialStep))
        self._SourceMapPunctual[
            LocForRefocusing[0], LocForRefocusing[1], LocForRefocusing[2]
        ] = 1

        if self._bDisplay:
            plt.figure(figsize=(12, 4))
            for n in range(1, 4):
                plt.plot(
                    TimeVectorSource * 1e6,
                    PulseSource[int(PulseSource.shape[0] / 4) * n, :],
                )
                plt.title("CW signal, example %i" % (n))

            plt.xlim(0, 50)

            plt.figure(figsize=(5, 4))
            plt.imshow(self._SourceMap[:, :, LocZ])
            plt.title("source map - source ids")

    def BackPropagationRayleigh(self, deviceName="6800"):
        assert np.all(
            np.array(self._SourceMapRayleigh.shape)
            == np.array(self._PressMapFourierBack.shape)
        )
        SelRegRayleigh = np.abs(self._SourceMapRayleigh) > 0
        ypp, xpp = np.meshgrid(self._YDim, self._XDim)
        ypp = ypp[SelRegRayleigh]
        xpp = xpp[SelRegRayleigh]
        center = np.zeros((ypp.size, 3), np.float32)
        center[:, 0] = xpp.flatten()
        center[:, 1] = ypp.flatten()
        center[:, 2] = self._ZDim[self._ZSourceLocation]

        ds = np.ones((center.shape[0])) * self._SpatialStep**2

        # we apply an homogeneous pressure
        u0 = self._PressMapFourierBack[SelRegRayleigh]

        cwvnb_extlay = np.array(
            2 * np.pi * self._Frequency / Material["Water"][1] + 1j * 0
        ).astype(np.complex64)

        u2back = ForwardSimple(
            cwvnb_extlay,
            center.astype(np.float32),
            ds.astype(np.float32),
            u0,
            self._Tx["elemcenter"].astype(np.float32),
            deviceMetal=deviceName,
        )

        # now we calculate forward back

        u0 = np.zeros((self._Tx["center"].shape[0], 1), np.complex64)
        nBase = 0
        for n in range(self._Tx["NumberElems"]):
            phi = np.angle(np.conjugate(u2back[n]))
            self.BasePhasedArrayProgrammingRefocusing[n] = np.conjugate(u2back[n])
            u0[nBase : nBase + self._Tx["elemdims"]] = (
                self._SourceAmpPa * np.exp(1j * phi)
            ).astype(np.complex64)
            nBase += self._Tx["elemdims"]

        nxf = len(self._XDim)
        nyf = len(self._YDim)
        nzf = len(self._ZDim)
        # ZDim=self._ZDim-self._ZDim[self._ZSourceLocation]+self._TxMechanicalAdjustmentZ

        xp, yp, zp = np.meshgrid(self._XDim, self._YDim, self._ZDim, indexing="ij")

        rf = np.hstack(
            (
                np.reshape(xp, (nxf * nyf * nzf, 1)),
                np.reshape(yp, (nxf * nyf * nzf, 1)),
                np.reshape(zp, (nxf * nyf * nzf, 1)),
            )
        ).astype(np.float32)

        u2 = ForwardSimple(
            cwvnb_extlay,
            self._Tx["center"].astype(np.float32),
            self._Tx["ds"].astype(np.float32),
            u0,
            rf,
            deviceMetal=deviceName,
        )
        u2 = np.reshape(u2, xp.shape)
        self._SourceMapRayleighRefocus = u2[:, :, self._ZSourceLocation].copy()
        self._SourceMapRayleighRefocus[: self._PMLThickness, :] = 0
        self._SourceMapRayleighRefocus[-self._PMLThickness :, :] = 0
        self._SourceMapRayleighRefocus[:, : self._PMLThickness] = 0
        self._SourceMapRayleighRefocus[:, -self._PMLThickness :] = 0

    def CreateSourcesRefocus(self, ramp_length=4):
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

        LocZ = self._ZSourceLocation

        SourceMaskIND = np.where(np.abs(self._SourceMapRayleigh) > 0)

        RefI = (
            int((SourceMaskIND[0].max() - SourceMaskIND[0].min()) / 2)
            + SourceMaskIND[0].min()
        )
        RefJ = (
            int((SourceMaskIND[1].max() - SourceMaskIND[1].min()) / 2)
            + SourceMaskIND[1].min()
        )
        AngRef = np.angle(self._SourceMapRayleighRefocus[RefI, RefJ])
        PulseSource = np.zeros(
            (
                np.sum(np.abs(self._SourceMapRayleighRefocus) > 0),
                TimeVectorSource.shape[0],
            )
        )
        nSource = 1
        for i, j in zip(SourceMaskIND[0], SourceMaskIND[1]):
            u0 = self._SourceMapRayleighRefocus[i, j]
            # we recover amplitude and phase from Rayleigh field
            PulseSource[nSource - 1, :] = np.abs(u0) * np.sin(
                2 * np.pi * self._Frequency * TimeVectorSource + np.angle(u0)
            )
            PulseSource[nSource - 1, : int(ramp_length_points)] *= ramp
            nSource += 1

        self._PulseSourceRefocus = PulseSource

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
