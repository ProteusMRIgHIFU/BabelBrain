# This Python file uses the following encoding: utf-8
import os

import numpy as np
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox
from trimesh import creation

from babel_transducers.simple_focused.BSonix.BSonix_form import BSonixForm
from babel_transducers.transducer_templates.babel_simple_focused_tx import SimpleFocusedTx, DistanceOutPlaneToFocus
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars
from Utils.paths import resource_path


class BSonix(SimpleFocusedTx):
    def __init__(self, parent=None, MainApp=None):
        config_file = os.path.join(resource_path(__file__), "default.yaml")
        super().__init__(parent,MainApp,config_file,BSonixForm)

    def _WirePanel(self):
        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)
        
        self.Widget.SkinDistanceSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.TransducerModelcomboBox.currentIndexChanged.connect(self.UpdateTxInfo)
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.CalculateMechAdj.clicked.connect(self.CalculateMechAdj)
        self.Widget.CalculateMechAdj.setEnabled(False)

    def NotifyGeneratedMask(self):
        super().NotifyGeneratedMask()
        self.Widget.SkinDistanceSpinBox.setValue(0.0)

    def GetTxModel(self):
        return "BSonix"+self.Widget.TransducerModelcomboBox.currentText()

    def UpdateLimits(self):
        model=self.GetTxModel()
        FocalLength = self.Config[model]['FocalLength']*1e3
        Diameter = self.Config[model]['TxDiam']*1e3
        DOut=DistanceOutPlaneToFocus(FocalLength,Diameter)-self.Config[model]['AdjustDistanceSkin']*1e3
        ZMax=DOut-self.Widget.DistanceSkinLabel.property('UserData')
        self._ZMaxSkin = np.round(ZMax,1)
        self.Widget.SkinDistanceSpinBox.setMaximum(self.Config['MaxDistanceToSkin'])
        self.Widget.SkinDistanceSpinBox.setMinimum(-self.Config['MaxNegativeDistance']) 
        self.UpdateDistanceLabels()

    def GetExtraSuffixAcFields(self):
        #By default, it returns empty string, useful when dealing with user-specified geometry
        model=self.GetTxModel()
        return model+'_'
    
    def GetExport(self):
        Export=super(BSonix,self).GetExport()
        Export['TxModel']=self.GetTxModel()
        return Export

    @Slot()
    def _ResolveSimulationFilenames(self):
        self._extrasuffix=self.GetExtraSuffixAcFields()
        model=self.GetTxModel()
        self._FocalLength = self.Config[model]['FocalLength']*1e3
        self._Diameter = self.Config[model]['TxDiam']*1e3
        self._prefix=self._MainApp._prefix_path[self._TrajectoryNumber]+model
        self._FullSolName=self._prefix+'_DataForSim.h5'
        self._WaterSolName=self._prefix+'_Water_DataForSim.h5'
        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)

    def _PromptReuseOrRecalc(self):
        Skull=ReadFromH5py(self._FullSolName)

        DistanceSkin = self._ZMaxSkin - Skull['TxMechanicalAdjustmentZ']*1e3

        ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                 "DistanceSkin=%3.2f\n" %(DistanceSkin)+
                                "Do you want to recalculate?\nSelect No to reload",
            QMessageBox.Yes | QMessageBox.No)

        if ret == QMessageBox.Yes:
            return True
        self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
        self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
        self.Widget.SkinDistanceSpinBox.setValue(DistanceSkin)
        if 'zLengthBeyonFocalPoint' in Skull:
            self.Widget.MaxDepthSpinBox.setValue(Skull['zLengthBeyonFocalPoint']*1e3)
        return False

    # BSonix reuses SingleTx._CreateAcousticWorker (same RunAcousticSim) but
    # runs its own post-processing step on completion instead of UpdateAcResults.
    def _SimulationFinishedSlot(self):
        return self.DoneAcSim

    def _ReloadExistingResults(self):
        self.DoneAcSim()

    def DoneAcSim(self):
        RADIUS = self.Config['CaseDiameter']/2/1e3 # m dimension of "puck"
        HEIGHT = self.Config['CaseHeight']/1e3 # m
        InitTrans=np.eye(4)
        LocSpot=np.array(np.where(np.flip(self._MainApp._FinalMask[self._TrajectoryNumber],axis=2)==5.0)).flatten()
        SpatialStep=np.mean(self._MainApp._MaskNib[0].header.get_zooms())/1e3
        InitTrans[0,3]=LocSpot[0]
        InitTrans[1,3]=LocSpot[1]
        InitTrans[2,3]=LocSpot[2]+HEIGHT/2/SpatialStep+(self.Widget.DistanceSkinLabel.property('UserData')+self.Widget.SkinDistanceSpinBox.value())/1e3/SpatialStep
        #we create first a cylinder in voxel dimensions
        cylinder = creation.cylinder(radius=RADIUS/SpatialStep,height=HEIGHT/SpatialStep,sections=20,transform=InitTrans)
        
        affine=self._MainApp._MaskNib[self._TrajectoryNumber].affine.copy()
        
        #and we apply the conversion from voxel space to subject space in mm
        cylinder.apply_transform(affine)

        cylinder.export(self._prefix+'_BSonixCase.stl')
        self.UpdateAcResults()
