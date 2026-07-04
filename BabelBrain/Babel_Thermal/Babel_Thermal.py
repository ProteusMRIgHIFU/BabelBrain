# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
from multiprocessing import Process,Queue
from collections import UserDict

from PySide6.QtWidgets import (QApplication, QWidget,QGridLayout,
                QHBoxLayout,QVBoxLayout,QLineEdit,QDialog,QTextEdit,
                QGridLayout, QSpacerItem, QInputDialog, QFileDialog,QFrame,
                QErrorMessage, QMessageBox,QDialogButtonBox,QLabel,QTableWidgetItem,
                QTabWidget)
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread,Qt
from PySide6 import QtCore,QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPalette, QTextCursor,QColor


import numpy as np

from scipy.io import loadmat
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,NavigationToolbar2QT)
from GUIComponents.AppStyle import style_nav_toolbar

import os
import sys
import shutil
from datetime import datetime
import time
import yaml
from ThermalModeling.CalculateTemperatureEffects import GetThermalOutName
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
from .CalculateThermalProcess import CalculateThermalProcess
import pandas as pd
import platform
import nibabel

_IS_MAC = platform.system() == 'Darwin'

# Sentinel marking a per-trajectory attribute that is absent on the controller
# (so _LoadThermalPanelState removes it rather than setting a stale value; this
# preserves the hasattr(...) guards used throughout _showMatplotlibVisualization).
_MISSING = object()

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_Thermal'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

def RCoeff(Temperature):
    R = np.ones(Temperature.shape)*0.25
    R[Temperature>=43]=0.5
    return R


class ThermalProfileConfig(UserDict):
    """Load a thermal YAML profile and normalize optional timing parameters."""

    OPTIONAL_DEFAULTS_DC_PRF = {
        'Repetitions': 1,
        'NumberGroupedSonications': 1,
        'PauseBetweenGroupedSonications': 0.0,
    }

    OPTIONAL_DEFAULTS = {
        'bConcatenateSimulations': False
    }

    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as file:
            config_data = yaml.safe_load(file) or {}
        super().__init__(config_data)
        self._apply_optional_defaults()

    def _apply_optional_defaults(self):
        for key, default_value in self.OPTIONAL_DEFAULTS.items():
            if key not in self.data:
                self.data[key] = default_value
            
        for timing_config in self.data.get('AllDC_PRF_Duration', []):
            for key, default_value in self.OPTIONAL_DEFAULTS_DC_PRF.items():
                if key not in timing_config:
                    timing_config[key] = default_value

class Babel_Thermal(QWidget):
    # Step-3 layout mirrors Step 2 (_BabelBaseTx): one tab per trajectory, each
    # an independent ThermalForm with its own controls, plot and results table.
    # self.Widget always points at the active tab's form and self._TrajectoryNumber
    # at its index; both are re-synced on tab change.  Per-trajectory computation
    # state (thermal H5 results, matplotlib figure/artists, mesh grids, \u2026) lives
    # in self._thPanels[i]; it is stashed/restored into the self._* aliases that
    # _showMatplotlibVisualization reads, so switching tabs keeps each trajectory's
    # results and plot intact (exactly like the Step-2 _acPanels mechanism).
    #
    # A Step-3 tab starts disabled and is enabled by EnableTrajectoryTab once the
    # matching Step-2 acoustic simulation finishes (see _BabelBaseTx.UpdateAcResults).
    #
    # Controller attributes that carry per-trajectory state between calls; these
    # are swapped in/out of self on every tab change (see _Save/_LoadThermalPanelState).
    _PANEL_ATTRS = ('_ThermalResults', '_NiftiThermalNames', '_xf', '_yf', '_zf',
                    '_XX', '_ZZ', '_LastTMap', '_bRecalculated', '_prevDisplay',
                    '_prevView', '_layout', '_figIntThermalFields', 'static_canvas',
                    '_static_ax1', '_static_ax2', '_IntensityIm', '_ThermalIm',
                    '_contour1', '_contour2', '_airmask1', '_airmask2',
                    '_ListMarkers', '_TempPlot')

    def __init__(self,parent=None,MainApp=None):
        super(Babel_Thermal, self).__init__(parent)
        self._MainApp=MainApp
        self._bMultiPoint = False
        self.bDisableUpdate=False
        self.static_canvas=None
        self.load_ui()
        self.DefaultConfig()

    def load_ui(self):
        self._setupTrajectoryTabs()

    def _NewThermalPanel(self):
        '''Fresh per-trajectory state (empty results, no computed figure yet).'''
        return {'_ThermalResults': [], '_LastTMap': -1, 'static_canvas': None}

    def _setupTrajectoryTabs(self):
        IDs = list(self._MainApp.Config['ID'])
        self._txTabs = QTabWidget(self)
        self._txTabs.tabBar().setElideMode(Qt.ElideNone)
        self._txTabs.tabBar().setExpanding(False)
        self._txTabs.setUsesScrollButtons(True)
        # Single trajectory (the common case): hide the tab bar / pane frame so
        # Step 3 looks like the original single-panel view.  Several trajectories:
        # keep the bar and only a top line under the tab row.
        if len(IDs) == 1:
            self._txTabs.tabBar().setVisible(False)
            self._txTabs.setStyleSheet("QTabWidget::pane { border: 0px; }")
        else:
            self._txTabs.setStyleSheet(
                "QTabWidget::pane { border: 0px; border-top: 1px solid palette(mid); }")

        _l = QVBoxLayout(self)
        _l.setContentsMargins(0, 0, 0, 0)
        _l.addWidget(self._txTabs)

        self._Widgets = []
        self._thPanels = []
        for i, tid in enumerate(IDs):
            form = self._CreateForm()
            self._txTabs.addTab(form, str(tid))
            self._Widgets.append(form)
            self._thPanels.append(self._NewThermalPanel())
            # Point at this form while wiring so the connections bind to *its*
            # widgets (button -> self.RunSimulation, etc.).
            self.Widget = form
            self._TrajectoryNumber = i
            self._WirePanel()
            # Each Step-3 tab stays HIDDEN until its Step-2 sim completes, so a
            # trajectory's thermal tab only appears once its acoustic field exists
            # (revealed by EnableTrajectoryTab / RefreshTrajectoryTabs).  We hide
            # (setTabVisible) rather than disable (setTabEnabled): the main-window
            # QSS collapses *disabled* tabs to zero size and QTabBar does not
            # relayout on re-enable, which left a just-enabled tab zero-width and
            # gave the tab bar a stale height.
            self._txTabs.setTabVisible(i, False)

        # Activate the first tab; only now listen for user tab switches so the
        # addTab loop above doesn't fire the handler prematurely.
        self.Widget = self._Widgets[0]
        self._TrajectoryNumber = 0
        self._LoadThermalPanelState(0)
        self._txTabs.setCurrentIndex(0)
        self._txTabs.currentChanged.connect(self._OnTrajectoryTabChanged)

    def _CreateForm(self,bMergedResults=False):
        from Babel_Thermal.ThermalForm import ThermalForm
        return ThermalForm(self,bMergedResults=bMergedResults)

    def _WirePanel(self,bMergedResults=False):
        w = self.Widget
        if not bMergedResults:
            w.SelectProfile.clicked.connect(self.SelectProfile)
            w.SelectProfile.setStyleSheet("color: #2db52d")   # bright green, readable on light & dark
            w.CalculateThermal.clicked.connect(self.RunSimulation)
            w.CalculateThermal.setStyleSheet("color: #e03030")  # bright red, readable on light & dark
        w.ExportSummary.clicked.connect(self.ExportSummary)
        w.ExportMaps.clicked.connect(self.ExportMaps)
        if hasattr(w,'CombineTrajectories'):
            w.CombineTrajectories.clicked.connect(self.CombineTrajectories)

        w.SelCombinationDropDown.currentIndexChanged.connect(self.UpdateSelCombination)
        w.IsppaSpinBox.valueChanged.connect(self._showMatplotlibVisualization)
        w.IsppaWaterSpinBox.valueChanged.connect(self.UpdateIsppaWater)
        w.IsppaScrollBar.valueChanged.connect(self._showMatplotlibVisualization)
        w.HideMarkscheckBox.stateChanged.connect(self.HideMarkChange)
        w.IsppaScrollBar.setEnabled(False)
        w.SelCombinationDropDown.setEnabled(False)
        w.IsppaSpinBox.setEnabled(False)
        w.IsppaWaterSpinBox.setEnabled(False)

        w.LocMTB.clicked.connect(self.LocateMTB)
        w.LocMTB.setEnabled(False)
        w.LocMTC.clicked.connect(self.LocateMTC)
        w.LocMTC.setEnabled(False)
        w.LocMTS.clicked.connect(self.LocateMTS)
        w.LocMTS.setEnabled(False)

        w.DisplayDropDown.currentIndexChanged.connect(self.UpdateDisplay)
        w.DisplayDropDown.setEnabled(False)

        w.SelViewDropDown.currentIndexChanged.connect(self.UpdateView)
        w.SelViewDropDown.setEnabled(False)

        self._SetupResultsTable(w)

    def _SetupResultsTable(self, w):
        Ids=['Isppa at target (W/cm2):',
             'Req. Isppa water (W/cm2):',
             'Ispta (W/cm2):',
             'Ispta at target (W/cm2):',
             'Adjustment in RAS T1W space:',
             'Max. temp. target ('+"\u2103"+') - CEM43:',
             'Max. temp. brain ('+"\u2103"+') - CEM43:',
             'Max. temp. skin ('+"\u2103"+') - CEM43:',
             'Max. temp. skull ('+"\u2103"+') - CEM43:',
             'Mechanical index:',
             'Distance from MTB to MTT (mm):']
        bg_color = w.tableWidget.parent().palette().color(w.backgroundRole())
        text_color = w.tableWidget.parent().palette().color(w.foregroundRole())
        table_palette = w.tableWidget.palette()
        table_palette.setColor(QPalette.Base, bg_color)
        table_palette.setColor(QPalette.Text, text_color)
        w.tableWidget.setPalette(table_palette)
        # NOTE: a former Windows-only setSizePolicy(Fixed, Fixed) was removed
        # here. The programmatic ThermalForm adds the table with stretch=1 and an
        # Expanding policy so it fills the left column; pinning it Fixed capped
        # its height on Windows, pushing the rows down and forcing a vertical
        # scrollbar. Leaving it Expanding makes Windows match macOS.
        for n,v in enumerate(Ids):
            item=QTableWidgetItem(v)
            item.setTextAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
            if 'Windows' in platform.system():
                font = item.font()
                font.setPointSize(8)
                item.setFont(font)
            else:
                font = item.font()
                font.setPointSize(12)
                item.setFont(font)

            w.tableWidget.setItem(n,0,item)
        if 'Windows' in platform.system():
            w.tableWidget.setColumnWidth(0,180)
            w.tableWidget.setColumnWidth(1,w.tableWidget.width()-180)
        else:
            w.tableWidget.setColumnWidth(0,220)
            w.tableWidget.setColumnWidth(1,w.tableWidget.width()-220)
        if 'Windows' in platform.system():
            w.tableWidget.verticalHeader().setDefaultSectionSize(5)
        else:
            w.tableWidget.verticalHeader().setDefaultSectionSize(25)
        w.tableWidget.setFrameShape(QFrame.NoFrame)

    
    def _SaveThermalPanelState(self, idx):
        '''Copy the active self._* per-trajectory state into panel *idx*.'''
        if not hasattr(self, '_thPanels') or not (0 <= idx < len(self._thPanels)):
            return
        panel = self._thPanels[idx]
        for a in self._PANEL_ATTRS:
            panel[a] = getattr(self, a, _MISSING)

    def _LoadThermalPanelState(self, idx):
        '''Restore panel *idx*'s per-trajectory state into the self._* aliases.'''
        panel = self._thPanels[idx]
        for a in self._PANEL_ATTRS:
            val = panel.get(a, _MISSING)
            if val is _MISSING:
                if hasattr(self, a):
                    delattr(self, a)
            else:
                setattr(self, a, val)

    @Slot(int)
    def _OnTrajectoryTabChanged(self, idx):
        if not hasattr(self, '_Widgets') or idx < 0 or idx >= len(self._Widgets):
            return
        self._SaveThermalPanelState(self._TrajectoryNumber)
        self._TrajectoryNumber = idx
        self.Widget = self._Widgets[idx]
        self._LoadThermalPanelState(idx)
        # Redraw the newly shown tab from its own state. Swapping the panel
        # aliases alone leaves whatever was last drawn (e.g. the Merged tab's
        # plot) on screen — mirroring Step 2's _ActivatePanel, which re-renders
        # on tab change. Force the rebuild path so the tab's own plot host is
        # repopulated; only for tabs that already carry computed results, so an
        # as-yet-uncomputed tab stays blank until CalculateThermal runs.
        if getattr(self, '_figIntThermalFields', None) is not None \
                and len(getattr(self, '_ThermalResults', [])) > 0 \
                and not self.bDisableUpdate:
            self._bRecalculated = True
            self._showMatplotlibVisualization()

    # Read Step-2 acoustic results by trajectory index 
    # Step 3 must NEVER change Step 2's active trajectory: doing so clobbered
    # which trajectory the next Step-2 run computed and which Step-3 tab got
    # enabled (a Step-3 visit in between broke the next Step-2 run).  Each Step-3
    # tab instead reads the acoustic result for *its own* trajectory straight from
    # AcSim's per-trajectory panels/forms, keyed by self._TrajectoryNumber, and
    # leaves AcSim's active tab untouched.

    def _AcSimPanel(self, t=None):
        if t is None:
            t = self._TrajectoryNumber
        return self._MainApp.AcSim._acPanels[t]

    def _AcSimFullSolName(self, t=None):
        '''The acoustic DataForSim result for trajectory *t* (list for phased arrays).'''
        return self._AcSimPanel(t)['FullSolName']
    
    def _AcSimFullMergedSolName(self):
        return self._MainApp.AcSim._MergedResultsFullSolName

    def _IsMergedTab(self, idx=None):
        '''True when *idx* (default: the active tab) is the combined "Merged" tab.'''
        if idx is None:
            idx = self._TrajectoryNumber
        return getattr(self, '_mergedTabIndex', None) == idx

    def _AcSimMergedSkullMap(self):
        '''
        The combined skull/material map for the mask overlay on the Merged tab,
        read from Step 2's merged acoustic panel (built when Step 2 combined the
        trajectories, which is a precondition for combining thermal results).
        '''
        AcSim = self._MainApp.AcSim
        idx = getattr(AcSim, '_mergedTabIndex', None)
        if idx is not None and 0 <= idx < len(AcSim._acPanels):
            panel = AcSim._acPanels[idx]
            if panel is not None and 'Skull' in panel:
                return panel['Skull']['MaterialMap']
        return None

    def _AcSimSkull(self, t=None):
        '''The Step-2 skull/material data for trajectory *t* (mask overlay).'''
        return self._AcSimPanel(t)['Skull']

    def _AcSimForm(self, t=None):
        '''The Step-2 device form for trajectory *t* (e.g. RefocusingcheckBox).'''
        if t is None:
            t = self._TrajectoryNumber
        return self._MainApp.AcSim._Widgets[t]

    def _CaptureFromAcSim(self, fn, t=None):
        '''
        Call an AcSim device method (GetExport / GetExtraDataForThermal) as if
        trajectory *t* were active, then restore AcSim exactly as it was.  The
        swap is synchronous (no tab move, no event-loop turn), so AcSim's active
        trajectory is unchanged on return and Step 2 is never disturbed.  Only the
        per-trajectory aliases those methods read are swapped (form, SDR, cone
        distance); the values come from trajectory *t*'s stored acoustic panel.
        '''
        if t is None:
            t = self._TrajectoryNumber
        AcSim = self._MainApp.AcSim
        panel = self._AcSimPanel(t)
        aliases = {'_SDR': 'SDR', '_LastDistanceConeToFocus': 'LastDistanceConeToFocus'}
        _NA = object()
        saved = {a: getattr(AcSim, a, _NA)
                 for a in ('_TrajectoryNumber', 'Widget', *aliases)}
        try:
            AcSim._TrajectoryNumber = t
            AcSim.Widget = AcSim._Widgets[t]
            for attr, key in aliases.items():
                if panel is not None and key in panel:
                    setattr(AcSim, attr, panel[key])
            return fn()
        finally:
            for a, v in saved.items():
                if v is _NA:
                    if hasattr(AcSim, a):
                        delattr(AcSim, a)
                else:
                    setattr(AcSim, a, v)

    def _AcFieldDone(self, i):
        '''True once trajectory *i*'s Step-2 acoustic field has been computed.'''
        AcSim = self._MainApp.AcSim
        if not hasattr(AcSim, '_acPanels') or not (0 <= i < len(AcSim._acPanels)):
            return False
        panel = AcSim._acPanels[i]
        return panel is not None and panel.get('figure') is not None

    def RefreshTrajectoryTabs(self):
        '''
        Reveal every Step-3 tab whose Step-2 acoustic field exists.  Driven from
        ground truth (AcSim's per-trajectory result panels) rather than from a
        single completion notification, so the set of visible thermal tabs is
        always correct no matter the order Step-2 trajectories were run in or when
        the user switches to the thermal step.
        '''
        if not hasattr(self, '_txTabs'):
            return
        for i in range(self._txTabs.count()):
            # The Merged tab is not a trajectory: its visibility is driven by
            # combining thermal results (_AddMergedResultsTab), not by an acoustic
            # field, so leave it out of the per-trajectory reveal.
            if self._IsMergedTab(i):
                continue
            if self._AcFieldDone(i) and not self._txTabs.isTabVisible(i):
                self._txTabs.setTabVisible(i, True)

    def showEvent(self, event):
        # Whenever Step 3 becomes visible, make sure every trajectory whose
        # acoustic field is ready has its tab shown (self-corrects regardless
        # of what happened while the user was on other steps).
        super().showEvent(event)
        self.RefreshTrajectoryTabs()

    def EnableTrajectoryTab(self, idx):
        '''
        Reveal the Step-3 tab for trajectory *idx*.  Called from Step 2 once the
        matching acoustic simulation completes, so each trajectory's thermal tab
        only appears after its acoustic field exists.  Step 3 always lands on the
        first tab, so the user starts there when opening the step.
        '''
        if not hasattr(self, '_txTabs') or not (0 <= idx < self._txTabs.count()):
            return
        self._txTabs.setTabVisible(idx, True)
        # Also pick up any other trajectories already computed (belt-and-braces
        # with showEvent), then land on the first tab.
        self.RefreshTrajectoryTabs()
        self._txTabs.setCurrentIndex(0)

    def DefaultConfig(self):
        #Specific parameters for the thermal simulation - to be configured  via a yaml
        config = ThermalProfileConfig(self._MainApp.Config['ThermalProfile'])
        print("Thermal configuration:")
        print(config)
        self.Config=config
        self.bDisableUpdate=True
        for form in self._Widgets:
            self._PopulateCombination(form)
        self.bDisableUpdate=False

    def _PopulateCombination(self, form):
        dd = form.SelCombinationDropDown
        while dd.count()>0:
            dd.removeItem(0)

        if self.Config['bConcatenateSimulations']:
            dd.addItem('CONCATENATED PROTOCOL')
        else:
            for c in self.Config['AllDC_PRF_Duration']:
                if c['Duration']<1.0:
                    sOn = '%3.2fs-On' % (c['Duration'])
                else:
                    sOn = '%3.1fs-On' % (c['Duration'])
                if c['DurationOff']<1.0:
                    sOff = '%3.2fs-Off' % (c['DurationOff'])
                else:
                    sOff = '%3.1fs-Off' % (c['DurationOff'])
                stritem = sOn + ' ' + sOff + ' %3.1f%% %3.1fHz' %(c['DC']*100,c['PRF'])
                if c['Repetitions'] >1:
                    stritem += ' %iReps' %(c['Repetitions'])
                dd.addItem(stritem)

    def EnableMultiPoint(self):
        self._bMultiPoint=True

    @Slot()
    def SelectProfile(self):
        curdir = os.path.dirname(self._MainApp.Config['ThermalProfile'])
        fThermalProfile=QFileDialog.getOpenFileName(self,"Select thermal profile",curdir,"yaml (*.yaml)")[0]
        if len(fThermalProfile)>0:
            if self._MainApp.UpdateThermalProfile(fThermalProfile):
                self.Widget.SelectProfile.setProperty('UserData',fThermalProfile)  
                self.DefaultConfig()  
                self.RunSimulation()


    @Slot()
    def CombineTrajectories(self):
        print('this',self._TrajectoryNumber)
        MergedPressureRatio=[]
        self._SaveThermalPanelState(self._TrajectoryNumber)
        for n in range(len(self._MainApp.Config['ID'])):
            SelIsppa=self._Widgets[n].IsppaSpinBox.value()
            print(self._thPanels[n]['_ThermalResults'][0]['PressureRatio'])
            MergedPressureRatio.append(self._thPanels[n]['_ThermalResults'][0]['PressureRatio']*np.sqrt(SelIsppa/self.Config['BaseIsppa']))
        self.RunSimulation(True,MergedPressureRatio)
            
    @Slot()
    def RunSimulation(self,bMergedSimulation=False,MergedPressureRatio=[]):
        bCalcFields=False

        # Acoustic result for THIS thermal tab's trajectory (read by index; the
        # thermal worker uses the same index, so Step 2 is never re-pointed).
        if not bMergedSimulation:
            BaseField=self._AcSimFullSolName()
        else:
            BaseField=self._AcSimFullMergedSolName()

        self._bRunningMerged=bMergedSimulation
        
        if type(BaseField) is list:
            BaseField=BaseField[0]

        PrevFiles=[]
        for combination in self.Config['AllDC_PRF_Duration']:
            ThermalName=GetThermalOutName(BaseField,combination['Duration'],
                                                    combination['DurationOff'],
                                                    combination['DC'],
                                                    self.Config['BaseIsppa'],
                                                    combination['PRF'],
                                                    combination['Repetitions'])+'.h5'

            if os.path.isfile(ThermalName):
                PrevFiles.append(ThermalName)
        if len(PrevFiles)==len(self.Config['AllDC_PRF_Duration']):
            ret = QMessageBox.question(self,'', "Thermal sim files already exist\n" +
                                "Do you want to recalculate?\nSelect No to reload",
            QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
        else:
            bCalcFields = True
        
        self._bRecalculated=True
        self._ThermalResults=[]
        if bCalcFields:
            # Capture everything the worker needs from Step 2 for THIS trajectory
            # up front (case file, refocus selection, device extra-data), so the
            # off-thread worker never reads AcSim's live/active state.
            t = self._TrajectoryNumber

            if not bMergedSimulation:
                case = self._AcSimFullSolName(t)
            else:
                case=self._AcSimFullMergedSolName()


            bRefocus = (hasattr(self._AcSimForm(t), 'RefocusingcheckBox') and
                        self._AcSimForm(t).RefocusingcheckBox.isChecked())
            ExtraData = self._CaptureFromAcSim(
                self._MainApp.AcSim.GetExtraDataForThermal, t)
            self.thread = QThread()
            self.worker = RunThermalSim(self._MainApp, case, bRefocus, ExtraData,bMergedSimulation,MergedPressureRatio)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.UpdateThermalResults)
            self.worker.finished.connect(self._MainApp.SendTelemetry)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self._MainApp.SendTelemetry)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)

            self.worker.logTelemetry.connect(self._MainApp.LogTelemetry)
 
            self.thread.start()
            self._MainApp.Widget.tabWidget.setEnabled(False)
            self._MainApp.showClockDialog()
        else:
            self.UpdateThermalResults()

    def NotifyError(self):
        self._MainApp.hideClockDialog()
        if 'BABEL_PYTEST' not in os.environ:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("There was an error in execution -\nconsult log window for details")
            msgBox.exec()
        else:
            #this will unblock for PyTest
            self._MainApp.testing_error = True
            self._MainApp.Widget.tabWidget.setEnabled(True)

    @Slot()
    def HideMarkChange(self,val):
        self._showMatplotlibVisualization()

    @Slot()
    def UpdateSelCombination(self):
        self._showMatplotlibVisualization()
        
    @Slot()
    def UpdateDisplay(self,val):
        self._showMatplotlibVisualization()

    @Slot()
    def UpdateView(self,val):
        self._showMatplotlibVisualization()

    def _GetViewParams(self):
        '''
        Map the "Select view" choice to the fixed (scroll) axis, the two displayed
        axes and their coordinate vectors.  XZ (default) fixes Y and shows X-Z; YZ
        fixes X and shows Y-Z; XY fixes Z and shows X-Y.  hvec/vvec are the
        horizontal/vertical plot coordinates (x/y raw, z relative to the skin).
        '''
        view = self.Widget.SelViewDropDown.currentText()
        if view == 'YZ':
            return dict(view=view, axis=0, haxis=1, vaxis=2,
                        hvec=self._yf, vvec=self._zf,
                        hlabel='Y (mm)', vlabel='Distance from skin (mm)',
                        poslabel='X pos')
        if view == 'XY':
            return dict(view=view, axis=2, haxis=0, vaxis=1,
                        hvec=self._xf, vvec=self._yf,
                        hlabel='X (mm)', vlabel='Y (mm)',
                        poslabel='Z pos')
        return dict(view='XZ', axis=1, haxis=0, vaxis=2,
                    hvec=self._xf, vvec=self._zf,
                    hlabel='X (mm)', vlabel='Distance from skin (mm)',
                    poslabel='Y pos')

    def _SlicePlane(self, Field, Sel, axis):
        '''Take the 2D slice at index *Sel* along the fixed *axis* (shape [h,v]).'''
        if axis == 0:
            return Field[Sel, :, :]
        if axis == 2:
            return Field[:, :, Sel]
        return Field[:, Sel, :]

    @Slot()
    def UpdateIsppaWater(self,val):
        self._showMatplotlibVisualization(bIsppaBrainToWater=False)

    def _showMatplotlibVisualization(self,bUpdatePlot=True,OverWriteIsppa=None,bIsppaBrainToWater=True):
        if self.bDisableUpdate:
            return
        self._MainApp.Widget.tabWidget.setEnabled(True)
        self.Widget.ExportSummary.setEnabled(True)
        self.Widget.ExportMaps.setEnabled(True)
        self.Widget.SelCombinationDropDown.setEnabled(True)
        self.Widget.IsppaSpinBox.setEnabled(True)
        self.Widget.IsppaWaterSpinBox.setEnabled(True)
        self.Widget.DisplayDropDown.setEnabled(True)
        WhatDisplay = self.Widget.DisplayDropDown.currentIndex()
        if WhatDisplay==0:
            self.Widget.LocMTS.setEnabled(True)
            self.Widget.LocMTC.setEnabled(True)
            self.Widget.LocMTB.setEnabled(True)
            self.Widget.IsppaScrollBar.setEnabled(True)
            self.Widget.SelViewDropDown.setEnabled(True)
            if self.Widget.HideMarkscheckBox.isEnabled()== False:
                self.Widget.HideMarkscheckBox.setEnabled(True)
        else:
            self.Widget.LocMTS.setEnabled(False)
            self.Widget.LocMTC.setEnabled(False)
            self.Widget.LocMTB.setEnabled(False)
            self.Widget.IsppaScrollBar.setEnabled(False)
            self.Widget.SelViewDropDown.setEnabled(False)
            if self.Widget.HideMarkscheckBox.isEnabled()== True:
                self.Widget.HideMarkscheckBox.setEnabled(False)
        
        if not self._IsMergedTab():
            BaseField=self._AcSimFullSolName()
        else:
            BaseField=self._AcSimFullMergedSolName()

        if type(BaseField) is list:
            BaseField=BaseField[0]

        if len(self._ThermalResults)==0:
            self._MainApp.hideClockDialog()
            self._NiftiThermalNames=[]
            self._LastTMap=-1
            for combination in self.Config['AllDC_PRF_Duration']:
                ThermalName=GetThermalOutName(BaseField,combination['Duration'],
                                                        combination['DurationOff'],
                                                        combination['DC'],
                                                        self.Config['BaseIsppa'],
                                                        combination['PRF'],
                                                        combination['Repetitions'])+'.h5'
                self._NiftiThermalNames.append(os.path.splitext(ThermalName)[0])
                self._ThermalResults.append(ReadFromH5py(ThermalName))
            if self.Config['bConcatenateSimulations']:
                DataThermal=self._ThermalResults[-1] #we pick the latest one as that has the concatenated results
            else:
                DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
            self._xf=DataThermal['x_vec']
            self._yf=DataThermal['y_vec']
            self._zf=DataThermal['z_vec']
            SkinZ=np.array(np.where(DataThermal['MaterialMap']==1)).T.min(axis=0)[1]
            self._zf-=self._zf[SkinZ]
        if self.Config['bConcatenateSimulations']:
            DataThermal=self._ThermalResults[-1] #we pick the latest one as that has the concatenated results
        else:
            DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        if 'BaselineTemperature' in DataThermal:
            BaselineTemperature=DataThermal['BaselineTemperature']
        else:
            BaselineTemperature=37.0
        
        Loc=DataThermal['TargetLocation']

        # Selected orthogonal view: which axis the slice scrollbar scrolls, and the
        # two displayed axes + coordinate vectors.
        vp = self._GetViewParams()
        axis = vp['axis']
        if not hasattr(self, '_prevView'):
            self._prevView = vp['view']
        bViewChanged = (self._prevView != vp['view'])

        # (Re)scale the slice scrollbar to the fixed axis on first display or when
        # the view changes; centre it on the target's index along that axis.
        if self._LastTMap==-1 or bViewChanged:
            self.bDisableUpdate=True
            self.Widget.IsppaScrollBar.setMaximum(DataThermal['MaterialMap'].shape[axis]-1)
            self.Widget.IsppaScrollBar.setValue(int(Loc[axis]))
            self.bDisableUpdate=False
            self.Widget.IsppaScrollBar.setEnabled(True)

        if self.Config['bConcatenateSimulations']:
            self._LastTMap=len(self._ThermalResults)-1
        else:
            self._LastTMap=self.Widget.SelCombinationDropDown.currentIndex()
            
        if OverWriteIsppa is None:
            if bIsppaBrainToWater:
                SelIsppa=self.Widget.IsppaSpinBox.value()
            else:
                SelIsppaWater=self.Widget.IsppaWaterSpinBox.value()
                if self._bMultiPoint:
                    RatioLossess=np.mean(DataThermal['RatioLosses'])
                else:
                    RatioLossess=DataThermal['RatioLosses']
                SelIsppa=SelIsppaWater*RatioLossess
        else:
            SelIsppa=OverWriteIsppa

        xf=self._xf
        zf=self._zf

        SelY=self.Widget.IsppaScrollBar.value()

        IsppaRatio=SelIsppa/self.Config['BaseIsppa']
        
        if self._bMultiPoint:
            AdjustedIsspa = SelIsppa/DataThermal['RatioLosses']
            AdjustedIsspaStDev = np.std(AdjustedIsspa)
            AdjustedIsspa=np.mean(AdjustedIsspa)
        else:
            AdjustedIsspa = SelIsppa/DataThermal['RatioLosses']
        
        if self.Config['bConcatenateSimulations']:
            DutyCycle=[]
            for entry in self.Config['AllDC_PRF_Duration']:
                DutyCycle.append(entry['DC'])
        else:
            DutyCycle=self.Config['AllDC_PRF_Duration'][self.Widget.SelCombinationDropDown.currentIndex()]['DC']

        def NewItem(str,data,color="blue",visible=True):
            item=QTableWidgetItem(str)
            item.setData(QtCore.Qt.UserRole,data)
            is_dark = self.Widget.tableWidget.palette().color(QPalette.Base).lightness() < 128
            if color == "blue":
                resolved_color = QColor("#5ba3ff" if is_dark else "#0050cc")
            elif color == "red":
                resolved_color = QColor("#e03030")
            else:
                resolved_color = QColor(color)
            item.setForeground(resolved_color)
            # Set the font style to bold
            font = item.font()
            font.setBold(True)
            if 'Windows' in platform.system():
                font.setPointSize(8)
            else:
                font.setPointSize(12)
            item.setFont(font)
            return item
        
        DensityMap=DataThermal['MaterialList']['Density'][DataThermal['MaterialMap']]
        SoSMap=    DataThermal['MaterialList']['SoS'][DataThermal['MaterialMap']]

        ImpedanceTarget = DensityMap[Loc[0],Loc[1],Loc[2]]*SoSMap[Loc[0],Loc[1],Loc[2]]
        if self._MainApp.Config['bForceHomogenousMedium']:
            BrainID=[1]
        elif self._MainApp.Config['bUseCT']:
            if self._MainApp._bSegmentedBrain:
                BrainID=[2,3,4,5]
            else:
                BrainID=[2]
        else:
            if self._MainApp._bSegmentedBrain:
                BrainID=[4,5,6,7]
            else:
                BrainID=[4]

        SelBrain=np.isin(DataThermal['MaterialMap'],BrainID)

        if self._IsMergedTab():
            AcSimMask=self._AcSimMergedSkullMap()
        else:
            AcSimMask=self._AcSimSkull()['MaterialMap']

        IsppaTarget = DataThermal['p_map'][Loc[0],Loc[1],Loc[2]]**2/2/ImpedanceTarget/1e4*IsppaRatio
        
        LocMax=np.array(np.where(DataThermal['p_map']==DataThermal['p_map'][SelBrain].max())).flatten()
        ImpedanceLocMax= DensityMap[LocMax[0],LocMax[1],LocMax[2]]*SoSMap[LocMax[0],LocMax[1],LocMax[2]]
        
        self.Widget.tableWidget.setItem(0,1,NewItem('%4.2f' % IsppaTarget,IsppaTarget))

        if self._bMultiPoint:
            self.Widget.tableWidget.setItem(1,1,NewItem('%4.2f (%4.2f)' % (AdjustedIsspa,AdjustedIsspaStDev),AdjustedIsspa))
        else:
            self.Widget.tableWidget.setItem(1,1,NewItem('%4.2f' % AdjustedIsspa,AdjustedIsspa))
        self.bDisableUpdate=True
        if bIsppaBrainToWater:
            self.Widget.IsppaWaterSpinBox.setValue(np.round(AdjustedIsspa,2))
        else:
            self.Widget.IsppaSpinBox.setValue(np.round(SelIsppa,2))
        self.bDisableUpdate=False

        if self.Config['bConcatenateSimulations']:
            st=','.join(format(x*SelIsppa, "2.1f") for x in DutyCycle)
            self.Widget.tableWidget.setItem(2,1,NewItem(st,[x * SelIsppa for x in DutyCycle]))
            st=','.join(format(x*IsppaTarget, "2.1f") for x in DutyCycle)
            self.Widget.tableWidget.setItem(3,1,NewItem(st,[x * IsppaTarget for x in DutyCycle]))
        else:
            self.Widget.tableWidget.setItem(2,1,NewItem('%4.2f' % (SelIsppa*DutyCycle),SelIsppa*DutyCycle))
            self.Widget.tableWidget.setItem(3,1,NewItem('%4.2f' % (IsppaTarget*DutyCycle),IsppaTarget*DutyCycle))

        self.Widget.tableWidget.setItem(4,1,NewItem(np.array2string(DataThermal['AdjustmentInRAS'],
                                               formatter={'float_kind':lambda x: "%3.2f" % x}),DataThermal['AdjustmentInRAS']))

        AdjustedTemp=((DataThermal['TemperaturePoints']-BaselineTemperature)*IsppaRatio+BaselineTemperature)
        DoseUpdate=np.trapz(RCoeff(AdjustedTemp)**(43.0-AdjustedTemp),dx=DataThermal['dt'],axis=1)/60
   
        MTT=(DataThermal['TempEndFUS'][Loc[0],Loc[1],Loc[2]]-BaselineTemperature)*IsppaRatio+BaselineTemperature
        if self._MainApp.Config['bForceHomogenousMedium']:
            MTTCEM=DoseUpdate[0]
        else:
            MTTCEM=DoseUpdate[3] if len(DoseUpdate)==4 else DoseUpdate[1]
        self.Widget.tableWidget.setItem(5,1,NewItem('%3.1f - %4.1G' % (MTT,MTTCEM),[MTT,MTTCEM],"red" if (MTT >= 39 or MTTCEM >=2.0) else "blue"))

        MTB=DataThermal['TI']*IsppaRatio+BaselineTemperature
        if self._MainApp.Config['bForceHomogenousMedium']:
            MTBCEM=DoseUpdate[0]
        else:
            MTBCEM=DoseUpdate[1]
        self.Widget.tableWidget.setItem(6,1,NewItem('%3.1f - %4.1G' % (MTB,MTBCEM),[MTB,MTBCEM],"red" if (MTB >= 39 or MTBCEM >=2.0) else "blue"))
        
        MTS=DataThermal['TIS']*IsppaRatio+BaselineTemperature
        MTSCEM=DoseUpdate[0]
        self.Widget.tableWidget.setItem(7,1,NewItem('%3.1f - %4.1G' % (MTS,MTSCEM),[MTS,MTSCEM],"red" if MTS >= 39  else "blue"))

        MTC=DataThermal['TIC']*IsppaRatio+BaselineTemperature
        if self._MainApp.Config['bForceHomogenousMedium']:
            MTCCEM=DoseUpdate[0]
        else:
            MTCCEM=DoseUpdate[2]
        self.Widget.tableWidget.setItem(8,1,NewItem('%3.1f - %4.1G' % (MTC,MTCCEM),[MTC,MTCCEM],"red" if MTC >= 39  else "blue"))

        MI=np.sqrt(SelIsppa*1e4*ImpedanceLocMax*2)/1e6/np.sqrt(self._MainApp._Frequency/1e6)
        self.Widget.tableWidget.setItem(9,1,NewItem('%3.1f ' % (MI),MI,"red" if MI > 1.9 else "blue"))

        Distance_MTB_MTT = np.linalg.norm(DataThermal['mBrain']-Loc)*(xf[1]-xf[0])
        self.Widget.tableWidget.setItem(10,1,NewItem('%3.1f ' % (Distance_MTB_MTT),Distance_MTB_MTT))

        if self._bRecalculated or bViewChanged:
            self._XX,self._ZZ=np.meshgrid(vp['hvec'],vp['vvec'])

        if bUpdatePlot:
            Intensity=DataThermal['p_map']**2/2/DensityMap/SoSMap/1e4*IsppaRatio
            Temperature=(DataThermal['TempEndFUS']-BaselineTemperature)*IsppaRatio+BaselineTemperature

            if 'ZIntoSkinPixels' in DataThermal:
                Intensity[:,:,DataThermal['ZIntoSkinPixels']]=0
            else:
                Intensity[:,:,0]=0
           
            if not hasattr(self,'_prevDisplay'):
                self._prevDisplay = -1 #we set the initial plotting
            IntensityMap=self._SlicePlane(Intensity,SelY,axis).T.copy()
            Tmap=self._SlicePlane(Temperature,SelY,axis)


            crlims=[0,1,2]

            if (self._bRecalculated or self._prevDisplay != WhatDisplay or bViewChanged or self.Config['bConcatenateSimulations']) and hasattr(self,'_figIntThermalFields'):
                children = []
                for i in range(self._layout.count()):
                    child = self._layout.itemAt(i).widget()
                    if child:
                        children.append(child)
                for child in children:
                    child.deleteLater()
                delattr(self,'_figIntThermalFields')
                # self._layout.deleteLater()
                

            if hasattr(self,'_figIntThermalFields'):
                if WhatDisplay==0:
                    self._IntensityIm.set_data(IntensityMap)
                    self._IntensityIm.set(clim=[IntensityMap.min(),IntensityMap.max()])
                    self._ThermalIm.set_data(Tmap.T)
                    self._ThermalIm.set(clim=[BaselineTemperature,Tmap.max()])
                    if not self._MainApp.Config['bForceHomogenousMedium']:
                        if hasattr(self,'_contour1'):
                            ListObjects=[self._contour1,self._contour2]
                            if 'AirMask' in DataThermal:
                                ListObjects+=[self._airmask1,self._airmask2]
                            for c in ListObjects:
                                try: #this is for old Matplotlib
                                    for coll in c.collections:
                                        coll.remove()
                                except:
                                    c.remove()
                            del self._contour1
                            del self._contour2
                            if 'AirMask' in DataThermal:
                                del self._airmask1
                                del self._airmask2
                            
                            self._contour1=self._static_ax1.contour(self._XX,self._ZZ,self._SlicePlane(AcSimMask,SelY,axis).T,crlims, colors ='y',linestyles = ':')
                            self._contour2=self._static_ax2.contour(self._XX,self._ZZ,self._SlicePlane(AcSimMask,SelY,axis).T,crlims, colors ='y',linestyles = ':')

                            if 'AirMask' in DataThermal:
                                AirMap=self._SlicePlane(DataThermal['AirMask'],SelY,axis).T
                                AirMap=np.ma.masked_where(AirMap==0 , AirMap)
                                self._airmask1 = self._static_ax1.contourf(self._XX,self._ZZ,AirMap,[0,1],cmap=plt.cm.gray_r)
                                self._airmask2 = self._static_ax2.contourf(self._XX,self._ZZ,AirMap,[0,1],cmap=plt.cm.gray_r)

                    while len(self._ListMarkers)>0:
                        obj= self._ListMarkers.pop()
                        obj.remove()
                else:
                    curylim=list(self._static_ax1.get_ylim())
                    MaxPrevTemp=0.0
                    MinPrevTemp=1e6
                    for e in self._TempPlot:
                        ydata=e.get_ydata()
                        MaxPrevTemp=np.max([MaxPrevTemp,np.max(ydata)])
                        MinPrevTemp=np.min([MinPrevTemp,np.min(ydata)])
                    perUpLim=curylim[1]/MaxPrevTemp
                    perUpDown=curylim[0]/MinPrevTemp
                    curylim[1]=np.max(AdjustedTemp)*perUpLim
                    curylim[0]=np.min(AdjustedTemp)*perUpDown
                            
                    for n in range(AdjustedTemp.shape[0]):
                        self._TempPlot[n].set_ydata(AdjustedTemp[n,:])
                    self._static_ax1.set_ylim(curylim)
                    
                self._figIntThermalFields.canvas.draw_idle()
            else:
                if not hasattr(self,'_layout'):
                    self._layout = QVBoxLayout(self.Widget.AcField_plot1)
                if WhatDisplay==0:
                    self._ListMarkers=[]
                    self._figIntThermalFields=Figure(figsize=(14, 12))
                    self.static_canvas = FigureCanvas(self._figIntThermalFields)
                    toolbar=style_nav_toolbar(NavigationToolbar2QT(self.static_canvas,self))
                    self._layout.addWidget(toolbar)
                    self._layout.addWidget(self.static_canvas)
                    static_ax1,static_ax2 = self.static_canvas.figure.subplots(1,2)
                    self._static_ax1=static_ax1
                    self._static_ax2=static_ax2

                    _extent=[vp['hvec'].min(),vp['hvec'].max(),vp['vvec'].max(),vp['vvec'].min()]
                    self._IntensityIm=static_ax1.imshow(IntensityMap,extent=_extent,
                            cmap=plt.cm.jet)
                    static_ax1.set_title('Isppa (W/cm$^2$)')
                    plt.colorbar(self._IntensityIm,ax=static_ax1)
                    if not self._MainApp.Config['bForceHomogenousMedium']:
                        self._contour1=static_ax1.contour(self._XX,self._ZZ,self._SlicePlane(AcSimMask,SelY,axis).T,crlims,colors ='y',linestyles = ':')

                    static_ax1.set_xlabel(vp['hlabel'])
                    static_ax1.set_ylabel(vp['vlabel'])

                    self._ThermalIm=static_ax2.imshow(Tmap.T,
                            extent=_extent,cmap=plt.cm.jet,vmin=BaselineTemperature)
                    static_ax2.set_title('Temperature ($^{\circ}$C)')
                    static_ax2.set_xlabel(vp['hlabel'])
                    static_ax2.set_ylabel(vp['vlabel'])

                    plt.colorbar(self._ThermalIm,ax=static_ax2)
                    if not self._MainApp.Config['bForceHomogenousMedium']:
                        self._contour2=static_ax2.contour(self._XX,self._ZZ,self._SlicePlane(AcSimMask,SelY,axis).T,crlims, colors ='y',linestyles = ':')

                    if 'AirMask' in DataThermal:
                        AirMap=self._SlicePlane(DataThermal['AirMask'],SelY,axis).T
                        AirMap=np.ma.masked_where(AirMap==0 , AirMap)
                        self._airmask1 = static_ax1.contourf(self._XX,self._ZZ,AirMap,[0,1],cmap=plt.cm.gray_r)
                        self._airmask2 = static_ax2.contourf(self._XX,self._ZZ,AirMap,[0,1],cmap=plt.cm.gray_r)

                    self._figIntThermalFields.set_facecolor(self._MainApp._BackgroundColorFigures)
                else:
                    self._figIntThermalFields=Figure(figsize=(14, 12))
                    self.static_canvas = FigureCanvas(self._figIntThermalFields)
                    toolbar=style_nav_toolbar(NavigationToolbar2QT(self.static_canvas,self))
                    self._layout.addWidget(toolbar)
                    self._layout.addWidget(self.static_canvas)
                    static_ax1 = self.static_canvas.figure.subplots(1,1)
                    timevec=np.arange(AdjustedTemp.shape[1])*DataThermal['dt']
                    self._TempPlot=static_ax1.plot(timevec,AdjustedTemp.T)
                    static_ax1.set_xlabel('time (s)')
                    static_ax1.set_ylabel('temperature (degrees C)')
                    leg=static_ax1.legend(['Skin','Brain','Skull','Target'], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                    self._static_ax1=static_ax1
                    self._figIntThermalFields.set_facecolor(self._MainApp._BackgroundColorFigures)
                    self._static_ax1.set_facecolor(self._MainApp._BackgroundColorFigures)
                    leg.get_frame().set_facecolor(self._MainApp._BackgroundColorFigures)
                    

            self._bRecalculated=False
            if WhatDisplay==0:
                hvec, vvec = vp['hvec'], vp['vvec']
                haxis, vaxis = vp['haxis'], vp['vaxis']
                # Fixed-axis position readout, relative to the target.
                posvec=DataThermal[['x_vec','y_vec','z_vec'][axis]].copy()
                posvec=posvec-posvec[int(Loc[axis])]
                if not self.Widget.HideMarkscheckBox.isChecked():
                    self._ListMarkers.append(self._static_ax1.plot(hvec[Loc[haxis]],vvec[Loc[vaxis]],'k+',markersize=18)[0])
                    self._ListMarkers.append(self._static_ax2.plot(hvec[Loc[haxis]],vvec[Loc[vaxis]],'k+',markersize=18,)[0])
                    for k,kl in zip(['mSkin','mBrain','mSkull'],['MTS','MTB','MTC']):
                        if SelY == DataThermal[k][axis]:
                            self._ListMarkers.append(self._static_ax2.plot(hvec[DataThermal[k][haxis]],
                                            vvec[DataThermal[k][vaxis]],'wx',markersize=12)[0])
                            self._ListMarkers.append(self._static_ax2.text(hvec[DataThermal[k][haxis]]-5,
                                            vvec[DataThermal[k][vaxis]]+5,kl,color='w',fontsize=10))

                self.Widget.SliceLabel.setText("%s = %3.2f mm" %(vp['poslabel'],posvec[self.Widget.IsppaScrollBar.value()]))
            self._prevDisplay=WhatDisplay
            self._prevView=vp['view']

            # Per-trajectory NIfTI/VTK overlays are keyed by trajectory index; the
            # Merged tab has no such slot (there is no merged thermal viewer), so
            # skip the 3D update for it — the 2D matplotlib view above is enough.
            if not self._IsMergedTab():
                NiftiIntensity=nibabel.Nifti1Image(np.flip(Intensity,axis=2).astype(np.float32),affine=self._MainApp._NiftiSkull[self._TrajectoryNumber].affine)
                NiftiTemperature=nibabel.Nifti1Image(np.flip(Temperature,axis=2).astype(np.float32),affine=self._MainApp._NiftiSkull[self._TrajectoryNumber].affine)
                self._MainApp.UpdateNiftiTemperatureResults(NiftiIntensity,NiftiTemperature,self._TrajectoryNumber)

    @Slot()
    def UpdateThermalResults(self):
        # A merged run renders into the dedicated "Merged" tab; a normal run
        # renders into the active trajectory tab.
        if getattr(self, '_bRunningMerged', False):
            self._bRunningMerged = False
            self._AddMergedResultsTab()
        else:
            self._showMatplotlibVisualization()
        self.UpdateCombineResultsButton()

    def _AddMergedResultsTab(self):
        '''
        Show the combined thermal results in a dedicated "Merged" tab, mirroring
        _BabelBaseTx._AddMergedResultsTab.  Unlike Step 2's read-only merged form,
        Step 3 keeps the full ThermalForm (left panel + results table); only the
        compute actions (Calculate / Update-profile / Combine) are disabled, since
        this tab just displays the already-combined thermal maps.  Re-running the
        combination re-renders the existing tab from the freshly written files.
        '''
        if getattr(self, '_mergedTabIndex', None) is None:
            form = self._CreateForm(bMergedResults=True)
            idx = self._txTabs.addTab(form, 'Merged')
            self._Widgets.append(form)
            self._thPanels.append(self._NewThermalPanel())
            self._mergedTabIndex = idx
            # Wire the merged form (bind signals/table to *its* widgets), then make
            # it read-only for the compute actions.
            prevW, prevT = self.Widget, self._TrajectoryNumber
            self.Widget = form
            self._TrajectoryNumber = idx
            self._WirePanel(bMergedResults=True)
            self._PopulateCombination(form)
            self.Widget, self._TrajectoryNumber = prevW, prevT
            if hasattr(form, 'CombineTrajectories'):
                form.CombineTrajectories.setEnabled(False)
            self._txTabs.setTabVisible(idx, True)
        idx = self._mergedTabIndex

        # Point the controller at the merged tab and (re)render the combined
        # result.  We do NOT stash the current (post-run, emptied) self state into
        # the trajectory panel — CombineTrajectories already saved that tab before
        # launching the run — so the source trajectory tab keeps its own results.
        self.Widget = self._Widgets[idx]
        self._TrajectoryNumber = idx
        self._LoadThermalPanelState(idx)
        self._bRecalculated = True
        self._ThermalResults = []
        self._showMatplotlibVisualization()
        self._txTabs.setCurrentIndex(idx)

    def UpdateCombineResultsButton(self):
        if hasattr(self.Widget,'CombineTrajectories'):
            for n in range(len(self._MainApp.Config['ID'])):
                self._Widgets[n].CombineTrajectories.setEnabled(self._MainApp.AllThermalFieldsDone() and len(self._MainApp.AcSim._Widgets)>len(self._MainApp.Config['ID']))

    @Slot()
    def LocateMTB(self):
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        self.Widget.IsppaScrollBar.setValue(DataThermal['mBrain'][1])
    @Slot()
    def LocateMTC(self):
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        self.Widget.IsppaScrollBar.setValue(DataThermal['mSkull'][1])
    @Slot()
    def LocateMTS(self):
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        self.Widget.IsppaScrollBar.setValue(DataThermal['mSkin'][1])

    @Slot()
    def ExportSummary(self):
        if 'BABEL_PYTEST' not in os.environ:
            DefaultPath=os.path.split(self._MainApp.Config['T1W'])[0]
            outCSV=QFileDialog.getSaveFileName(self,"Select export CSV file",DefaultPath,"csv (*.csv)")[0]
            if len(outCSV)==0:
                return
        else:
            DefaultPath=self._MainApp.Config['OutputFilesPath']+os.sep
            outCSV=DefaultPath+'Test_Export.csv'
        
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        DataToExport={}
        #we recover specifics of main app and acoustic simulation (the acoustic
        #export is read for THIS tab's trajectory without disturbing Step 2). The
        #Merged tab has no per-trajectory acoustic form, so skip its acoustic export.
        DataToExport = DataToExport | self._MainApp.GetExport()
        if not self._IsMergedTab():
            DataToExport = DataToExport | self._CaptureFromAcSim(self._MainApp.AcSim.GetExport)
        DataToExport['AdjustRAS']=self.Widget.tableWidget.item(4,1).data(QtCore.Qt.UserRole)
        
        pd.DataFrame.from_dict(data=DataToExport, orient='index').to_csv(outCSV, header=False)
        currentIsppa=self.Widget.IsppaSpinBox.value()
        currentCombination=self.Widget.SelCombinationDropDown.currentIndex()
        #now we create new Table to export safety metrics based on timing options and Isppa
        for n in range(self.Widget.SelCombinationDropDown.count()):
            self.Widget.SelCombinationDropDown.setCurrentIndex(n)
            with open(outCSV,'a') as f:
                f.write('*'*80+'\n')
                f.write('TimingExposure,'+self.Widget.SelCombinationDropDown.currentText()+'\n')
                f.write('*'*80+'\n')

            DataToExport={}
            DataToExport['Distance from MTB to MTT:']=self.Widget.tableWidget.item(10,1).data(QtCore.Qt.UserRole)
            pd.DataFrame.from_dict(data=DataToExport, orient='index').to_csv(outCSV,mode='a', header=False)
                
            DataToExport={}
            DataToExport['Isppa']=np.arange(0.5,self.Widget.IsppaSpinBox.maximum()+0.5,0.5)
            for v in DataToExport['Isppa']:
                self._showMatplotlibVisualization(bUpdatePlot=False,OverWriteIsppa=v)
                collection = ['Isppa target']
                source = [0,1]
                if self._bMultiPoint:
                    collection+=['Isppa water (avg)','Isppa water (std)']
                    source+=[1]
                else:
                    collection+=['Isppa water']
                collection+=['Mechanical index',
                            'Ispta',
                            'Ispta target',
                            'Max. temp. target',
                            'Max. temp. brain',
                            'Max. temp. skin',
                            'Max. temp. skull',
                            'CEM target',
                            'CEM brain',
                            'CEM skin',
                            'CEM skull']
                source+=[9,2,3,5,6,7,8,5,6,7,8]
                for k,index in zip(collection,source):
                    if k not in DataToExport:
                        DataToExport[k]=[]
                   
                    data=self.Widget.tableWidget.item(index,1).data(QtCore.Qt.UserRole)
                    if 'temp.' in k:
                        data=data[0]
                    elif 'CEM' in k:
                        data=data[1]
                    elif k in ['Isppa water (avg)','Isppa water']:
                        data=np.mean(data)
                    elif k == 'Isppa water (std)':
                        data=np.std(data)
                    DataToExport[k].append(data)
                
            pd.DataFrame.from_dict(data=DataToExport).to_csv(outCSV,mode='a',index=False)
        if currentCombination !=self.Widget.SelCombinationDropDown.currentIndex():
            self.Widget.SelCombinationDropDown.setCurrentIndex(currentCombination) #this will refresh
        else:
            self._showMatplotlibVisualization(bUpdatePlot=True,OverWriteIsppa=currentIsppa)
        
    @Slot()
    def ExportMaps(self):
        OutName=self._NiftiThermalNames[self.Widget.SelCombinationDropDown.currentIndex()]
        SelIsppa=self.Widget.IsppaSpinBox.value()
        IsppaRatio=SelIsppa/self.Config['BaseIsppa']
        BasePath = OutName.split('_DataForSim')[0]
        
        OutName = OutName.replace('_DataForSim','')
        OutName = OutName.split('-Isppa')[0] + ('_Isppa_%2.1fW' % (SelIsppa)).replace('.','p') + '-PRF' + OutName.split('-PRF')[1]
        OutName+='.nii.gz'
        print(OutName)
        
        suffix='_FullElasticSolution_Sub_NORM.nii.gz'
        if self._MainApp.Config['TxSystem'] not in ['CTX_500','CTX_250','DPX_500','Single','H246','BSonix','CTX_250_2ch',
                                                    'H246','R15287','R15473','DPXPC_300']:
            if hasattr(self._AcSimForm(),'RefocusingcheckBox') and self._AcSimForm().RefocusingcheckBox.isChecked():
                suffix='_FullElasticSolutionRefocus_Sub_NORM.nii.gz'
        if self._IsMergedTab():
            # The combined maps live on the merged grid; take the affine from the
            # merged acoustic NORM instead of a per-trajectory solution file.
            nidata = nibabel.load(self._MainApp._merged_prefix_path + 'Merged_NORM.nii.gz')
        else:
            BasePath+=suffix
            nidata = nibabel.load(BasePath)
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        if 'BaselineTemperature' in DataThermal:
            BaselineTemperature=DataThermal['BaselineTemperature']
        else:
            BaselineTemperature=37.0
        Tmap=(DataThermal['TempEndFUS']-BaselineTemperature)*IsppaRatio+BaselineTemperature
        Tmap=np.flip(Tmap,axis=2)
        nii=nibabel.Nifti1Image(Tmap.astype(np.float32),affine=nidata.affine)
        nii.to_filename(OutName)

        pressureField = DataThermal['p_map'] * np.sqrt(IsppaRatio)
        
        DensityMap=DataThermal['MaterialList']['Density'][DataThermal['MaterialMap']]
        SoSMap=    DataThermal['MaterialList']['SoS'][DataThermal['MaterialMap']]
        intensityField=pressureField**2/2/DensityMap/SoSMap/1e4

        pressureField=np.flip(pressureField,axis=2)
        intensityField=np.flip(intensityField,axis=2)

        OutName2=OutName.replace('ThermalField','PressureField')
        nii=nibabel.Nifti1Image(pressureField.astype(np.float32),affine=nidata.affine)
        nii.to_filename(OutName2)

        OutName3=OutName.replace('ThermalField','IntensityField')
        nii=nibabel.Nifti1Image(intensityField.astype(np.float32),affine=nidata.affine)
        nii.to_filename(OutName3)

        MIField = pressureField/1e6/np.sqrt(self._MainApp._Frequency/1e6)
        OutName4=OutName.replace('ThermalField','MI')
        nii=nibabel.Nifti1Image(MIField.astype(np.float32),affine=nidata.affine)
        nii.to_filename(OutName4)
            

        #If running with Brainsight, we save the path of thermal map
        if self._MainApp.Config['bInUseWithBrainsight']:
            with open(self._MainApp.Config['Brainsight-ThermalOutput'],'w') as f:
                f.write(OutName)
        txt =  'Thermal map file\n' + os.path.basename(OutName) +',\n'
        txt += 'Pressure map file\n' + os.path.basename(OutName2) +',\n'
        txt += 'Intensiy map file\n' + os.path.basename(OutName3) +',\n'
        txt += 'MI map file\n' + os.path.basename(OutName4) +',\n'
        txt += 'saved at:\n '+os.path.dirname(OutName)

        maxL=np.max([len(os.path.basename(OutName)), len(os.path.dirname(OutName))])
        if 'BABEL_PYTEST' not in os.environ:
            msgBox = DialogShowText(txt,"Saved maps")
            msgBox.exec()


class DialogShowText(QDialog):
    def __init__(self, text,title,parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel(text)
        message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        message.setStyleSheet(f"qproperty-alignment: {int(Qt.AlignmentFlag.AlignCenter)};")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    

class RunThermalSim(QObject):

    finished = Signal()
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self,mainApp,case,bRefocus,ExtraData,bMergedSimulation=False,MergedPressureRatio=[]):
         super(RunThermalSim, self).__init__()
         self._mainApp=mainApp
         # Captured on the main thread for the active thermal trajectory, so the
         # worker never reads AcSim's live/active state (which Step 3 must not
         # disturb): the acoustic case file, refocus selection and device extras.
         self._case=case
         self._bRefocus=bRefocus
         self._ExtraData=ExtraData
         self._bMergedSimulation=bMergedSimulation
         self._MergedPressureRatio=MergedPressureRatio

    def run(self):

        case=self._case
        print('Calculating thermal maps for configurations\n',self._mainApp.ThermalSim.Config['AllDC_PRF_Duration'])
        T0=time.time()
        kargs={}
        kargs['deviceName']=self._mainApp.Config['ComputingDevice']
        kargs['COMPUTING_BACKEND']=self._mainApp.Config['ComputingBackend']
        kargs['Isppa']=self._mainApp.ThermalSim.Config['BaseIsppa']
        kargs['Frequency']=self._mainApp._Frequency
        kargs['BaselineTemperature']=self._mainApp.Config['BaselineTemperature']
        kargs['LimitBHTEIterationsPerProcess']=self._mainApp.Config['LimitBHTEIterationsPerProcess']
        kargs['bForceHomogenousMedium']=self._mainApp.Config['bForceHomogenousMedium']
        kargs['HomogenousMediumValues']=self._mainApp.Config['HomogenousMediumValues']
        kargs['bForceNoAbsorptionSkullScalp']=self._mainApp.Config['bForceNoAbsorptionSkullScalp']
        kargs['bConcatenateSimulations']=self._mainApp.ThermalSim.Config['bConcatenateSimulations']
        kargs['bMergedSimulation']=self._bMergedSimulation
        kargs['MergedPressureRatio']=self._MergedPressureRatio

        kargs['TxSystem']=self._mainApp.Config['TxSystem']
        if kargs['TxSystem'] in ['CTX_500','CTX_250','DPX_500','DPXPC_300','CTX_250_2ch',
                                 'Single','H246','BSonix','R15287','R15473']:
            kargs['sel_p']='p_amp'
        else:
            if self._bRefocus:
                kargs['sel_p']='p_amp_refocus'
            else:
                kargs['sel_p']='p_amp'

        # Start mask generation as separate process.
        queue=Queue()
        ExtraData=self._ExtraData
        fieldWorkerProcess = Process(target=CalculateThermalProcess,
                                    args=(queue,case,self._mainApp.ThermalSim.Config['AllDC_PRF_Duration'],ExtraData),
                                    kwargs=kargs)
        fieldWorkerProcess.start()      
        # progress.
        T0=time.time()
        bNoError=True
        while fieldWorkerProcess.is_alive():
            time.sleep(0.1)
            while queue.empty() == False:
                cMsg=queue.get()
                print(cMsg,end='')
                if 'CTS:' in cMsg:
                    self.logTelemetry.emit(cMsg)
                if '--Babel-Brain-Low-Error' in cMsg:
                    self.logTelemetry.emit("CTS:L1:S3: "+cMsg)
                    bNoError=False  
        fieldWorkerProcess.join()
        while queue.empty() == False:
            cMsg=queue.get()
            if 'CTS:' in cMsg:
                self.logTelemetry.emit(cMsg)
            if '--Babel-Brain-Low-Error' in cMsg:
                self.logTelemetry.emit("CTS:L1:S3: "+cMsg)
                bNoError=False 
        if bNoError:
            TEnd=time.time()
            TotalTime = TEnd-T0
            print('Total time',TotalTime)
            print("*"*40)
            print("*"*5+" DONE thermal simulation.")
            print("*"*40)
            self.logTelemetry.emit("CTS:L2:S3: TOTAL TIME " + str(TotalTime))
            self._mainApp.UpdateComputationalTime('thermal',TotalTime)
            self.finished.emit()
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()

if __name__ == "__main__":
    app = QApplication([])
    widget = Babel_Thermal()
    widget.show()
    sys.exit(app.exec_())
