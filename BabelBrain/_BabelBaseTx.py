'''
Base Class for Tx GUI, not to be instantiated directly
'''

from PySide6.QtWidgets import QWidget, QVBoxLayout,QMessageBox, QTabWidget
from PySide6.QtCore import Slot, QThread, Qt
from BabelViscoFDTD.H5pySimple import ReadFromH5py

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,NavigationToolbar2QT)

from GUIComponents.nifti_viewer import NiftiViewerWindow
from GUIComponents.AppStyle import style_nav_toolbar

from skimage.measure import label, regionprops
import nibabel
#auxiliary functions to measure metrics in acoustic fields

def ellipsoid_axis_lengths(central_moments):
    """Compute ellipsoid major, intermediate and minor axis length.

    Parameters
    ----------
    central_moments : ndarray
        Array of central moments as given by ``moments_central`` with order 2.

    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
    return tuple([np.sqrt(20.0 * e) for e in eigvals]) 

def CalcVolumetricMetrics(Data,voxelsize,Threshold=0.5):
        '''
        Threshold=0.25 
        '''
        label_img=label(Data>=Threshold)
        props = regionprops(label_img)#, properties=('centroid',   'area', 'moments_central','axis_major_length','axis_minor_length'))
        Res={}
        if len(props)>1:
            Volumes=[]
            for p in props:
                Volumes.append(p['area'])  
            p=props[np.argmax(Volumes)]
        else:
            p=props[0]
        Res['centroid']=p['centroid']*voxelsize
        Res['volume']=p['area']*np.prod(voxelsize)
        Axes=ellipsoid_axis_lengths(p['moments_central'])
        Res['long_axis']=Axes[0]*voxelsize[0]
        Res['minor_axis_1']=Axes[1]*voxelsize[1]
        Res['minor_axis_2']=Axes[2]*voxelsize[2]
        # print(Res)
        return Res

#Main Tx base class
class BabelBaseTx(QWidget):
    def __init__(self,parent=None):
        super(BabelBaseTx, self).__init__(parent)

    # ──────────────────────────────────────────────────────────────────────
    # Step-2 layout: one tab per trajectory, each an independent transducer
    # form (own inputs, scrollbars, plot and Calculate action).  self.Widget
    # always points at the active tab's form, and self._TrajectoryNumber at its
    # index; both are re-synced on tab change.  Because every tab is a distinct
    # form instance, parameter values are preserved per tab automatically.
    #
    # Each device subclass provides:
    #   _CreateForm()  – build and return a fresh transducer form (TxPanelBase).
    #   _WirePanel()   – wire the form currently held in self.Widget (signals,
    #                    spin-box ranges, per-form IsppaScrollBars, up_load_ui()).
    # ──────────────────────────────────────────────────────────────────────

    def _setupTrajectoryTabs(self):
        IDs = list(self._MainApp.Config['ID'])
        self._txTabs = QTabWidget(self)
        self._txTabs.tabBar().setElideMode(Qt.ElideNone)
        self._txTabs.tabBar().setExpanding(False)
        self._txTabs.setUsesScrollButtons(True)
        # Single trajectory (the common case): hide the tab bar and drop the pane
        # frame entirely so Step 2 looks like the original single-panel view.
        # Several trajectories: keep the tab bar and only a top line under the tab
        # row (its left/right/bottom borders would double up against the Step-2
        # frame and look weird).
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
        self._acPanels = [None] * len(IDs)
        for i, tid in enumerate(IDs):
            form = self._CreateForm()
            self._txTabs.addTab(form, str(tid))
            self._Widgets.append(form)
            # Point at this form while wiring so the connections bind to *its*
            # widgets (button -> self.RunSimulation, etc.).
            self.Widget = form
            self._TrajectoryNumber = i
            self._WirePanel()

        # Activate the first tab; only now listen for user tab switches so the
        # addTab loop above doesn't fire the handler prematurely.
        self.Widget = self._Widgets[0]
        self._TrajectoryNumber = 0
        self._txTabs.setCurrentIndex(0)
        self._txTabs.currentChanged.connect(self._OnTrajectoryTabChanged)

    @Slot(int)
    def _OnTrajectoryTabChanged(self, idx):
        if not hasattr(self, '_Widgets') or idx < 0 or idx >= len(self._Widgets):
            return
        self.Widget = self._Widgets[idx]
        self._TrajectoryNumber = idx
        # Re-point the controller's active-trajectory state so Mechanical-Adjust,
        # the distance readout and the nifti reload act on the visible tab.  The
        # form keeps its own plot and scrollbar positions, so nothing is redrawn
        # or reset here (this is what makes the scrollbars independent per tab).
        panel = self._acPanels[idx]
        if panel is not None and panel.get('figure') is not None:
            self._ActivatePanel(panel)

    def _CreateForm(self):
        raise NotImplementedError("Subclasses must build their transducer form")

    def _WirePanel(self):
        raise NotImplementedError("Subclasses must wire their transducer form")

    def _SyncActiveTrajectoryFromMainApp(self):
        '''
        Point self.Widget / _TrajectoryNumber (and the visible tab) at the
        trajectory Step 1 just finished.  Step 1 calls NotifyGeneratedMask once
        per trajectory (UpdateMask -> UpdateAcousticTab) with _MainApp._TrajectoryNumber
        set, so each device's NotifyGeneratedMask calls this first to initialize
        the matching tab's form.
        '''
        idx = self._MainApp._TrajectoryNumber
        if hasattr(self, '_Widgets') and 0 <= idx < len(self._Widgets):
            self._TrajectoryNumber = idx
            self.Widget = self._Widgets[idx]
            self._txTabs.setCurrentIndex(idx)

    def CalculateDistanceFromSkin(self):
        VoxelSize=self._MainApp._MaskNib[self._TrajectoryNumber].header.get_zooms()[0]
        TargetLocation =np.array(np.where(self._MainApp._FinalMask[self._TrajectoryNumber]==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[self._TrajectoryNumber][TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize
        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)
        return DistanceFromSkin

    def ExportStep2Results(self,Results):
        FocIJK=np.ones((4,1))
        FocIJK[:3,0]=np.array(np.where(self._MainApp._FinalMask[self._TrajectoryNumber]==5)).flatten()

        FocXYZ=self._MainApp._MaskNib[self._TrajectoryNumber].affine@FocIJK
        FocIJKAdjust=FocIJK.copy()
        #we adjust in steps
        FocIJKAdjust[0,0]+=self.Widget.XMechanicSpinBox.value()/self._MainApp._MaskNib[0].header.get_zooms()[0]
        FocIJKAdjust[1,0]+=self.Widget.YMechanicSpinBox.value()/self._MainApp._MaskNib[0].header.get_zooms()[1]

        FocXYZAdjust=self._MainApp._MaskNib[self._TrajectoryNumber].affine@FocIJKAdjust
        AdjustmentInRAS=(FocXYZ-FocXYZAdjust).flatten()[:3]

        print('AdjustmentInRAS recalc',AdjustmentInRAS)
        print('AdjustmentInRAS orig',Results['AdjustmentInRAS'])


        fnameTrajectory=self._MainApp.ExportTrajectory(CorX=Results['AdjustmentInRAS'][0],
                                        CorY=Results['AdjustmentInRAS'][1],
                                        CorZ=Results['AdjustmentInRAS'][2],
                                        Ntraj=self._TrajectoryNumber)
        if self._MainApp.Config['bInUseWithBrainsight']:
            with open(self._MainApp.Config['Brainsight-Output'],'w') as f:
                f.write(self._MainApp._BrainsightInput)
            with open(self._MainApp.Config['Brainsight-Target'],'w') as f:
                f.write(fnameTrajectory)

    def GetExtraSuffixAcFields(self):
        #By default, it returns empty string, useful when dealing with user-specified geometry
        return ""

    def up_load_ui(self):
        #please note this one needs to be called after child class called its load_ui
        self.Widget.CalculateAcField.clicked.connect(self.RunSimulation)
        self.Widget.ShowWaterResultscheckBox.stateChanged.connect(self._showMatplotlibVisualization)
        self.Widget.HideMarkscheckBox.stateChanged.connect(self._showMatplotlibVisualization)
        if hasattr(self.Widget,'CombineTrajectories'):
            self.Widget.CombineTrajectories.clicked.connect(self.CombineTrajectories)


    @Slot()
    def NotifyError(self):
        self._MainApp.SetErrorAcousticsCode()
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

    # ──────────────────────────────────────────────────────────────────────
    # Acoustic simulation execution (Step 2)
    #
    # RunSimulation is a single template method shared by every transducer.
    # The parts that genuinely differ between devices are isolated in the
    # hooks below; subclasses override only what they need:
    #
    #   _ResolveSimulationFilenames()  – set self._FullSolName / self._WaterSolName
    #                                    (and stash any device parameters needed
    #                                    later by _CreateAcousticWorker).
    #   _ExistingSimulationFiles()     – True if results already on disk
    #                                    (default: scalar single-file check;
    #                                    phased arrays override for file lists).
    #   _PromptReuseOrRecalc()         – when files exist, ask the user and, on
    #                                    reload, repopulate the device widgets;
    #                                    return True to (re)compute.
    #   _CreateAcousticWorker()        – build the device's RunAcousticSim worker
    #                                    with its specific parameters.
    #   _SimulationFinishedSlot()      – slot connected to worker.finished
    #                                    (default: UpdateAcResults).
    #   _ReloadExistingResults()       – called when results are reused as-is
    #                                    (default: UpdateAcResults).
    # ──────────────────────────────────────────────────────────────────────

    def RunSimulation(self):
        # Runs only the active trajectory (the visible tab).  The user runs each
        # trajectory's tab one at a time, adjusting and re-running as needed;
        # self._TrajectoryNumber tracks the active tab.
        self._ResolveSimulationFilenames()
        if self._ExistingSimulationFiles():
            bCalcFields = self._PromptReuseOrRecalc()
        else:
            bCalcFields = True
        self._bRecalculated = True
        if bCalcFields:
            self._LaunchAcousticSim(self._CreateAcousticWorker(),
                                    self._SimulationFinishedSlot())
        else:
            self._ReloadExistingResults()

    def _LaunchAcousticSim(self, worker, on_finished):
        '''
        Common worker-thread plumbing shared by every transducer.  Moves the
        device's RunAcousticSim worker onto a QThread and wires the standard
        finished/error/telemetry signals before starting it.
        '''
        self._MainApp.Widget.tabWidget.setEnabled(False)
        self.thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(on_finished)
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
        self._MainApp.showClockDialog()

    # ── Default hook implementations (overridable per device) ───────────────

    def _ResolveSimulationFilenames(self):
        raise NotImplementedError(
            "Subclasses must set self._FullSolName / self._WaterSolName")

    def _ExistingSimulationFiles(self):
        return os.path.isfile(self._FullSolName) and \
               os.path.isfile(self._WaterSolName)

    def _PromptReuseOrRecalc(self):
        raise NotImplementedError(
            "Subclasses must implement the reuse/recalculate prompt")

    def _CreateAcousticWorker(self):
        raise NotImplementedError(
            "Subclasses must build their RunAcousticSim worker")

    def _SimulationFinishedSlot(self):
        return self.UpdateAcResults

    def _ReloadExistingResults(self):
        self.UpdateAcResults()

    # ──────────────────────────────────────────────────────────────────────
    # Acoustic-result display — rendered into the active trajectory tab's own
    # plot host (self.Widget.AcField_plot1).  Each trajectory keeps its full
    # result/plot state in self._acPanels[i]; switching tabs re-points self._*
    # at the visible trajectory via _ActivatePanel (the form keeps its own
    # canvas and scrollbar positions, so nothing is rebuilt or reset).
    #
    # Device specifics live in three small hooks:
    #   _LoadAcResultData(panel) – read the H5 results and stash the per-panel
    #                              arrays (base: single field; phased: field cols).
    #   _AcResultFigure()        – the matplotlib Figure (phased uses a larger one).
    #   _GetActiveFields(panel)  – return (IWater, ISkull) to display for the panel
    #                              (phased selects by the multifocus dropdown).
    # ──────────────────────────────────────────────────────────────────────

    def _AcPanel(self, idx):
        '''Per-trajectory Step-2 result/plot state, lazily created, keyed by tab.'''
        if self._acPanels[idx] is None:
            host = self._Widgets[idx].AcField_plot1
            lay = host.layout()
            if lay is None:
                lay = QVBoxLayout(host)
                lay.setContentsMargins(0, 0, 0, 0)
            self._acPanels[idx] = {'layout': lay, 'figure': None}
        return self._acPanels[idx]

    def _showMatplotlibVisualization(self):
        '''
        (Re)draw the acoustic-field result for the active trajectory's tab.
        On a fresh result (_bRecalculated) the figure is built into that tab's
        own AcField_plot1; user interactions (slice scroll / water-skull toggle /
        hide-marks / multifocus dropdown) re-render the same tab.
        '''
        panel = self._AcPanel(self._TrajectoryNumber)
        if self._bRecalculated:
            if self.Widget.ShowWaterResultscheckBox.isEnabled() == False:
                self.Widget.ShowWaterResultscheckBox.setEnabled(True)
            if self.Widget.HideMarkscheckBox.isEnabled() == False:
                self.Widget.HideMarkscheckBox.setEnabled(True)
            self._MainApp.Widget.tabWidget.setEnabled(True)

            self._LoadAcResultData(panel)        # device-specific data load
            self._BuildAcResultFigure(panel)     # fresh fig/canvas in this tab's plot host
            self._ActivatePanel(panel)           # mirror data into self._* aliases
            self._SyncScrollAndDistance(panel)   # scrollbar defaults + distance label
            self._RenderAcResultPanel(panel)
            self._bRecalculated = False
        else:
            if panel.get('figure') is None:
                return
            self._ActivatePanel(panel)
            self._RenderAcResultPanel(panel)

    def _ActivatePanel(self, panel):
        '''
        Mirror a tab's stored result state into the self._* attributes that the
        shared helpers (CalculateDistancesTarget / CalculateMechAdj /
        UpdateAcResults / _RenderAcResultPanel) read, so they act on the visible
        trajectory.
        '''
        self._Skull = panel['Skull']
        self._XX, self._ZZX = panel['XX'], panel['ZZX']
        self._YY, self._ZZY = panel['YY'], panel['ZZY']
        self._DistanceToTarget = panel['DistanceToTarget']
        self._figAcField = panel['figure']
        self._static_ax1 = panel.get('static_ax1')
        self._static_ax2 = panel.get('static_ax2')
        self._marker1 = panel.get('marker1')
        self._marker2 = panel.get('marker2')
        self._FullSolName = panel['FullSolName']
        self._WaterSolName = panel['WaterSolName']
        if 'SDR' in panel:
            self._SDR = panel['SDR']
        if 'AcResults' in panel:
            self._AcResults = panel['AcResults']
        if 'LastDistanceConeToFocus' in panel:
            self._LastDistanceConeToFocus = panel['LastDistanceConeToFocus']
        if 'ISkullCol' in panel:
            self._ISkullCol = panel['ISkullCol']
            self._IWaterCol = panel['IWaterCol']
        IWater, ISkull = self._GetActiveFields(panel)
        self._IWater = IWater
        self._ISkull = ISkull

    def _SyncScrollAndDistance(self, panel):
        '''Set this tab's slice scrollbar defaults and distance readout (build only).'''
        self.Widget.IsppaScrollBars.set_default_values(
            panel['LocTarget'], panel['xvec'], panel['yvec'])
        Total_Distance, X_dist, Y_dist, Z_dist = self.CalculateDistancesTarget()
        self.Widget.DistanceTargetLabel.setText(
            '[%2.1f, %2.1f ,%2.1f]' % (X_dist, Y_dist, Z_dist))

    def _BuildAcResultFigure(self, panel):
        '''Create a fresh figure/canvas/axes/markers inside this tab's AcField_plot1.'''
        # Clear any previous content (toolbar + canvas) from this tab's plot host.
        while (child := panel['layout'].takeAt(0)) is not None:
            w = child.widget()
            if w is not None:
                w.deleteLater()
        for k in ('imContourf1', 'imContourf2', 'contour1', 'contour2',
                  'airmask1', 'airmask2'):
            panel[k] = None
        panel['cbDone'] = False

        fig = self._AcResultFigure()
        panel['figure'] = fig
        canvas = FigureCanvas(fig)
        panel['canvas'] = canvas
        self.static_canvas = canvas
        toolbar = style_nav_toolbar(NavigationToolbar2QT(canvas, self))
        panel['layout'].addWidget(toolbar)
        panel['layout'].addWidget(canvas)
        # Each plot box is centered on its scrollbar (left col x=0.25, right
        # col x=0.75). A dedicated colorbar axis keeps the plot itself
        # centered (plt.colorbar(ax=...) would shrink/shift the plot).
        static_ax1 = fig.add_axes([0.10, 0.12, 0.30, 0.80])
        cax1       = fig.add_axes([0.43, 0.12, 0.015, 0.80])
        static_ax2 = fig.add_axes([0.60, 0.12, 0.30, 0.80])
        cax2       = fig.add_axes([0.93, 0.12, 0.015, 0.80])
        panel['static_ax1'] = static_ax1
        panel['static_ax2'] = static_ax2
        panel['cax1'] = cax1
        panel['cax2'] = cax2
        for ax, xl in ((static_ax1, 'X mm'), (static_ax2, 'Y mm')):
            ax.set_aspect('equal')
            ax.set_xlabel(xl)
            ax.set_ylabel('Z mm')
        panel['marker1'], = static_ax1.plot(0, panel['DistanceToTarget'], '+k', markersize=18)
        panel['marker2'], = static_ax2.plot(0, panel['DistanceToTarget'], '+k', markersize=18)
        fig.set_facecolor(self._MainApp._BackgroundColorFigures)

    def _RenderAcResultPanel(self, panel):
        '''Draw / refresh the acoustic-field contours in a panel from self._* data.'''
        homog = self._MainApp.Config['bForceHomogenousMedium']
        Skull = self._Skull
        XX, ZZX, YY, ZZY = self._XX, self._ZZX, self._YY, self._ZZY
        ax1, ax2 = panel['static_ax1'], panel['static_ax2']

        SelY, SelX = self.Widget.IsppaScrollBars.get_scroll_values()
        if self.Widget.ShowWaterResultscheckBox.isChecked():
            Field = self._IWater
        else:
            Field = self._ISkull

        # Remove the previous contour/contourf artists for this panel.
        if panel.get('imContourf1') is not None:
            listObjects = [panel['imContourf1'], panel['imContourf2']]
            if not homog:
                listObjects += [panel['contour1'], panel['contour2']]
                if 'AirMask' in Skull:
                    listObjects += [panel['airmask1'], panel['airmask2']]
            for c in listObjects:
                try:  # this is for old Matplotlib
                    for coll in c.collections:
                        coll.remove()
                except:
                    c.remove()

        panel['imContourf1'] = ax1.contourf(XX, ZZX, Field[:, SelY, :].T, np.arange(2, 22, 2) / 20, cmap=plt.cm.jet)
        if not homog:
            panel['contour1'] = ax1.contour(XX, ZZX, Skull['MaterialMap'][:, SelY, :].T, [0, 1, 2], colors='k', linestyles=':')
            if 'AirMask' in Skull:
                AirMap = Skull['AirMask'][:, SelY, :].T
                AirMap = np.ma.masked_where(AirMap == 0, AirMap)
                panel['airmask1'] = ax1.contourf(XX, ZZX, AirMap, [0, 1], cmap=plt.cm.gray_r)

        panel['imContourf2'] = ax2.contourf(YY, ZZY, Field[SelX, :, :].T, np.arange(2, 22, 2) / 20, cmap=plt.cm.jet)
        if not homog:
            panel['contour2'] = ax2.contour(YY, ZZY, Skull['MaterialMap'][SelX, :, :].T, [0, 1, 2], colors='k', linestyles=':')
            if 'AirMask' in Skull:
                AirMap = Skull['AirMask'][SelX, :, :].T
                AirMap = np.ma.masked_where(AirMap == 0, AirMap)
                panel['airmask2'] = ax2.contourf(YY, ZZY, AirMap, [0, 1], cmap=plt.cm.gray_r)

        # Colourbars and the y-axis flip are set once, with the first contourf.
        if not panel.get('cbDone'):
            h = plt.colorbar(panel['imContourf1'], cax=panel['cax1'])
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            h = plt.colorbar(panel['imContourf1'], cax=panel['cax2'])
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            ax1.invert_yaxis()
            ax2.invert_yaxis()
            panel['cbDone'] = True

        mc = [0.0, 0.0, 0.0, 1.0]
        if self.Widget.HideMarkscheckBox.isChecked():
            mc[3] = 0.0
        panel['marker1'].set_markerfacecolor(mc)
        panel['marker2'].set_markerfacecolor(mc)

        panel['figure'].canvas.draw_idle()
        self.Widget.IsppaScrollBars.update_labels(SelX, SelY)

    # ── Device hooks for the acoustic-result display ────────────────────────

    def _AcResultFigure(self):
        return Figure()

    def _GetActiveFields(self, panel):
        return panel['IWater'], panel['ISkull']

    def _LoadAcResultData(self, panel):
        '''Read the (single) skull/water H5 result and stash per-panel arrays.'''
        Water = ReadFromH5py(self._WaterSolName)
        Skull = ReadFromH5py(self._FullSolName)
        print('_FullSolName', self._FullSolName)

        if 'SDR' in Skull and hasattr(self.Widget, 'SDRLabel'):
            self._SDR = Skull['SDR']
            self.Widget.SDRLabel.setText('%0.2f' % (Skull['SDR']))
            panel['SDR'] = Skull['SDR']

        extrasuffix = self.GetExtraSuffixAcFields()
        self._MainApp._BrainsightInput = self._MainApp._prefix_path[self._TrajectoryNumber] + extrasuffix + 'FullElasticSolution_Sub_NORM.nii.gz'

        self.ExportStep2Results(Skull)

        LocTarget = Skull['TargetLocation']
        print('LocTarget', LocTarget)

        for d in [Water, Skull]:
            keys = ['p_amp', 'MaterialMap']
            if 'AirMask' in d:
                keys.append('AirMask')
            for t in keys:
                d[t] = np.ascontiguousarray(np.flip(d[t], axis=2))

        DistanceToTarget = self.Widget.DistanceSkinLabel.property('UserData')

        Water['z_vec'] *= 1e3
        Skull['z_vec'] *= 1e3
        Skull['x_vec'] *= 1e3
        Skull['y_vec'] *= 1e3
        DensityMap = Skull['Material'][:, 0][Skull['MaterialMap']]
        SoSMap = Skull['Material'][:, 1][Skull['MaterialMap']]

        Skull['MaterialMap'][Skull['MaterialMap'] == 3] = 2
        Skull['MaterialMap'][Skull['MaterialMap'] == 4] = 3

        IWater = Water['p_amp'] ** 2 / 2 / Water['Material'][0, 0] / Water['Material'][0, 1]
        ISkull = Skull['p_amp'] ** 2 / 2 / DensityMap / SoSMap

        if not self._MainApp.Config['bForceHomogenousMedium']:
            ISkull[Skull['MaterialMap'] < 3] = 0

        ISkull /= ISkull.max()
        IWater /= IWater.max()

        Zvec = Skull['z_vec'].copy()
        Zvec -= Zvec[LocTarget[2]]
        Zvec += DistanceToTarget
        XX, ZZX = np.meshgrid(Skull['x_vec'], Zvec)
        YY, ZZY = np.meshgrid(Skull['y_vec'], Zvec)

        panel.update({
            'Skull': Skull, 'Water': Water,
            'IWater': IWater, 'ISkull': ISkull,
            'XX': XX, 'ZZX': ZZX, 'YY': YY, 'ZZY': ZZY,
            'DistanceToTarget': DistanceToTarget, 'LocTarget': LocTarget,
            'xvec': Skull['x_vec'] - Skull['x_vec'][LocTarget[0]],
            'yvec': Skull['y_vec'] - Skull['y_vec'][LocTarget[1]],
            'FullSolName': self._FullSolName, 'WaterSolName': self._WaterSolName,
        })

    @Slot()
    def UpdateAcResults(self):
        '''
        This is a common function for most Tx to show results
        '''
        self._MainApp.SetSuccesCode()
        self.Widget.CalculateMechAdj.setEnabled(True)
        if self._bRecalculated:
            self._MainApp.ThermalSim.setEnabled(True)
            self._MainApp.hideClockDialog()

        self._showMatplotlibVisualization()
        NiftiSkull=nibabel.load(self._FullSolName.replace('DataForSim.h5','FullElasticSolution_Sub_NORM.nii.gz'))
        NiftiWater=nibabel.load(self._FullSolName.replace('DataForSim.h5','Water_FullElasticSolution_Sub_NORM.nii.gz'))
        self._MainApp.UpdateNiftiAcResults(NiftiSkull,NiftiWater,self._TrajectoryNumber)
        if hasattr(self.Widget,'CombineTrajectories'):
            self.Widget.CombineTrajectories.setEnabled(self._MainApp.AllAcFieldsDone())
        
    def GetExport(self):
        Export={}
        Export['DistanceSkinToTarget']=self.Widget.DistanceSkinLabel.property('UserData')
        if hasattr(self,'_SDR'):
            Export['SDR']=self._SDR
        return Export
    
    def GetExtraDataForThermal(self):
        retDict={}
        return retDict
    
    def EnableMultiPoint(self,MultiPoint):
        #MuliPoint is a list of dictionaries with entries ['X':value,'Y':value,'Z':value], each indicating steering conditions for each point
        pass #to be defined by those Tx capable of multi-point
    
    def CalculateDistancesTarget(self):
        # Get voxel size
        dx=  np.mean(np.diff(self._Skull['x_vec']))
        voxelsize=np.array([dx,dx,dx])
        
        # Determine plot to use for calculating distances
        if hasattr(self.Widget,'SelCombinationDropDown'):
            # For phased arrays
            if self.Widget.SelCombinationDropDown.isVisible():
                # For multifocal sims, use the central focal spot to find mechanical adjustments
                central_plot_index = self.Widget.SelCombinationDropDown.findText('X:0.0 Y:0.0 Z:0.0')
            else:
                # For single focus sims
                central_plot_index = 0          
            central_focal_spot_plot = self._ISkullCol[central_plot_index]
        else:
            # For single focus txs
            central_focal_spot_plot = self._ISkull
            
        stats=CalcVolumetricMetrics(central_focal_spot_plot,voxelsize)
        x_o=np.unique(self._XX)
        y_o=np.unique(self._YY)
        z_o=np.unique(self._ZZX)
        #we get the centroid in the displayed axes convention
        centroid=stats['centroid']+np.array([x_o.min(),y_o.min(),z_o.min()])
        X_dist = centroid[0]-x_o[self._Skull['TargetLocation'][0]]
        Y_dist = centroid[1]-y_o[self._Skull['TargetLocation'][1]]
        Z_dist = centroid[2]-z_o[self._Skull['TargetLocation'][2]]
        Total_Distance= np.round(np.sqrt(X_dist**2+Y_dist**2+Z_dist**2),1)
        X_dist=np.round(X_dist,1)
        Y_dist=np.round(Y_dist,1)
        Z_dist=np.round(Z_dist,1)
        return Total_Distance,X_dist,Y_dist,Z_dist
    
    @Slot()
    def CalculateMechAdj(self):
        #this calculates the required mechanical correction to center acoustic beam
        #to the target
        Total_Distance,X_correction,Y_correction,Z_correction = self.CalculateDistancesTarget()
        ret = QMessageBox.question(self,'', "The focal spot's center of mass (-6dB) "+
                                   'is [%3.1f,%3.1f]' % (X_correction,Y_correction) + " mm-off in [X,Y] relative to the target.\n"+
                                    "Do you want to apply a mechanical correction?",
                QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            curX=self.Widget.XMechanicSpinBox.value()
            curY=self.Widget.YMechanicSpinBox.value()
            self.Widget.XMechanicSpinBox.setValue(curX-X_correction)
            self.Widget.YMechanicSpinBox.setValue(curY-Y_correction)

    @Slot()
    def CombineTrajectories(self):
        from MergeNifti.MergeNiftiComplexAligned import do_complex_merge
        from pathlib import Path

        AllInputs=[]
        for p in self._MainApp._prefix_path:
            entry={}
            entry['Sub_Norm']=p+'FullElasticSolution_Sub_NORM.nii.gz'
            entry['PhaseSub_Norm']=p+'FullElasticSolutionPhase_Sub_NORM.nii.gz'
            entry['Sub']=p+'FullElasticSolution_Sub.nii.gz'
            entry['PhaseSub']=p+'FullElasticSolutionPhase_Sub.nii.gz'
            AllInputs.append(entry)

        cfgBase={}
        cfgBase['orientation']='coronal'
        cfgBase['interp']=1
        #first we do for visualization

        cfg=cfgBase.copy()
        cfg['pairs']=[]
        for e in AllInputs:
            cfg['pairs'].append({'amp':Path(e['Sub_Norm']),'phase':Path(e['PhaseSub_Norm'])})
        cfg['output']={'amp':Path(self._MainApp._merged_prefix_path+'Merged_NORM.nii.gz')}

        do_complex_merge(cfg)
        MergedNifti=nibabel.load(cfg['output']['amp'])
        self._MainApp.UpdateNiftiMergedAcResults(MergedNifti)


