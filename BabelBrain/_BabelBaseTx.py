'''
Base Class for Tx GUI, not to be instantiated directly
'''

from PySide6.QtWidgets import QWidget, QVBoxLayout,QMessageBox
from PySide6.QtCore import Slot
from PySide6.QtGui import QPalette
from BabelViscoFDTD.H5pySimple import ReadFromH5py

import numpy as np
import os
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,NavigationToolbar2QT)
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

from skimage.measure import label, regionprops, regionprops_table
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

    def ExportStep2Results(self,Results):
        FocIJK=np.ones((4,1))
        FocIJK[:3,0]=np.array(np.where(self._MainApp._FinalMask==5)).flatten()

        FocXYZ=self._MainApp._MaskData.affine@FocIJK
        FocIJKAdjust=FocIJK.copy()
        #we adjust in steps
        FocIJKAdjust[0,0]+=self.Widget.XMechanicSpinBox.value()/self._MainApp._MaskData.header.get_zooms()[0]
        FocIJKAdjust[1,0]+=self.Widget.YMechanicSpinBox.value()/self._MainApp._MaskData.header.get_zooms()[1]

        FocXYZAdjust=self._MainApp._MaskData.affine@FocIJKAdjust
        AdjustmentInRAS=(FocXYZ-FocXYZAdjust).flatten()[:3]

        print('AdjustmentInRAS recalc',AdjustmentInRAS)
        print('AdjustmentInRAS orig',Results['AdjustmentInRAS'])


        fnameTrajectory=self._MainApp.ExportTrajectory(CorX=Results['AdjustmentInRAS'][0],
                                        CorY=Results['AdjustmentInRAS'][1],
                                        CorZ=Results['AdjustmentInRAS'][2])
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
        self.Widget.ShowWaterResultscheckBox.stateChanged.connect(self.UpdateAcResults)
        self.Widget.HideMarkscheckBox.stateChanged.connect(self.UpdateAcResults)

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
    
    @Slot()
    def UpdateAcResults(self):
        '''
        This is a common function for most Tx to show results
        '''
        self._MainApp.SetSuccesCode()
        self.Widget.CalculateMechAdj.setEnabled(True)
        if self._bRecalculated:
            self._MainApp.hideClockDialog()
            if self.Widget.ShowWaterResultscheckBox.isEnabled()== False:
                self.Widget.ShowWaterResultscheckBox.setEnabled(True)
            if self.Widget.HideMarkscheckBox.isEnabled()== False:
                self.Widget.HideMarkscheckBox.setEnabled(True)
            self._MainApp.Widget.tabWidget.setEnabled(True)
            self._MainApp.ThermalSim.setEnabled(True)
            Water=ReadFromH5py(self._WaterSolName)
            Skull=ReadFromH5py(self._FullSolName)

            extrasuffix=self.GetExtraSuffixAcFields()

            self._MainApp._BrainsightInput=self._MainApp._prefix_path+extrasuffix+'FullElasticSolution_Sub_NORM.nii.gz'

            self.ExportStep2Results(Skull)    

            LocTarget=Skull['TargetLocation']
            print(LocTarget)

            for d in [Water,Skull]:
                for t in ['p_amp','MaterialMap']:
                    d[t]=np.ascontiguousarray(np.flip(d[t],axis=2))

            DistanceToTarget=self.Widget.DistanceSkinLabel.property('UserData')
            dx=  np.mean(np.diff(Skull['x_vec']))

            Water['z_vec']*=1e3
            Skull['z_vec']*=1e3
            Skull['x_vec']*=1e3
            Skull['y_vec']*=1e3
            Skull['MaterialMap'][Skull['MaterialMap']==3]=2
            Skull['MaterialMap'][Skull['MaterialMap']==4]=3

            IWater=Water['p_amp']**2/2/Water['Material'][0,0]/Water['Material'][0,1]

            DensityMap=Skull['Material'][:,0][Skull['MaterialMap']]
            SoSMap=    Skull['Material'][:,1][Skull['MaterialMap']]

            ISkull=Skull['p_amp']**2/2/Skull['Material'][4,0]/Skull['Material'][4,1]

            IntWaterLocation=IWater[LocTarget[0],LocTarget[1],LocTarget[2]]
            IntSkullLocation=ISkull[LocTarget[0],LocTarget[1],LocTarget[2]]
            
            ISkull[Skull['MaterialMap']!=3]=0
            cxr,cyr,czr=np.where(ISkull==ISkull.max())
            cxr=cxr[0]
            cyr=cyr[0]
            czr=czr[0]

            EnergyAtFocusSkull=ISkull[:,:,czr].sum()*dx**2

            cxr,cyr,czr=np.where(IWater==IWater.max())
            cxr=cxr[0]
            cyr=cyr[0]
            czr=czr[0]
            
            ISkull/=ISkull.max()
            IWater/=IWater.max()

            dz=np.diff(Skull['z_vec']).mean()
            Zvec=Skull['z_vec'].copy()
            Zvec-=Zvec[LocTarget[2]]
            Zvec+=DistanceToTarget
            XX,ZZ=np.meshgrid(Skull['x_vec'],Zvec)
            self._XX = XX
            self._ZZX = ZZ
            YY,ZZ=np.meshgrid(Skull['y_vec'],Zvec)
            self._YY = YY
            self._ZZY = ZZ

            self.Widget.IsppaScrollBars.set_default_values(LocTarget,Skull['x_vec']-Skull['x_vec'][LocTarget[0]],Skull['y_vec']-Skull['y_vec'][LocTarget[1]])

            self._Water = Water
            self._IWater = IWater
            self._Skull = Skull
            self._ISkull = ISkull
            self._DistanceToTarget = DistanceToTarget
        
            if hasattr(self,'_figAcField'):
                    children = []
                    for i in range(self._layout.count()):
                        child = self._layout.itemAt(i).widget()
                        if child:
                            children.append(child)
                    for child in children:
                        child.deleteLater()
                    delattr(self,'_figAcField')
                    self.Widget.AcField_plot1.repaint()
                    
            Total_Distance,X_dist,Y_dist,Z_dist=self.CalculateDistancesTarget()
            self.Widget.DistanceTargetLabel.setText('[%2.1f, %2.1f ,%2.1f]' %(X_dist,Y_dist,Z_dist))
        
        if self.Widget.ShowWaterResultscheckBox.isChecked():
            Field=self._IWater
        else:
            Field=self._ISkull

        SelY, SelX = self.Widget.IsppaScrollBars.get_scroll_values()

        if hasattr(self,'_figAcField'):
            if hasattr(self,'_imContourf1'):
                for c in [self._imContourf1,self._imContourf2,self._contour1,self._contour2]:
                    for coll in c.collections:
                        coll.remove()
                del self._imContourf1
                del self._imContourf2
                del self._contour1
                del self._contour2

            self._imContourf1=self._static_ax1.contourf(self._XX,self._ZZX,Field[:,SelY,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            self._contour1 = self._static_ax1.contour(self._XX,self._ZZX,self._Skull['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)

            self._imContourf2=self._static_ax2.contourf(self._YY,self._ZZY,Field[SelX,:,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            self._contour2 = self._static_ax2.contour(self._YY,self._ZZY,self._Skull['MaterialMap'][SelX,:,:].T,[0,1,2,3], cmap=plt.cm.gray)

            self._figAcField.canvas.draw_idle()
        else:
            self._figAcField=Figure(figsize=(14, 12))

            if not hasattr(self,'_layout'):
                self._layout = QVBoxLayout(self.Widget.AcField_plot1)

            self.static_canvas = FigureCanvas(self._figAcField)
            toolbar=NavigationToolbar2QT(self.static_canvas,self)
            self._layout.addWidget(toolbar)
            self._layout.addWidget(self.static_canvas)
            static_ax1,static_ax2 = self.static_canvas.figure.subplots(1,2)
            self._static_ax1 = static_ax1
            self._static_ax2 = static_ax2

            self._imContourf1=static_ax1.contourf(self._XX,self._ZZX,Field[:,SelY,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax1)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour1 = static_ax1.contour(self._XX,self._ZZX,self._Skull['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax1.set_aspect('equal')
            static_ax1.set_xlabel('X mm')
            static_ax1.set_ylabel('Z mm')
            static_ax1.invert_yaxis()
            self._marker1,=static_ax1.plot(0,self._DistanceToTarget,'+k',markersize=18)

            self._imContourf2=static_ax2.contourf(self._YY,self._ZZY,Field[SelX,:,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax2)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour2 = static_ax2.contour(self._YY,self._ZZY,self._Skull['MaterialMap'][SelX,:,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax2.set_aspect('equal')
            static_ax2.set_xlabel('Y mm')
            static_ax2.set_ylabel('Z mm')
            static_ax2.invert_yaxis()
            self._marker2,=static_ax2.plot(0,self._DistanceToTarget,'+k',markersize=18)

        self._figAcField.set_facecolor(self._MainApp._BackgroundColorFigures)

        mc=[0.0,0.0,0.0,1.0]
        if self.Widget.HideMarkscheckBox.isChecked():
             mc[3] = 0.0
        self._marker1.set_markerfacecolor(mc)
        self._marker2.set_markerfacecolor(mc)
        self.Widget.IsppaScrollBars.update_labels(SelX, SelY)
        self._bRecalculated = False
 
    def GetExport(self):
        Export={}
        Export['DistanceSkinToTarget']=self.Widget.DistanceSkinLabel.property('UserData')
        return Export
    
    def GetExtraDataForThermal(self):
        #we use this to save extra data in thermal files if required,
        #to be redefined in child class 
        return {}
    
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
