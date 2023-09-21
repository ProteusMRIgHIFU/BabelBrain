'''
Base Class for Tx GUI, not to be instantiated directly
'''

from PySide6.QtWidgets import QWidget, QVBoxLayout
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

class BabelBaseTx(QWidget):
    def __init__(self,parent=None):
        super(BabelBaseTx, self).__init__(parent)
    
    @Slot()
    def UpdateAcResults(self):
        '''
        This is a common function for most Tx to show results
        '''
        if self._bRecalculated:
            #this will generate a modified trajectory file
            if self.Widget.ShowWaterResultscheckBox.isEnabled()== False:
                self.Widget.ShowWaterResultscheckBox.setEnabled(True)
            self._MainApp.Widget.tabWidget.setEnabled(True)
            self._MainApp.ThermalSim.setEnabled(True)
            Water=ReadFromH5py(self._WaterSolName)
            Skull=ReadFromH5py(self._FullSolName)
            if self._MainApp._bInUseWithBrainsight:
                with open(self._MainApp._BrainsightSyncPath+os.sep+'Output.txt','w') as f:
                    f.write(self._MainApp._BrainsightInput)    
            self._MainApp.ExportTrajectory(CorX=Skull['AdjustmentInRAS'][0],
                                        CorY=Skull['AdjustmentInRAS'][1],
                                        CorZ=Skull['AdjustmentInRAS'][2])

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

            EnergyAtFocusWater=IWater[:,:,czr].sum()*dx**2

            print('EnergyAtFocusWater',EnergyAtFocusWater,'EnergyAtFocusSkull',EnergyAtFocusSkull)
            
            Factor=EnergyAtFocusWater/EnergyAtFocusSkull
            print('*'*40+'\n'+'*'*40+'\n'+'Correction Factor for Isppa',Factor,'\n'+'*'*40+'\n'+'*'*40+'\n')
            
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
            static_ax1.plot(0,self._DistanceToTarget,'+y',markersize=18)

            self._imContourf2=static_ax2.contourf(self._YY,self._ZZY,Field[SelX,:,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax2)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour2 = static_ax2.contour(self._YY,self._ZZY,self._Skull['MaterialMap'][SelX,:,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax2.set_aspect('equal')
            static_ax2.set_xlabel('Y mm')
            static_ax2.set_ylabel('Z mm')
            static_ax2.invert_yaxis()
            static_ax2.plot(0,self._DistanceToTarget,'+y',markersize=18)

        self._figAcField.set_facecolor(np.array(self.Widget.palette().color(QPalette.Window).getRgb())/255)

        #f.set_title('MAIN SIMULATION RESULTS')
        self.Widget.IsppaScrollBars.update_labels(SelX, SelY)
        self._bRecalculated = False
 
