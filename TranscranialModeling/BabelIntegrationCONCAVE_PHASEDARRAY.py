'''
Pipeline to execute viscoleastic simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

'''
from .BabelIntegrationBASE import (RUN_SIM_BASE, 
                            BabelFTD_Simulations_BASE,
                            SimulationConditionsBASE,
                            Material)
import numpy as np
from sys import platform
import os
from stl import mesh
import scipy
from trimesh import creation 
import matplotlib.pyplot as plt
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple
from .H317 import GenerateH317Tx
import nibabel
    
def CreateCircularCoverage(DiameterFocalBeam=1.5e-3,DiameterCoverage=10e-3):
    RadialL=np.arange(DiameterFocalBeam,DiameterCoverage/2,DiameterFocalBeam)
    ListPoints=[[1e-6,0.0]] #center , and we do a trick to be sure all points gets the same treatment below (just make one coordinate different to 0 but very small)
    nEven=0
    for r in RadialL:
        Perimeter=np.pi*r*2
        nSteps=int(Perimeter/DiameterFocalBeam)
        theta=np.arange(nSteps)*np.pi*2/nSteps
        if nEven%2==0:
            theta+=(theta[1]-theta[0])/2
        nEven+=1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        xxyy=np.vstack((x,y)).T
        ListPoints+=xxyy.tolist()
    ListPoints=np.array(ListPoints)
  
#    plt.figure()
#    plt.plot(ListPoints[:,0],ListPoints[:,1],':+')
#    plt.gca().set_aspect('equal')
#    plt.title('Trajectory of points')
    return ListPoints

def CreateSpreadFocus(DiameterFocalBeam=1.5e-3):
    BaseTriangle =  DiameterFocalBeam/2
    HeightTriangle = np.sin(np.pi/3)*DiameterFocalBeam
    ListPoints = [[0,HeightTriangle/2]]
    ListPoints += [[BaseTriangle,-HeightTriangle/2]]
    ListPoints += [[-BaseTriangle,-HeightTriangle/2]]
    ListPoints=np.array(ListPoints)
#    plt.figure()
#    plt.plot(ListPoints[:,0]*1e3,ListPoints[:,1]*1e3,':+')
#    plt.gca().set_aspect('equal')
#    plt.title('Trajectory of points')
    return ListPoints
    

class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    DistanceConeToFocus=self._DistanceConeToFocus,
                                     **kargs)
    def RunCases(self,
                    XSteering=0.0,
                    YSteering=0.0,
                    ZSteering=0.0,
                    RotationZ=0.0,
                    DistanceConeToFocus=27e-3,
                    MultiPoint=None,
                    **kargs):
        self._RotationZ=RotationZ
        self._DistanceConeToFocus=DistanceConeToFocus
        if MultiPoint is None:
            self._XSteering=XSteering
            self._YSteering=YSteering
            self._ZSteering=ZSteering
            ExtraAdjustX = [XSteering]
            ExtraAdjustY = [YSteering]
            return super().RunCases(ExtraAdjustX=ExtraAdjustX,
                                    ExtraAdjustY=ExtraAdjustY,
                                    **kargs)
        else:
            #we need to expand accordingly to all points
            ExtraAdjustX=[]
            ExtraAdjustY=[]
            for entry in MultiPoint:
                ExtraAdjustX.append(entry['X']+XSteering)
                ExtraAdjustY.append(entry['Y']+YSteering)
            fnames=[]
            for entry in MultiPoint:
                newextrasufffix="_Steer_X_%2.1f_Y_%2.1f_Z_%2.1f_" % (entry['X']*1e3,entry['Y']*1e3,entry['Z']*1e3)
                self._XSteering=entry['X']+XSteering
                self._YSteering=entry['Y']+YSteering
                self._ZSteering=entry['Z']+ZSteering
                fnames+=super().RunCases(extrasuffix=newextrasufffix,
                                         ExtraAdjustX=ExtraAdjustX,
                                         ExtraAdjustY=ExtraAdjustY,
                                         **kargs)     
            
        return fnames

##########################################

class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,
                 XSteering=0.0,
                 YSteering=0.0,
                 ZSteering=0.0,
                 RotationZ=0.0,
                 DistanceConeToFocus=27e-3,
                 **kargs):
        
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._DistanceConeToFocus=DistanceConeToFocus
        self._RotationZ=RotationZ
        super().__init__(**kargs)

    def CreateSimConditions(self,**kargs):
        return SimulationConditions(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    DistanceConeToFocus=self._DistanceConeToFocus,
                                    RotationZ=self._RotationZ,
                                    Aperture=0.16, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                                    FocalLength=135e-3,
                                    **kargs)

    def AdjustMechanicalSettings(self,SkullMaskDataOrig,voxelS):
        Target=np.array(np.where(SkullMaskDataOrig==5.0)).flatten()
        LineSight=SkullMaskDataOrig[Target[0],Target[1],:]
        Distance=(Target[2]-np.where(LineSight>0)[0][0])*voxelS[2]
        print('*'*20+'\n'+'Distance to target from skin (mm)=',Distance*1e3)
        print('*'*20+'\n')
        self._TxMechanicalAdjustmentZ=   self._DistanceConeToFocus - Distance
        if self._ZSteering > 0:
            print('Adjust extra depth for cone with ',self._ZSteering*1e3)
            self._ExtraDepthAdjust = self._ZSteering

        print('*'*20+'\n'+'Overwriting  TxMechanicalAdjustmentZ=',self._TxMechanicalAdjustmentZ*1e3)
        print('*'*20+'\n')

    def GenerateSTLTx(self,prefix):
        #we also export the STL of the Tx for display in Brainsight or 3D slicer
        TxVert=self._SIM_SETTINGS._TxOrig['VertDisplay'].T.copy()
        TxVert/=self._SIM_SETTINGS.SpatialStep
        TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])
        affine=self._SkullMask.affine
        
        LocSpot=np.array(np.where(self._SkullMask.get_fdata()==5.0)).flatten()

        TxVert[2,:]=-TxVert[2,:]
        TxVert[0,:]+=LocSpot[0]
        TxVert[1,:]+=LocSpot[1]
        TxVert[2,:]+=LocSpot[2]+self._SIM_SETTINGS._OrigFocalLength/self._SIM_SETTINGS.SpatialStep

        TxVert=np.dot(affine,TxVert)

        TxStl = mesh.Mesh(np.zeros(self._SIM_SETTINGS._TxOrig['FaceDisplay'].shape[0]*2, dtype=mesh.Mesh.dtype))

        TxVert=TxVert.T[:,:3]
        for i, f in enumerate(self._SIM_SETTINGS._TxOrig['FaceDisplay']):
            TxStl.vectors[i*2][0] = TxVert[f[0],:]
            TxStl.vectors[i*2][1] = TxVert[f[1],:]
            TxStl.vectors[i*2][2] = TxVert[f[3],:]

            TxStl.vectors[i*2+1][0] = TxVert[f[1],:]
            TxStl.vectors[i*2+1][1] = TxVert[f[2],:]
            TxStl.vectors[i*2+1][2] = TxVert[f[3],:]
        
        bdir=os.path.dirname(self._MASKFNAME)
        TxStl.save(bdir+os.sep+prefix+'Tx.stl')
        
        TransformationCone=np.eye(4)
        TransformationCone[2,2]=-1
        OrientVec=np.array([0,0,1]).reshape((1,3))
        TransformationCone[0,3]=LocSpot[0]
        TransformationCone[1,3]=LocSpot[1]
        RadCone=self._SIM_SETTINGS._OrigAperture/self._SIM_SETTINGS.SpatialStep/2
        HeightCone=self._SIM_SETTINGS._FocalLength/self._SIM_SETTINGS._FactorEnlarge/self._SIM_SETTINGS.SpatialStep
        HeightCone=np.sqrt(HeightCone**2-RadCone**2)
        TransformationCone[2,3]=LocSpot[2]+HeightCone - self._SIM_SETTINGS._TxMechanicalAdjustmentZ/self._SIM_SETTINGS.SpatialStep
        Cone=creation.cone(RadCone,HeightCone,transform=TransformationCone)
        Cone.apply_transform(affine)
        #we save the final cone profile
        Cone.export(bdir+os.sep+prefix+'_Cone.stl')
        

    def AddSaveDataSim(self,DataForSim):
        DataForSim['XSteering']=self._XSteering
        DataForSim['YSteering']=self._YSteering
        DataForSim['ZSteering']=self._ZSteering
        DataForSim['RotationZ']=self._RotationZ
        DataForSim['bDoRefocusing']=self._bDoRefocusing
        DataForSim['DistanceConeToFocus']=self._DistanceConeToFocus
        DataForSim['BasePhasedArrayProgrammingRefocusing']=self._SIM_SETTINGS.BasePhasedArrayProgrammingRefocusing
        DataForSim['BasePhasedArrayProgramming']=self._SIM_SETTINGS.BasePhasedArrayProgramming
    
class SimulationConditions(SimulationConditionsBASE):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,FactorEnlarge = 1.0, #putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
                      Aperture=0.16, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                      FocalLength=135e-3,
                      XSteering=0.0, #lateral steering
                      YSteering=0.0,
                      ZSteering=0.0,
                      RotationZ=0.0,#rotation of Tx over Z axis
                      DistanceConeToFocus=0.0,
                      **kargs):
        super().__init__(Aperture=Aperture*FactorEnlarge,FocalLength=FocalLength*FactorEnlarge,**kargs)
        self._FactorEnlarge=FactorEnlarge
        self._OrigAperture=Aperture
        self._OrigFocalLength=FocalLength
        self._Aperture=Aperture*FactorEnlarge
        self._FocalLength=FocalLength*FactorEnlarge
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._DistanceConeToFocus=DistanceConeToFocus
        self._RotationZ=RotationZ

    def GenTransducerGeom(self):
        self._Tx=GenerateH317Tx(Frequency=self._Frequency,RotationZ=self._RotationZ,FactorEnlarge=self._FactorEnlarge)
        self._TxOrig=GenerateH317Tx(Frequency=self._Frequency,RotationZ=self._RotationZ)
        
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elements
        self.GenTransducerGeom()
        #We replicate as in the GUI as need to account for water pixels there in calculations where to truly put the Tx
        TargetLocation =np.array(np.where(self._SkullMaskDataOrig==5.0)).flatten()
        LineOfSight=self._SkullMaskDataOrig[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()*self._SkullMaskNii.header.get_zooms()[2]/1e3
        print('StartSkin',StartSkin)
        
        if self._bDisplay:
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = Axes3D(fig)

            ax.add_collection3d(Poly3DCollection(self._Tx['VertDisplay'][self._Tx['FaceDisplay']]*1e3)) #we plot the units in mm
            #3D display are not so smart as regular 2D, so we have to adjust manually the limits so we can see the figure correctly
            ax.set_xlim(-self._Tx['Aperture']/2*1e3-5,self._Tx['Aperture']/2*1e3+5)
            ax.set_ylim(-self._Tx['Aperture']/2*1e3-5,self._Tx['Aperture']/2*1e3+5)
            ax.set_zlim(0,135)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.show()
        
        for k in ['center','elemcenter','VertDisplay']:
            self._Tx[k][:,0]+=self._TxMechanicalAdjustmentX
            self._Tx[k][:,1]+=self._TxMechanicalAdjustmentY
            self._Tx[k][:,2]+=self._TxMechanicalAdjustmentZ-StartSkin

        Correction=0.0
        while np.max(self._Tx['center'][:,2])>=self._ZDim[self._ZSourceLocation]:
            #at the most, we could be too deep only a fraction of a single voxel, in such case we just move the Tx back a single step
            for k in ['center','VertDisplay','elemcenter']:
                self._Tx[k][:,2]-=self._SkullMaskNii.header.get_zooms()[2]/1e3
            Correction+=self._SkullMaskNii.header.get_zooms()[2]/1e3
        if Correction>0:
            print('Warning: Need to apply correction to reposition Tx for',Correction)
        #if yet we are not there, we need to stop
        if np.max(self._Tx['center'][:,2])>self._ZDim[self._ZSourceLocation]:
            print("np.max(self._Tx['center'][:,2]),self._ZDim[self._ZSourceLocation]",np.max(self._Tx['center'][:,2]),self._ZDim[self._ZSourceLocation])
            raise RuntimeError("The Tx limit in Z is below the location of the layer for source location for forward propagation.")
      
        #we apply an homogeneous pressure 
       
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        #we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming=np.zeros(self._Tx['NumberElems'],np.complex64)
        self.BasePhasedArrayProgrammingRefocusing=np.zeros(self._Tx['NumberElems'],np.complex64)
        
        if self._XSteering!=0.0 or self._YSteering!=0.0 or self._ZSteering!=0.0:
            print('Running Steering')
            ds=np.ones((1))*self._SpatialStep**2
        
        
            #we apply an homogeneous pressure 
            u0=np.zeros((1),np.complex64)
            u0[0]=1+0j
            center=np.zeros((1,3),np.float32)
            center[0,0]=self._XDim[self._FocalSpotLocation[0]]+self._TxMechanicalAdjustmentX+self._XSteering
            center[0,1]=self._YDim[self._FocalSpotLocation[1]]+self._TxMechanicalAdjustmentY+self._YSteering
            center[0,2]=self._ZDim[self._FocalSpotLocation[2]]+self._TxMechanicalAdjustmentZ+self._ZSteering

            print('center',center)
            
            u2back=ForwardSimple(cwvnb_extlay,center,ds.astype(np.float32),u0,self._Tx['elemcenter'].astype(np.float32),deviceMetal=deviceName)
            u0=np.zeros((self._Tx['center'].shape[0],1),np.complex64)
            nBase=0
            for n in range(self._Tx['NumberElems']):
                phi=np.angle(np.conjugate(u2back[n]))
                self.BasePhasedArrayProgramming[n]=np.conjugate(u2back[n])
                u0[nBase:nBase+self._Tx['elemdims']]=(self._SourceAmpPa*np.exp(1j*phi)).astype(np.complex64)
                nBase+=self._Tx['elemdims']
        else:
             u0=(np.ones((self._Tx['center'].shape[0],1),np.float32)+ 1j*np.zeros((self._Tx['center'].shape[0],1),np.float32))*self._SourceAmpPa
        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        yp,xp,zp=np.meshgrid(self._YDim,self._XDim,self._ZDim)
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._Tx['center'].astype(np.float32),self._Tx['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)
        
        self._u2RayleighField=u2
        self._SourceMapRayleigh=u2[:,:,self._ZSourceLocation].copy()
        self._SourceMapRayleigh[:self._PMLThickness,:]=0
        self._SourceMapRayleigh[-self._PMLThickness:,:]=0
        self._SourceMapRayleigh[:,:self._PMLThickness]=0
        self._SourceMapRayleigh[:,-self._PMLThickness:]=0
        
        if self._bDisplay:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(np.abs(self._SourceMapRayleigh)/1e6,
                       vmin=np.abs(self._SourceMapRayleigh[RegionMap]).min()/1e6,cmap=plt.cm.jet)
            plt.colorbar()
            plt.title('Incident map to be forwarded propagated (MPa)')

            plt.subplot(1,2,2)
        
            plt.imshow((np.abs(u2[self._FocalSpotLocation[0],:,:]).T+
                                    ((self._MaterialMap[self._FocalSpotLocation[0],:,:].T>=3).astype(float))*
                                    np.abs(u2[self._FocalSpotLocation[0],:,:]).max()/10)/1e6,
                                    extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()],
                                    cmap=plt.cm.jet)
            plt.colorbar()
            plt.title('Acoustic field with Rayleigh with skull and brain (MPa)')

          
        
    def CreateSources(self,ramp_length=4):
        #we create the list of functions sources taken from the Rayliegh incident field
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)
        
        self._SourceMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        LocZ=self._ZSourceLocation
        
        SourceMaskIND=np.where(np.abs(self._SourceMapRayleigh)>0)
        SourceMask=np.zeros((self._N1,self._N2),np.uint32)
        
        RefI= int((SourceMaskIND[0].max()-SourceMaskIND[0].min())/2)+SourceMaskIND[0].min()
        RefJ= int((SourceMaskIND[1].max()-SourceMaskIND[1].min())/2)+SourceMaskIND[1].min()
        AngRef=np.angle(self._SourceMapRayleigh[RefI,RefJ])
        PulseSource = np.zeros((np.sum(np.abs(self._SourceMapRayleigh)>0),TimeVectorSource.shape[0]))
        nSource=1                       
        for i,j in zip(SourceMaskIND[0],SourceMaskIND[1]):
            SourceMask[i,j]=nSource
            u0=self._SourceMapRayleigh[i,j]
            #we recover amplitude and phase from Rayleigh field
            PulseSource[nSource-1,:] = np.abs(u0) *np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(u0))
            PulseSource[nSource-1,:int(ramp_length_points)]*=ramp
            nSource+=1
        self._SourceMap[:,:,LocZ]=SourceMask 
            
        self._PulseSource=PulseSource
        
        ## Now we create the sources for back propagation
        
        self._PunctualSource=np.sin(2*np.pi*self._Frequency*TimeVectorSource).reshape(1,len(TimeVectorSource))
        self._SourceMapPunctual=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        LocForRefocusing=self._FocalSpotLocation.copy()
        LocForRefocusing[0]+=int(np.round(self._XSteering/self._SpatialStep))
        LocForRefocusing[1]+=int(np.round(self._YSteering/self._SpatialStep))
        LocForRefocusing[2]+=int(np.round(self._ZSteering/self._SpatialStep))
        self._SourceMapPunctual[LocForRefocusing[0],LocForRefocusing[1],LocForRefocusing[2]]=1
        

        if self._bDisplay:
            plt.figure(figsize=(12,4))
            for n in range(1,4):
                plt.plot(TimeVectorSource*1e6,PulseSource[int(PulseSource.shape[0]/4)*n,:])
                plt.title('CW signal, example %i' %(n))
                
            plt.xlim(0,50)
                
            plt.figure(figsize=(5,4))
            plt.imshow(self._SourceMap[:,:,LocZ])
            plt.title('source map - source ids')


    def BackPropagationRayleigh(self,deviceName='6800'):
        assert(np.all(np.array(self._SourceMapRayleigh.shape)==np.array(self._PressMapFourierBack.shape)))
        SelRegRayleigh=np.abs(self._SourceMapRayleigh)>0
        ypp,xpp=np.meshgrid(self._YDim,self._XDim)
        ypp=ypp[SelRegRayleigh]
        xpp=xpp[SelRegRayleigh]
        center=np.zeros((ypp.size,3),np.float32)
        center[:,0]=xpp.flatten()
        center[:,1]=ypp.flatten()
        center[:,2]=self._ZDim[self._ZSourceLocation]
            
        ds=np.ones((center.shape[0]))*self._SpatialStep**2
        
        #we apply an homogeneous pressure 
        u0=self._PressMapFourierBack[SelRegRayleigh]
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)

        u2back=ForwardSimple(cwvnb_extlay,center.astype(np.float32),ds.astype(np.float32),u0,self._Tx['elemcenter'].astype(np.float32),deviceMetal=deviceName)
        
        #now we calculate forward back
        
        u0=np.zeros((self._Tx['center'].shape[0],1),np.complex64)
        nBase=0
        for n in range(self._Tx['NumberElems']):
            phi=np.angle(np.conjugate(u2back[n]))
            self.BasePhasedArrayProgrammingRefocusing[n]=np.conjugate(u2back[n])
            u0[nBase:nBase+self._Tx['elemdims']]=(self._SourceAmpPa*np.exp(1j*phi)).astype(np.complex64)
            nBase+=self._Tx['elemdims']

        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        yp,xp,zp=np.meshgrid(self._YDim,self._XDim,self._ZDim)
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._Tx['center'].astype(np.float32),self._Tx['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)
        self._SourceMapRayleighRefocus=u2[:,:,self._ZSourceLocation].copy()
        self._SourceMapRayleighRefocus[:self._PMLThickness,:]=0
        self._SourceMapRayleighRefocus[-self._PMLThickness:,:]=0
        self._SourceMapRayleighRefocus[:,:self._PMLThickness]=0
        self._SourceMapRayleighRefocus[:,-self._PMLThickness:]=0
        
        
    def CreateSourcesRefocus(self,ramp_length=4):
        #we create the list of functions sources taken from the Rayliegh incident field
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)
        
        LocZ=self._ZSourceLocation
        
        SourceMaskIND=np.where(np.abs(self._SourceMapRayleigh)>0)
           
        RefI= int((SourceMaskIND[0].max()-SourceMaskIND[0].min())/2)+SourceMaskIND[0].min()
        RefJ= int((SourceMaskIND[1].max()-SourceMaskIND[1].min())/2)+SourceMaskIND[1].min()
        AngRef=np.angle(self._SourceMapRayleighRefocus[RefI,RefJ])
        PulseSource = np.zeros((np.sum(np.abs(self._SourceMapRayleighRefocus)>0),TimeVectorSource.shape[0]))
        nSource=1                       
        for i,j in zip(SourceMaskIND[0],SourceMaskIND[1]):
            u0=self._SourceMapRayleighRefocus[i,j]
            #we recover amplitude and phase from Rayleigh field
            PulseSource[nSource-1,:] = np.abs(u0) *np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(u0))
            PulseSource[nSource-1,:int(ramp_length_points)]*=ramp
            nSource+=1
            
        self._PulseSourceRefocus=PulseSource
         
        
                
        
