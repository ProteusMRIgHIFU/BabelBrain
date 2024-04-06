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
from scipy.io import loadmat
from scipy.interpolate import interpn
from trimesh import creation 
import matplotlib.pyplot as plt
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple
from BabelViscoFDTD.PropagationModel import PropagationModel
import nibabel
from linetimer import CodeTimer

PITCH = 3.08e-3 
KERF = 0.5e-3
FREQ=300e3
APERTURE = 0.058
minPPPArray=60.08
DimensionElem = PITCH-KERF
ZDistance=-1.2e-3 #distance from Tx elements to outplane


def computeREMOPDGeometry():
    TxPos=loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)),'REMOPD_ElementPosition.mat'))['REMOPD_ElementPosition']
    return TxPos

def GenerateSingleElem(PPW=12.0):
    #60.08 PPW produces close to integer steps for both pitch and kerf
    
    Tx = {}

    step = 1500/FREQ/PPW
    hstep=step/2.0
    latSteps=int(np.round(DimensionElem/step))
    step=DimensionElem/latSteps

    centersX= np.arange(latSteps)*step
    centersX-=np.mean(centersX)

    SingElem = np.zeros((latSteps**2,3))
    N = np.zeros((SingElem.shape[0],3))
    N[:,2]=1
    ds = np.ones((SingElem.shape[0],1))*step**2

    VertDisplay=  np.zeros((SingElem.shape[0]*4,3))
    FaceDisplay= np.arange(SingElem.shape[0]*4,dtype=int).reshape((SingElem.shape[0],4))

    XX,YY=np.meshgrid(centersX,centersX)
    SingElem[:,0]=XX.flatten()
    SingElem[:,1]=YY.flatten()
    SingElem[:,2]=ZDistance

    VertDisplay[0::4,0]=SingElem[:,0]-hstep
    VertDisplay[0::4,1]=SingElem[:,1]-hstep
    
    VertDisplay[1::4,0]=SingElem[:,0]+hstep
    VertDisplay[1::4,1]=SingElem[:,1]-hstep

    VertDisplay[2::4,0]=SingElem[:,0]+hstep
    VertDisplay[2::4,1]=SingElem[:,1]+hstep

    VertDisplay[3::4,0]=SingElem[:,0]-hstep
    VertDisplay[3::4,1]=SingElem[:,1]+hstep

    VertDisplay[:,2]= ZDistance
   
    Tx['center'] = SingElem 
    Tx['ds'] = ds
    Tx['normal'] = N
    Tx['VertDisplay'] = VertDisplay 
    Tx['FaceDisplay'] = FaceDisplay 
    return Tx
    

def GenerateREMOPDTx(subsetLimit=128,RotationZ=0.0):
   
    #%This is the indiv tx element
    TxElem=GenerateSingleElem()


    transLoc = computeREMOPDGeometry()

    rotateMatrixZ = np.array([[-np.cos(RotationZ),np.sin(RotationZ),0],
                              [-np.sin(RotationZ),-np.cos(RotationZ),0],[0,0,1]])
            

    ALLConfigs={'Total':{},'Sector1':{},'Sector2':{}}
    for k in ALLConfigs:
        ALLConfigs[k]['center'] = np.zeros((0,3))
        if k == 'Total':
            ALLConfigs[k]['elemcenter'] = np.zeros((256,3))
        else:
            ALLConfigs[k]['elemcenter'] = np.zeros((subsetLimit,3))
        ALLConfigs[k]['ds'] = np.zeros((0,1))
        ALLConfigs[k]['normal'] = np.zeros((0,3))
        ALLConfigs[k]['elemdims']=TxElem['ds'].size
        ALLConfigs[k]['NumberElems']=ALLConfigs[k]['elemcenter'].shape[0]
        ALLConfigs[k]['VertDisplay'] = np.zeros((0,3))
        ALLConfigs[k]['FaceDisplay'] = np.zeros((0,4),np.int64)

    for n in range(transLoc.shape[0]):
        if n <subsetLimit: #first sector
            selc = ['Total','Sector1']
        else:
            selc = ['Total','Sector2']
        for k in selc:
            Tx=ALLConfigs[k]
            prevFaceLength=Tx['VertDisplay'].shape[0]
        
            center=TxElem['center']+transLoc[n,:]

            center=(rotateMatrixZ@center.T).T
            
            if k == 'Total':
                Tx['elemcenter'][n,:]=np.mean(center,axis=0) 
            else:
                Tx['elemcenter'][n%subsetLimit,:]=np.mean(center,axis=0)  
            
            normal=TxElem['normal'].copy()
            
            VertDisplay=TxElem['VertDisplay']+transLoc[n,:]

            VertDisplay=(rotateMatrixZ@VertDisplay.T).T
           
            Tx['center']=np.vstack((Tx['center'],center))
            Tx['ds'] =np.vstack((Tx['ds'],TxElem['ds']))
            Tx['normal'] =np.vstack((Tx['normal'],normal))
            Tx['VertDisplay'] =np.vstack((Tx['VertDisplay'],VertDisplay))
            Tx['FaceDisplay']=np.vstack((Tx['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        
    print('Aperture dimensions (x,y) =',ALLConfigs['Total']['VertDisplay'][:,0].max()-ALLConfigs['Total']['VertDisplay'][:,0].min(),
                                        ALLConfigs['Total']['VertDisplay'][:,1].max()-ALLConfigs['Total']['VertDisplay'][:,1].min())
    ALLConfigs['Aperture']=np.max([ALLConfigs['Total']['VertDisplay'][:,0].max()-ALLConfigs['Total']['VertDisplay'][:,0].min(),
                                        ALLConfigs['Total']['VertDisplay'][:,1].max()-ALLConfigs['Total']['VertDisplay'][:,1].min()]);
    return ALLConfigs

class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    TxSet=self._TxSet,
                                    **kargs)
    def RunCases(self,
                    XSteering=0.0,
                    YSteering=0.0,
                    ZSteering=60.0e-3,
                    RotationZ=0.0,
                    TxSet='Total', #Total selects all the 256 elements, Sector1 the central 128 elements, and Sector2 the external 128
                    **kargs):
        self._RotationZ=RotationZ
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._TxSet=TxSet
        
        return super().RunCases(**kargs)
        
##########################################

class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,
                 XSteering=0.0,
                 YSteering=0.0,
                 ZSteering=0.0,
                 RotationZ=0.0,
                 TxSet='Total', #Total selects all the 256 elements, Sector1 the central 128 elements, and Sector2 the external 128
                 **kargs):
        
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._RotationZ=RotationZ
        self._TxSet=TxSet
        super().__init__(**kargs)

    def CreateSimConditions(self,**kargs):
        return SimulationConditions(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    TxSet=self._TxSet,
                                    FocalLength=0.0,
                                    Aperture=APERTURE, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                                    **kargs)

    def GenerateSTLTx(self,prefix):
        #we also export the STL of the Tx for display in Brainsight or 3D slicer
        TxVert=self._SIM_SETTINGS._TxREMOPD['VertDisplay'].T.copy()
        TxVert/=self._SIM_SETTINGS.SpatialStep
        TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])
        affine=self._SkullMask.affine
        
        LocSpot=np.array(np.where(self._SkullMask.get_fdata()==5.0)).flatten()

        TxVert[2,:]=-TxVert[2,:]
        TxVert[0,:]+=LocSpot[0]+int(np.round(self._TxMechanicalAdjustmentX/self._SIM_SETTINGS.SpatialStep))
        TxVert[1,:]+=LocSpot[1]+int(np.round(self._TxMechanicalAdjustmentY/self._SIM_SETTINGS.SpatialStep))
        TxVert[2,:]+=LocSpot[2]+int(np.round((self._ZSteering-self._TxMechanicalAdjustmentZ)/self._SIM_SETTINGS.SpatialStep))

        TxVert=np.dot(affine,TxVert)

        TxStl = mesh.Mesh(np.zeros(self._SIM_SETTINGS._TxREMOPD['FaceDisplay'].shape[0]*2, dtype=mesh.Mesh.dtype))

        TxVert=TxVert.T[:,:3]
        for i, f in enumerate(self._SIM_SETTINGS._TxREMOPD['FaceDisplay']):
            TxStl.vectors[i*2][0] = TxVert[f[0],:]
            TxStl.vectors[i*2][1] = TxVert[f[1],:]
            TxStl.vectors[i*2][2] = TxVert[f[3],:]

            TxStl.vectors[i*2+1][0] = TxVert[f[1],:]
            TxStl.vectors[i*2+1][1] = TxVert[f[2],:]
            TxStl.vectors[i*2+1][2] = TxVert[f[3],:]
        
        bdir=os.path.dirname(self._MASKFNAME)
        TxStl.save(bdir+os.sep+prefix+'Tx.stl')
        
            # TransformationCone=np.eye(4)
            # TransformationCone[2,2]=-1
            # OrientVec=np.array([0,0,1]).reshape((1,3))
            # TransformationCone[0,3]=LocSpot[0]
            # TransformationCone[1,3]=LocSpot[1]
            # RadCone=self._SIM_SETTINGS._Aperture/self._SIM_SETTINGS.SpatialStep/2
            # HeightCone=self._SIM_SETTINGS._ZSteering/self._SIM_SETTINGS.SpatialStep
            # HeightCone=np.sqrt(HeightCone**2-RadCone**2)
            # TransformationCone[2,3]=LocSpot[2]+HeightCone - self._SIM_SETTINGS._TxMechanicalAdjustmentZ/self._SIM_SETTINGS.SpatialStep
            # Cone=creation.cone(RadCone,HeightCone,transform=TransformationCone)
            # Cone.apply_transform(affine)
            # #we save the final cone profile
            # Cone.export(bdir+os.sep+prefix+'_Cone.stl')
        

    def AddSaveDataSim(self,DataForSim):
        DataForSim['XSteering']=self._XSteering
        DataForSim['YSteering']=self._YSteering
        DataForSim['ZSteering']=self._ZSteering
        DataForSim['RotationZ']=self._RotationZ
        DataForSim['TxSet']=self._TxSet
        DataForSim['bDoRefocusing']=self._bDoRefocusing
        DataForSim['BasePhasedArrayProgrammingRefocusing']=self._SIM_SETTINGS.BasePhasedArrayProgrammingRefocusing
        DataForSim['BasePhasedArrayProgramming']=self._SIM_SETTINGS.BasePhasedArrayProgramming
    
class SimulationConditions(SimulationConditionsBASE):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,Aperture=APERTURE, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                      FocalLength=0.0,
                      XSteering=0.0, #lateral steering
                      YSteering=0.0,
                      ZSteering=0.0,
                      RotationZ=0.0,#rotation of Tx over Z axis
                      TxSet='Total', #Total selects all the 256 elements, Sector1 the central 128 elements, and Sector2 the external 128
                      **kargs):
        super().__init__(Aperture=Aperture,FocalLength=FocalLength,
                         ZTxCorrecton=-ZDistance, #this will put the required water space in the simulation domain
                         **kargs)
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._RotationZ=RotationZ
        self._TxSet = TxSet
        
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elements
        # and we select the set based on input
        self._TxREMOPD=GenerateREMOPDTx(RotationZ=self._RotationZ)[self._TxSet]
        
        
        #We replicate as in the GUI as need to account for water pixels there in calculations where to truly put the Tx
        TargetLocation =np.array(np.where(self._SkullMaskDataOrig==5.0)).flatten()
        LineOfSight=self._SkullMaskDataOrig[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()*self._SkullMaskNii.header.get_zooms()[2]/1e3
        print('StartSkin',StartSkin)
        
        if self._TxMechanicalAdjustmentZ <0:
            zCorrec= self._TxMechanicalAdjustmentZ
        else:
            zCorrec=0.0
        
        for k in ['center','elemcenter','VertDisplay']:
            self._TxREMOPD[k][:,0]+=self._TxMechanicalAdjustmentX
            self._TxREMOPD[k][:,1]+=self._TxMechanicalAdjustmentY
            self._TxREMOPD[k][:,2]=self._ZDim[self._ZSourceLocation]-self._SkullMaskNii.header.get_zooms()[2]/1e3+zCorrec
            
        Correction=0.0
        while np.max(self._TxREMOPD['center'][:,2])>=self._ZDim[self._ZSourceLocation]:
            #at the most, we could be too deep only a fraction of a single voxel, in such case we just move the Tx back a single step
            for Tx in [self._TxREMOPD]:
                for k in ['center','VertDisplay','elemcenter']:
                    Tx[k][:,2]-=self._SkullMaskNii.header.get_zooms()[2]/1e3
            Correction+=self._SkullMaskNii.header.get_zooms()[2]/1e3
        if Correction>0:
            print('Warning: Need to apply correction to reposition Tx for',Correction)
        #if yet we are not there, we need to stop
        if np.max(self._TxREMOPD['center'][:,2])>self._ZDim[self._ZSourceLocation]:
            print("np.max(self._TxREMOPD['center'][:,2]),self._ZDim[self._ZSourceLocation]",np.max(self._TxREMOPD['center'][:,2]),self._ZDim[self._ZSourceLocation])
            raise RuntimeError("The Tx limit in Z is below the location of the layer for source location for forward propagation.")
      
        
        print("self._TxREMOPD['center'].min(axis=0)",self._TxREMOPD['center'].min(axis=0))
        print("self._TxREMOPD['elemcenter'].min(axis=0)",self._TxREMOPD['elemcenter'].min(axis=0))
      
        #we apply an homogeneous pressure 
       
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        #we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming=np.zeros(self._TxREMOPD['NumberElems'],np.complex64)
        self.BasePhasedArrayProgrammingRefocusing=np.zeros(self._TxREMOPD['NumberElems'],np.complex64)
        
        if self._XSteering!=0.0 or self._YSteering!=0.0 or self._ZSteering!=0.0:
            print('Running Steering')
            ds=np.ones((1))*self._SpatialStep**2
        
        
            #we apply an homogeneous pressure 
            u0=np.zeros((1),np.complex64)
            u0[0]=1+0j
            center=np.zeros((1,3),np.float32)
            center[0,0]=self._XDim[self._FocalSpotLocation[0]]+self._TxMechanicalAdjustmentX+self._XSteering
            center[0,1]=self._YDim[self._FocalSpotLocation[1]]+self._TxMechanicalAdjustmentY+self._YSteering
            center[0,2]=self._ZDim[self._ZSourceLocation]+self._ZSteering+zCorrec

            print('center',center,np.mean(self._TxREMOPD['elemcenter'][:,2]))
            
            u2back=ForwardSimple(cwvnb_extlay,center,ds.astype(np.float32),u0,self._TxREMOPD['elemcenter'].astype(np.float32),deviceMetal=deviceName)
            u0=np.zeros((self._TxREMOPD['center'].shape[0],1),np.complex64)
            nBase=0
            for n in range(self._TxREMOPD['NumberElems']):
                phi=np.angle(np.conjugate(u2back[n]))
                self.BasePhasedArrayProgramming[n]=np.conjugate(u2back[n])
                u0[nBase:nBase+self._TxREMOPD['elemdims']]=(self._SourceAmpPa*np.exp(1j*phi)).astype(np.complex64)
                nBase+=self._TxREMOPD['elemdims']

            
        else:
             u0=(np.ones((self._TxREMOPD['center'].shape[0],1),np.float32)+ 1j*np.zeros((self._TxREMOPD['center'].shape[0],1),np.float32))*self._SourceAmpPa
             
        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        xp,yp,zp=np.meshgrid(self._XDim,self._YDim,self._ZDim,indexing='ij')

        print('ZDim[self._ZSourceLocation]',self._ZDim[self._ZSourceLocation])
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._TxREMOPD['center'].astype(np.float32),
                         self._TxREMOPD['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)
        
        self._u2RayleighField=u2

        self._SourceMapRayleigh=u2[:,:,self._ZSourceLocation].copy()

        self._SourceMapRayleigh[:self._PMLThickness,:]=0
        self._SourceMapRayleigh[-self._PMLThickness:,:]=0
        self._SourceMapRayleigh[:,:self._PMLThickness]=0
        self._SourceMapRayleigh[:,-self._PMLThickness:]=0

          
        
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
        LocForRefocusing[2]=0.0
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
        center[:,2]=self._TxMechanicalAdjustmentZ
            
        ds=np.ones((center.shape[0]))*self._SpatialStep**2


        #we apply an homogeneous pressure 
        u0=self._PressMapFourierBack[SelRegRayleigh]
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)

        u2back=ForwardSimple(cwvnb_extlay,center.astype(np.float32),ds.astype(np.float32),
                             u0,self._TxREMOPD['elemcenter'].astype(np.float32),deviceMetal=deviceName)
        
        #now we calculate forward back
        
        u0=np.zeros((self._TxREMOPD['center'].shape[0],1),np.complex64)
        nBase=0
        for n in range(self._TxREMOPD['NumberElems']):
            phi=np.angle(np.conjugate(u2back[n]))
            self.BasePhasedArrayProgrammingRefocusing[n]=np.conjugate(u2back[n])
            u0[nBase:nBase+self._TxREMOPD['elemdims']]=(self._SourceAmpPa*np.exp(1j*phi)).astype(np.complex64)
            nBase+=self._TxREMOPD['elemdims']

        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        ZDim=self._ZDim-self._ZDim[self._ZSourceLocation]+self._TxMechanicalAdjustmentZ
        
        xp,yp,zp=np.meshgrid(self._XDim,self._YDim,ZDim,indexing='ij')
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._TxREMOPD['center'].astype(np.float32),self._TxREMOPD['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
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