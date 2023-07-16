'''
Pipeline to execute viscoleastic simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - May 19, 2022

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
import trimesh
import matplotlib.pyplot as plt
from BabelViscoFDTD.tools.RayleighAndBHTE import GenerateFocusTx,ForwardSimple, InitCuda,InitOpenCL,SpeedofSoundWater
from scipy.interpolate import interpn
from BabelViscoFDTD.H5pySimple import ReadFromH5py
###########################################
def GenerateSurface(lstep,Diam,Foc,IntDiam=0):
    Tx = {}
    rInt=IntDiam/2
    rExt=Diam/2
    Beta1= np.arcsin(rInt/Foc)
    Beta2= np.arcsin(rExt/Foc)
    
    CenterElementBeta=(Beta2+Beta1)/2
    ElementCenter=[[np.sin(CenterElementBeta)*Foc,
                   0.0,
                   -np.cos(CenterElementBeta)*Foc]]
                   
    DBeta= Beta2-Beta1

    ArcC = DBeta*Foc

    nrstep = np.ceil(ArcC/lstep)

    BetaStep = DBeta/nrstep
    
    print(Beta1+BetaStep/2,Beta1+BetaStep*(1/2 + nrstep),BetaStep)
    BetaC = np.arange(Beta1+BetaStep/2,Beta1+BetaStep*(1/2 + nrstep),BetaStep)
    
    Ind=0

    SingElem = np.zeros((0,3))
    N = np.zeros((0,3))
    ds = np.zeros((0,1))

    VertDisplay=  np.zeros((0,3))
    FaceDisplay= np.zeros((0,4),np.int64)

    for nr in range(len(BetaC)):

        Perim = np.sin(BetaC[nr])*Foc*2*np.pi

        nAlpha = np.ceil(Perim/lstep)
        sAlpha = 2*np.pi/nAlpha

        AlphaC = np.arange(sAlpha/2,sAlpha*(1/2 + nAlpha ),sAlpha)


        SingElem=np.vstack((SingElem,np.zeros((len(AlphaC),3))))
        N  = np.vstack((N,np.zeros((len(AlphaC),3))))
        ds = np.vstack((ds,np.zeros((len(AlphaC),1))))

        VertDisplay= np.vstack((VertDisplay,np.zeros((len(AlphaC)*4,3))))
        FaceDisplay= np.vstack((FaceDisplay,np.zeros((len(AlphaC),4),np.int64)))


        zc = -np.cos(BetaC[nr])*Foc
        Rc = np.sin(BetaC[nr])*Foc

        B1 = BetaC[nr]-BetaStep/2
        B2 = BetaC[nr]+BetaStep/2
        if nr==0 and IntDiam==0.0:
            Rc1=0
        else:
            Rc1 = np.sin(B1)*Foc
        
        Rc2 = np.sin(B2)*Foc

        zc1 =-np.cos(B1)*Foc
        zc2 =-np.cos(B2)*Foc
        
        SingElem[Ind:,0] = Rc*np.cos(AlphaC)
        SingElem[Ind:,1] = Rc*np.sin(AlphaC)
        SingElem[Ind:,2] = zc
        
        A1 = AlphaC-sAlpha/2;
        A2 = AlphaC+sAlpha/2;
        ds[Ind:,0]=Foc**2 *(np.cos(B1) - np.cos(B2))*(A2-A1)
        N[Ind:,:] =SingElem[Ind:,:]/np.repeat(np.linalg.norm(SingElem[Ind:,:],axis=1).reshape((len(AlphaC),1)),3,axis=1)
        VertDisplay[Ind*4::4,0]= Rc1*np.cos(A1)
        VertDisplay[Ind*4::4,1]= Rc1*np.sin(A1)
        VertDisplay[Ind*4::4,2]= zc1

        VertDisplay[Ind*4+1::4,0]= Rc1*np.cos(A2)
        VertDisplay[Ind*4+1::4,1]= Rc1*np.sin(A2)
        VertDisplay[Ind*4+1::4,2]= zc1

        VertDisplay[Ind*4+2::4,0]= Rc2*np.cos(A1)
        VertDisplay[Ind*4+2::4,1]= Rc2*np.sin(A1)
        VertDisplay[Ind*4+2::4,2]= zc2

        VertDisplay[Ind*4+3::4,0]= Rc2*np.cos(A2)
        VertDisplay[Ind*4+3::4,1]= Rc2*np.sin(A2)
        VertDisplay[Ind*4+3::4,2]= zc2

        FaceDisplay[Ind:,0] =(Ind+np.arange(len(AlphaC)))*4
        FaceDisplay[Ind:,1] =(Ind+np.arange(len(AlphaC)))*4+1
        FaceDisplay[Ind:,2] =(Ind+np.arange(len(AlphaC)))*4+3
        FaceDisplay[Ind:,3] =(Ind+np.arange(len(AlphaC)))*4+2
        Ind+=len(AlphaC)

    Tx['center'] = SingElem 
    Tx['ds'] = ds
    Tx['normal'] = N
    Tx['VertDisplay'] = VertDisplay 
    Tx['FaceDisplay'] = FaceDisplay 
    Tx['Beta1']=np.array([[Beta1]])
    Tx['Beta2']=np.array([[Beta2]])
    Tx['elemcenter']=np.array(ElementCenter)
    Tx['elemdims']=np.array([[len(ds)]])
    return Tx

def GenerateFocusTx(f,Foc,Diam,c,PPWSurface=4):
    wavelength = c/f
    lstep = wavelength/PPWSurface

    Tx = GenerateSurface(lstep,Diam,Foc)
    return Tx


def GeneratedRingArrayTx(f,Foc,InDiameters,OutDiameters,c,PPWSurface=4):
    wavelength = c/f
    lstep = wavelength/PPWSurface
    
    bFirstRing=True
    n=0
    for ID,OD in zip(InDiameters,OutDiameters):
        TxP = GenerateSurface(lstep,OD,Foc,IntDiam=ID)
#        SaveToH5py(TxP,'Ring%i.h5'%(n))
        n+=1
        TxVert=TxP['VertDisplay']*1e3
        TxVert[:,2]+=Foc
 
            
            
        if bFirstRing:
            Tx=TxP
            bFirstRing=False
            Tx['RingFaceDisplay']=[Tx.pop('FaceDisplay')]
            Tx['RingVertDisplay']=[Tx.pop('VertDisplay')]
                
        else:
            for k in TxP:
                if k in ['FaceDisplay','VertDisplay']:
                    Tx['Ring'+k].append(TxP[k])
                else:
                    Tx[k]=np.vstack((Tx[k],TxP[k]))
                     
    return Tx

class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(ZSteering=self._ZSteering,
                                     **kargs)
    def RunCases(self,
                    ZSteering=0.0,
                    **kargs):
        self._ZSteering=ZSteering
        return super().RunCases(**kargs)


class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,
                 ZSteering=0.0,
                 **kargs):
        self._ZSteering=ZSteering
        super().__init__(**kargs)
        
    def CreateSimConditions(self,**kargs):
        return SimulationConditions(ZSteering=self._ZSteering,
                                    Aperture=33.60e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                                    FocalLength=0.0,
                                    **kargs)
    
    def GenerateSTLTx(self,prefix):
        n=1
        affine=self._SkullMask.affine
        LocSpot=np.array(np.where(self._SkullMask.get_fdata()==5.0)).flatten()
        for VertDisplay,FaceDisplay in zip(self._SIM_SETTINGS._TxRCOrig['RingVertDisplay'],
                                self._SIM_SETTINGS._TxRCOrig['RingFaceDisplay']):
            #we also export the STL of the Tx for display in Brainsight or 3D slicer
            TxVert=VertDisplay.T.copy()
            TxVert/=self._SIM_SETTINGS.SpatialStep
            TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])

            TxVert[2,:]=-TxVert[2,:]
            TxVert[0,:]+=LocSpot[0]+int(np.round(self._TxMechanicalAdjustmentX/self._SIM_SETTINGS.SpatialStep))
            TxVert[1,:]+=LocSpot[1]+int(np.round(self._TxMechanicalAdjustmentY/self._SIM_SETTINGS.SpatialStep))
            TxVert[2,:]+=LocSpot[2]+int(np.round((self._ZSteering-self._TxMechanicalAdjustmentZ)/self._SIM_SETTINGS.SpatialStep))

            TxVert=np.dot(affine,TxVert)

            TxStl = mesh.Mesh(np.zeros(FaceDisplay.shape[0]*2, dtype=mesh.Mesh.dtype))

            TxVert=TxVert.T[:,:3]
            for i, f in enumerate(FaceDisplay):
                TxStl.vectors[i*2][0] = TxVert[f[0],:]
                TxStl.vectors[i*2][1] = TxVert[f[1],:]
                TxStl.vectors[i*2][2] = TxVert[f[3],:]

                TxStl.vectors[i*2+1][0] = TxVert[f[1],:]
                TxStl.vectors[i*2+1][1] = TxVert[f[2],:]
                TxStl.vectors[i*2+1][2] = TxVert[f[3],:]

            bdir=os.path.dirname(self._MASKFNAME)
            TxStl.save(bdir+os.sep+prefix+'Tx_Ring_%i.stl' %(n))
            n+=1
        TransformationCone=np.eye(4)
        TransformationCone[2,2]=-1
        OrientVec=np.array([0,0,1]).reshape((1,3))
        TransformationCone[0,3]=LocSpot[0]+int(np.round(self._TxMechanicalAdjustmentX/self._SIM_SETTINGS.SpatialStep))
        TransformationCone[1,3]=LocSpot[1]+int(np.round(self._TxMechanicalAdjustmentY/self._SIM_SETTINGS.SpatialStep))
        RadCone=self._SIM_SETTINGS._OrigAperture/self._SIM_SETTINGS.SpatialStep/2
        HeightCone=self._ZSteering/self._SIM_SETTINGS.SpatialStep
        TransformationCone[2,3]=LocSpot[2]+HeightCone - self._SIM_SETTINGS._TxMechanicalAdjustmentZ/self._SIM_SETTINGS.SpatialStep
        Cone=creation.cone(RadCone,HeightCone,transform=TransformationCone)
        Cone.apply_transform(affine)
        #we save the final cone profile
        Cone.export(bdir+os.sep+prefix+'_Cone.stl')

    def AddSaveDataSim(self,DataForSim):
        DataForSim['ZSteering']=self._ZSteering
        DataForSim['BasePhasedArrayProgramming']=self._SIM_SETTINGS.BasePhasedArrayProgramming

    
########################################################
########################################################
class SimulationConditions(SimulationConditionsBASE):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,FactorEnlarge = 1, #putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
                      Aperture=33.60e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                      FocalLength=0.0,
                      ZSteering=0.0,
                      InDiameters= np.array([0.0    , 24.0e-3]), #inner diameter of rings
                      OutDiameters=np.array([23.3e-3,33.60e-3]), #outer diameter of rings
                      **kargs): # steering
        super().__init__(Aperture=Aperture*FactorEnlarge,FocalLength=0,**kargs)
        self._FactorEnlarge=FactorEnlarge
        self._OrigAperture=Aperture
        self._OrigFocalLength=FocalLength
        self._OrigInDiameters=InDiameters
        self._OrigOutDiameters=OutDiameters
        self._Aperture=Aperture*FactorEnlarge
        self._FocalLength=FocalLength*FactorEnlarge
        self._InDiameters=InDiameters*FactorEnlarge
        self._OutDiameters=OutDiameters*FactorEnlarge
        self._ZSteering=ZSteering
        
    
    def GenTx(self,bOrigDimensions=False):
        fScaling=1.0
        if bOrigDimensions:
            fScaling=self._FactorEnlarge
        print('self._InDiameters, self._OutDiameters,self._FocalLength',self._InDiameters/fScaling, self._OutDiameters/fScaling,self._FocalLength/fScaling)
        FocalLengthFlat=1e3
        TxRC=GeneratedRingArrayTx(self._Frequency,FocalLengthFlat, 
                             self._InDiameters/fScaling, 
                             self._OutDiameters/fScaling, 
                             SpeedofSoundWater(20.0))
        TxRC['Aperture']=self._Aperture/fScaling
        TxRC['NumberElems']=len(self._InDiameters)
        TxRC['center'][:,2]=0
        TxRC['elemcenter'][:,2]=0
        for n in range(len(TxRC['RingVertDisplay'])):
            TxRC['RingVertDisplay'][n][:,2]=0
        return TxRC
    
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        if platform != "darwin":
            InitCuda()
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elemens
        self._TxRC=self.GenTx()
        self._TxRCOrig=self.GenTx(bOrigDimensions=True)
        
        
        for Tx in [self._TxRC,self._TxRCOrig]:
            for k in ['center','RingVertDisplay','elemcenter']:
                if k == 'RingVertDisplay':
                    for n in range(len(Tx[k])):
                        Tx[k][n][:,0]+=self._TxMechanicalAdjustmentX
                        Tx[k][n][:,1]+=self._TxMechanicalAdjustmentY
                        Tx[k][n][:,2]+=self._TxMechanicalAdjustmentZ
                else:
                    Tx[k][:,0]+=self._TxMechanicalAdjustmentX
                    Tx[k][:,1]+=self._TxMechanicalAdjustmentY
                    Tx[k][:,2]+=self._TxMechanicalAdjustmentZ
        
        #we apply an homogeneous pressure 
       
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        #we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming=np.zeros(self._TxRC['NumberElems'],np.complex64)
        
        
        print('Running Steering')
        ds=np.ones((1))*self._SpatialStep**2

        center=np.zeros((1,3),np.float32)
        #to avoid adding an erroneus steering to the calculations, we need to discount the mechanical motion 
        center[0,0]=self._XDim[self._FocalSpotLocation[0]]+self._TxMechanicalAdjustmentX
        center[0,1]=self._YDim[self._FocalSpotLocation[1]]+self._TxMechanicalAdjustmentY
        center[0,2]=self._ZSteering

        u2back=np.zeros(self._TxRC['NumberElems'],np.complex64)
        nBase=0
        for n in range(self._TxRC['NumberElems']):
            u0=np.ones(self._TxRC['elemdims'][n][0],np.complex64)
            SelCenters=self._TxRC['center'][nBase:nBase+self._TxRC['elemdims'][n][0],:].astype(np.float32)
            SelDs=self._TxRC['ds'][nBase:nBase+self._TxRC['elemdims'][n][0],:].astype(np.float32)
            u2back[n]=ForwardSimple(cwvnb_extlay,SelCenters,SelDs,
                                u0,center,deviceMetal=deviceName)[0]
            nBase+=self._TxRC['elemdims'][n][0]

        AllPhi=np.zeros(self._TxRC['NumberElems'])
        for n in range(self._TxRC['NumberElems']):
            self.BasePhasedArrayProgramming[n]=np.exp(-1j*np.angle(u2back[n]))
            phi=-np.angle(u2back[n])
            AllPhi[n]=phi

        self.BasePhasedArrayProgramming=np.exp(1j*AllPhi)
        print('Phase for array: [',np.rad2deg(AllPhi).tolist(),']')
        u0=np.zeros((self._TxRC['center'].shape[0],1),np.complex64)
        nBase=0
        for n in range(self._TxRC['NumberElems']):
            u0[nBase:nBase+self._TxRC['elemdims'][n][0]]=(self._SourceAmpPa*np.exp(1j*AllPhi[n])).astype(np.complex64)
            nBase+=self._TxRC['elemdims'][n][0]

        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)

        xp,yp,zp=np.meshgrid(self._XDim,self._YDim,self._ZDim+self._ZSteering-self._TxMechanicalAdjustmentZ,indexing='ij')
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._TxRC['center'].astype(np.float32),
                        self._TxRC['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)
        u2[zp==0]=0
        
        self._u2RayleighField=u2
        
        self._SourceMapFlat=u2[:,:,self._PMLThickness]*0
        xpp,ypp=np.meshgrid(self._XDim+self._TxMechanicalAdjustmentX,self._YDim+self._TxMechanicalAdjustmentY,indexing='ij')
        
        
        EqCircle=xpp**2+ypp**2
        for n in range(2):
            RegionMap=(EqCircle>=(self._InDiameters[n]/2)**2) & (EqCircle<=(self._OutDiameters[n]/2)**2) 
            self._SourceMapFlat[RegionMap]=self._SourceAmpPa*np.exp(1j*AllPhi[n])

       
        if self._bDisplay:
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(np.abs(self._SourceMapFlat)/1e6,
                       vmin=np.abs(self._SourceMapFlat[RegionMap]).min()/1e6,cmap=plt.cm.jet)
            plt.colorbar()
            plt.title('Incident map to be forwarded propagated (MPa)')

            plt.subplot(1,2,2)
            
            plt.imshow((np.abs(u2[:,self._FocalSpotLocation[1],:]).T+
                                       self._MaterialMap[self._FocalSpotLocation[0],:,:].T*
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
        
        SourceMaskIND=np.where(np.abs(self._SourceMapFlat)>0)
        SourceMask=np.zeros((self._N1,self._N2),np.uint32)
        
        RefI= int((SourceMaskIND[0].max()-SourceMaskIND[0].min())/2)+SourceMaskIND[0].min()
        RefJ= int((SourceMaskIND[1].max()-SourceMaskIND[1].min())/2)+SourceMaskIND[1].min()
        AngRef=np.angle(self._SourceMapFlat[RefI,RefJ])
        PulseSource = np.zeros((np.sum(np.abs(self._SourceMapFlat)>0),TimeVectorSource.shape[0]))
        nSource=1                       
        for i,j in zip(SourceMaskIND[0],SourceMaskIND[1]):
            SourceMask[i,j]=nSource
            u0=self._SourceMapFlat[i,j]
            #we recover amplitude and phase from Rayleigh field
            PulseSource[nSource-1,:] = np.abs(u0) *np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(u0))
            PulseSource[nSource-1,:int(ramp_length_points)]*=ramp
            nSource+=1

        self._SourceMap[:,:,LocZ]=SourceMask 
            
        self._PulseSource=PulseSource
        
       
        if self._bDisplay:
            plt.figure(figsize=(6,3))
            for n in range(1,4):
                plt.plot(TimeVectorSource*1e6,PulseSource[int(PulseSource.shape[0]/4)*n,:])
                plt.title('CW signal, example %i' %(n))
                
            plt.xlim(0,50)
                
            plt.figure(figsize=(3,2))
            plt.imshow(self._SourceMap[:,:,LocZ])
            plt.title('source map - source ids')