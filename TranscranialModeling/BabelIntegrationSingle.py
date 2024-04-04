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
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, SpeedofSoundWater

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

    nrstep = np.ceil(ArcC/lstep);

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

def GenerateFocusTx(f,Foc,Diam,c,PPWSurface=5):
    wavelength = c/f
    lstep = wavelength/PPWSurface

    Tx = GenerateSurface(lstep,Diam,Foc)
    return Tx

class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(**kargs)
    def RunCases(self,**kargs):
        self._Aperture=kargs['Aperture']
        self._FocalLength=kargs['FocalLength']
        return super().RunCases(**kargs)


class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,
                 Aperture=50e-3,
                 FocalLength=50e-3,
                 **kargs):
        self._Aperture=Aperture
        self._FocalLength=FocalLength
        super().__init__(**kargs)
        
    def CreateSimConditions(self,**kargs):
        return SimulationConditions(Aperture=self._Aperture, 
                                    FocalLength=self._FocalLength,
                                    **kargs)
    
    def GenerateSTLTx(self,prefix):
        n=1
        VertDisplay=self._SIM_SETTINGS._TxRCOrig['VertDisplay']
        FaceDisplay=self._SIM_SETTINGS._TxRCOrig['FaceDisplay']
    
        #we also export the STL of the Tx for display in Brainsight or 3D slicer
        TxVert=VertDisplay.T.copy()
        TxVert/=self._SIM_SETTINGS.SpatialStep
        TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])
        affine=self._SkullMask.affine

        LocSpot=np.array(np.where(self._SkullMask.get_fdata()==5.0)).flatten()

        TxVert[2,:]=-TxVert[2,:]
        TxVert[0,:]+=LocSpot[0]
        TxVert[1,:]+=LocSpot[1]
        TxVert[2,:]+=LocSpot[2]+self._SIM_SETTINGS._FocalLength/self._SIM_SETTINGS._FactorEnlarge/self._SIM_SETTINGS.SpatialStep

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
        DataForSim['Aperture']=self._Aperture
        DataForSim['FocalLength']=self._FocalLength

    
########################################################
########################################################
class SimulationConditions(SimulationConditionsBASE):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,FactorEnlarge = 1.0, #putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
                      Aperture=64e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                      FocalLength=63.2e-3,
                      **kargs): # steering
        super().__init__(Aperture=Aperture*FactorEnlarge,FocalLength=FocalLength*FactorEnlarge,**kargs)
        self._FactorEnlarge=FactorEnlarge
        self._OrigAperture=Aperture
        self._OrigFocalLength=FocalLength
        self._Aperture=Aperture*FactorEnlarge
        self._FocalLength=FocalLength*FactorEnlarge
        
        
    
    def GenTx(self,bOrigDimensions=False):
        fScaling=1.0
        if bOrigDimensions:
            fScaling=self._FactorEnlarge
        TxRC=GenerateFocusTx(self._Frequency,self._FocalLength/fScaling, 
                             self._Aperture/fScaling, 
                             SpeedofSoundWater(20.0))
        TxRC['Aperture']=self._Aperture/fScaling
        TxRC['center'][:,2]+=self._FocalLength/fScaling
        TxRC['elemcenter'][:,2]+=self._FocalLength/fScaling
        TxRC['VertDisplay'][:,2]+=self._FocalLength/fScaling
        return TxRC
    
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elements
        self._TxRC=self.GenTx()
        self._TxRCOrig=self.GenTx(bOrigDimensions=True)
        
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
            
            ax.add_collection3d(Poly3DCollection(self._TxRC['VertDisplay'][self._TxRC['FaceDisplay']]*1e3)) #we plot the units in mm
                #3D display are not so smart as regular 2D, so we have to adjust manually the limits so we can see the figure correctly
            ax.set_xlim(-self._TxRC['Aperture']/2*1e3-5,self._TxRC['Aperture']/2*1e3+5)
            ax.set_ylim(-self._TxRC['Aperture']/2*1e3-5,self._TxRC['Aperture']/2*1e3+5)
            ax.set_zlim(0,135)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.show()
        
        for Tx in [self._TxRC,self._TxRCOrig]:
            for k in ['center','VertDisplay','elemcenter']:
                Tx[k][:,0]+=self._TxMechanicalAdjustmentX
                Tx[k][:,1]+=self._TxMechanicalAdjustmentY
                Tx[k][:,2]+=self._TxMechanicalAdjustmentZ-StartSkin
        Correction=0.0
        while np.max(self._TxRC['center'][:,2])>=self._ZDim[self._ZSourceLocation]:
            #at the most, we could be too deep only a fraction of a single voxel, in such case we just move the Tx back a single step
            for Tx in [self._TxRC,self._TxRCOrig]:
                for k in ['center','VertDisplay','elemcenter']:
                    Tx[k][:,2]-=self._SkullMaskNii.header.get_zooms()[2]/1e3
            Correction+=self._SkullMaskNii.header.get_zooms()[2]/1e3
        if Correction>0:
            print('Warning: Need to apply correction to reposition Tx for',Correction)
        #if yet we are not there, we need to stop
        if np.max(self._TxRC['center'][:,2])>self._ZDim[self._ZSourceLocation]:
            print("np.max(self._TxRC['center'][:,2]),self._ZDim[self._ZSourceLocation]",np.max(self._TxRC['center'][:,2]),self._ZDim[self._ZSourceLocation])
            raise RuntimeError("The Tx limit in Z is below the location of the layer for source location for forward propagation.")
      
        #we apply an homogeneous pressure 
       
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        u0=(np.ones((self._TxRC['center'].shape[0],1),np.float32)+ 1j*np.zeros((self._TxRC['center'].shape[0],1),np.float32))*self._SourceAmpPa
        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        yp,xp,zp=np.meshgrid(self._YDim,self._XDim,self._ZDim)
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._TxRC['center'].astype(np.float32),
                         self._TxRC['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)
        
        self._u2RayleighField=u2
        
        self._SourceMapRayleigh=u2[:,:,self._ZSourceLocation].copy()
        self._SourceMapRayleigh[:self._PMLThickness,:]=0
        self._SourceMapRayleigh[-self._PMLThickness:,:]=0
        self._SourceMapRayleigh[:,:self._PMLThickness]=0
        self._SourceMapRayleigh[:,-self._PMLThickness:]=0
        if self._bDisplay:
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(np.abs(self._SourceMapRayleigh)/1e6,
                       vmin=np.abs(self._SourceMapRayleigh[RegionMap]).min()/1e6,cmap=plt.cm.jet)
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
        
       
        if self._bDisplay:
            plt.figure(figsize=(6,3))
            for n in range(1,4):
                plt.plot(TimeVectorSource*1e6,PulseSource[int(PulseSource.shape[0]/4)*n,:])
                plt.title('CW signal, example %i' %(n))
                
            plt.xlim(0,50)
                
            plt.figure(figsize=(3,2))
            plt.imshow(self._SourceMap[:,:,LocZ])
            plt.title('source map - source ids')