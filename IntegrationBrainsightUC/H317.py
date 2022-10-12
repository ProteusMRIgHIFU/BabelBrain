from re import S
import numpy as np
#from mkl_fft import fft
from numpy.fft import fft
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from  scipy.io import loadmat
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, InitCuda, InitOpenCL , GenerateFocusTx,SpeedofSoundWater
from BabelViscoFDTD.PropagationModel import PropagationModel
from BabelViscoFDTD.H5pySimple import SaveToH5py, ReadFromH5py
from pprint import pprint
from scipy.io import loadmat
from sys import platform
from time import time
import os

PModel=PropagationModel()
import gc
import time

extlay={}
TemperatureWater=20.0
extlay['c']=SpeedofSoundWater(TemperatureWater)

DbToNeper=1/(20*np.log10(np.exp(1)))

Material={}
#Density (kg/m3), LongSoS (m/s), ShearSoS (m/s), Long Att (Np/m), Shear Att (Np/m)
Material['Water']=     np.array([1000.0, 1500.0, 0.0,   0.0,                   0.0] )
Material['SofTissue']= np.array([1000.0, 1500.0, 0.0,   1.0 * DbToNeper * 100*700/500,  0.0] )
Material['Cortical']=  np.array([1850.0, 2800.0, 1550.0, 38.18, 12 * DbToNeper * 100*700/500])
Material['Trabecular']=np.array([1700.0, 2300.0, 1400.0, 30.63, 12 * DbToNeper * 100*700/500])
Material['Skin']=      np.array([1090.0, 1610.0, 0.0,    0.2 * DbToNeper * 100*700/500, 0]) #we scaled from the value at 500 kHz
Material['Brain']=     np.array([1040.0, 1560.0, 0.0,    4.278, 0])

SmallestSOS=Material['Trabecular'][2]

def computeH317Geometry(bDoRunDeepSanityTest=False,ToleranceDistance=9.5,FocalDistance=135):
    MininalInterElementDistanceInRadians=ToleranceDistance/FocalDistance # we can test using the angular distance from center n to center m
    print('*****\nMinimal angular distance\n*****', MininalInterElementDistanceInRadians)
    transxyz = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'H-317 XYZ Coordinates_revB update 1.18.22.csv'),delimiter=',',skiprows=1)
    assert(transxyz.shape[0]==128) #number of elements
    assert(transxyz.shape[1]==4) #element, X,Y,Z coordinates in IN
    transxyz[:,1:]*=25.4 # from in to mm
    transxyz=transxyz[:,1:] #we skip the Tx # 
    if bDoRunDeepSanityTest:
        # we will calculate the closest Tx element distance to each other
        MinimalDistances=np.zeros(128)
        for n in range(128):
            #we select all the other Tx elements
            Sel=np.hstack((np.arange(0,n),np.arange(n+1,128)))
            SelElem=transxyz[n,:].reshape((1,3))
            RestTx=transxyz[Sel,:]
            #we calculate the Euclidean distance 
            DistanceToOtherElements=np.linalg.norm(RestTx-np.repeat(SelElem,127,axis=0),axis=1)
            NormalizedDistance=DistanceToOtherElements/FocalDistance
            #the angular distance between the tx elements is straighforward
            AngularDistance=2*np.arcsin(NormalizedDistance/2.0)
            
            assert(AngularDistance.size==127)
            MinimalDistances[n]=AngularDistance.min()
            print('Closest element distance',n,MinimalDistances[n]) # just for nicer printing
            if MinimalDistances[n]<MininalInterElementDistanceInRadians:
                print(' ******** overlap of elem %i with %i' % (n,Sel[np.argmin(DistanceToOtherElements)]) )
        assert(np.all(MinimalDistances>=MininalInterElementDistanceInRadians))
        
            
    return transxyz

def H317Locations(Foc=135e-3):
    radiusMm = Foc*1e3
    temp_positions = computeH317Geometry()
    temp_positions[:,2]=radiusMm-temp_positions[:,2]
    transLoc = temp_positions/1000
    return transLoc

def GenerateH317Tx(Frequency=700e3,RotationZ=0):


    f=Frequency;
    Foc=135e-3
    Diameter=9.5e-3

    #%This is the indiv tx element
    TxElem=GenerateFocusTx(f,Foc,Diameter,extlay['c'])

    # fig = plt.figure()
    # ax = Axes3D(fig)

    # for n in range(TxElem['FaceDisplay'].shape[0]):
        # verts=TxElem['VertDisplay'][TxElem['FaceDisplay'][n,:],:]
        # ax.scatter3D(verts[:,0],verts[:,1],verts[:,2],color='r')
    # plt.show()


    transLoc = H317Locations(Foc=Foc)
    if RotationZ!=0:
        theta=np.deg2rad(RotationZ)
        ct=np.cos(theta)
        st=np.sin(theta)
        Rz=np.zeros((3,3))
        Rz[0,0]=ct
        Rz[0,1]=-st
        Rz[1,0]=st 
        Rz[1,1]=ct
        Rz[2,2]=1
        transloc=(Rz@transLoc.T).T

    transLocDisplacedZ=transLoc.copy()
    transLocDisplacedZ[:,2]-=Foc

    XYNorm=np.linalg.norm(transLocDisplacedZ[:,:2],axis=1)
    VN=np.linalg.norm(transLocDisplacedZ,axis=1)

    theta=np.arcsin(XYNorm/VN)
    phi=np.arctan2(transLocDisplacedZ[:,1],transLocDisplacedZ[:,0])

    TxH317={}
    TxH317['center'] = np.zeros((0,3))
    TxH317['elemcenter'] = np.zeros((len(theta),3))
    TxH317['ds'] = np.zeros((0,1))
    TxH317['normal'] = np.zeros((0,3))
    TxH317['elemdims']=TxElem['ds'].size
    TxH317['NumberElems']=len(theta)
    TxH317['VertDisplay'] = np.zeros((0,3))
    TxH317['FaceDisplay'] = np.zeros((0,4),np.int)

    for n in range(len(theta)):
        prevFaceLength=TxH317['VertDisplay'].shape[0]
        rotateMatrixY = np.array([[np.cos(theta[n]),0,np.sin(theta[n])],[0,1,0],[-np.sin(theta[n]),0,np.cos(theta[n])]])
        rotateMatrixZ = np.array([[-np.cos(phi[n]),np.sin(phi[n]),0],[-np.sin(phi[n]),-np.cos(phi[n]),0],[0,0,1]])
        rotateMatrix = rotateMatrixZ@rotateMatrixY
       
        center=(rotateMatrix@TxElem['center'].T).T
        TxH317['elemcenter'][n,:]=center[0,:] # the very first subelement is at the center
        
        normal=(rotateMatrix@TxElem['normal'].T).T
        
        VertDisplay=(rotateMatrix@TxElem['VertDisplay'].T).T
       
        TxH317['center']=np.vstack((TxH317['center'],center))
        TxH317['ds'] =np.vstack((TxH317['ds'],TxElem['ds']))
        TxH317['normal'] =np.vstack((TxH317['normal'],normal))
        TxH317['VertDisplay'] =np.vstack((TxH317['VertDisplay'],VertDisplay))
        TxH317['FaceDisplay']=np.vstack((TxH317['FaceDisplay'],TxElem['FaceDisplay']+prevFaceLength))
        

    TxH317['VertDisplay'][:,2]+=Foc
    TxH317['center'][:,2]+=Foc
    TxH317['elemcenter'][:,2]+=Foc
    
    print('Aperture dimensions (x,y) =',TxH317['VertDisplay'][:,0].max()-TxH317['VertDisplay'][:,0].min(),
                                        TxH317['VertDisplay'][:,1].max()-TxH317['VertDisplay'][:,1].min())

    # fig = plt.figure(figsize=(8,8))
    # ax = Axes3D(fig)

    # ax.scatter3D(TxH317['center'][::5,0],TxH317['center'][::5,1],TxH317['center'][::5,2],color='r')
    # plt.show()
    TxH317['FocalLength']=Foc
    TxH317['Aperture']=np.max([TxH317['VertDisplay'][:,0].max()-TxH317['VertDisplay'][:,0].min(),
                                      TxH317['VertDisplay'][:,1].max()-TxH317['VertDisplay'][:,1].min()]);
    return TxH317


extlay={}
TemperatureWater=20.0
extlay['c']=SpeedofSoundWater(TemperatureWater)
bInitCuda=False



class AcFieldH317(object):
    def __init__(self,RotationZ=0):
        self.Data=loadmat('Oct252021-4V-XY-Scan.mat')
   
        
        self.Frequency=700e3
        self.SoS=1.4825e+03
        self.Foc=135e-3
        self.RotationZ=RotationZ
        
        SpatialStep=self.SoS/self.Frequency/4
        self.SpatialStep=SpatialStep
        
        xfmin=-3.5e-2
        xfmax=3.5e-2
        yfmin=-3.5e-2
        yfmax=3.5e-2
        zfmin=5e-2
        zfmax=17e-2
        
        self.xfmin=xfmin
        self.xfmax=xfmax
        self.yfmin=yfmin
        self.yfmax=yfmax
        self.zfmin=zfmin
        self.zfmax=zfmax

        xfield = np.linspace(xfmin,xfmax,int(np.ceil((xfmax-xfmin)/SpatialStep)+1))
        yfield = np.linspace(yfmin,yfmax,int(np.ceil((yfmax-yfmin)/SpatialStep)+1))
        zfield = np.linspace(zfmin,zfmax,int(np.ceil((zfmax-zfmin)/SpatialStep)+1))
        nxf=len(xfield)
        nyf=len(yfield)
        nzf=len(zfield)
        xp,yp,zp=np.meshgrid(xfield,yfield,zfield)
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        self.xp=xp
        self.yp=xp
        self.zp=xp
        
        self.cx=np.argmin(np.abs(xfield))
        self.cy=np.argmin(np.abs(yfield))
        self.cz=np.argmin(np.abs(zfield-self.Foc))
                     
        self.xfield=xfield
        self.yfield=yfield
        self.zfield=zfield

        self.rf=rf

        self.Amp=60e3 #60 kPa
        
        self.TxH317=GenerateH317Tx(RotationZ=RotationZ)

    def BackPropagation(self):
        nxf=self.Data['lengtharray'].size
        nyf=self.Data['widtharray'].size
        nzf=1
        xf=np.round(self.Data['lengtharray'],1)*1e-3
        yf=np.round(self.Data['widtharray'],1)*1e-3
        dx=np.diff(xf).mean()
        xo,yo=np.meshgrid(xf,yf)
        
        xp,yp,zp=np.meshgrid(xf,yf,self.Foc)
        Att=0.0
        cwvnb_extlay=np.array(2*np.pi*self.Frequency/self.SoS+(-1j*Att)).astype(np.complex64)
        CenterPlane=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        u0=(self.Data['Kwaveamplitude']*np.exp(-1j*self.Data['Kwavephase'])).flatten()
        ds=np.ones(u0.shape)*dx*dx
        u2=ForwardSimple(cwvnb_extlay,CenterPlane.astype(np.float32),
                         ds.astype(np.float32),
                         u0.astype(np.complex64),self.TxH317['elemcenter'].astype(np.float32),MacOsPlatform='OpenCL' ,deviceMetal='M1')
        
        self.u0_sources=u2
        return u2
        
        
    def AcousticSim(self,bUseBackProp=True,bPlot=False):
        u0=np.ones((self.TxH317['center'].shape[0],1),np.complex64)
        if bUseBackProp:
            uBack=self.BackPropagation()
            for n in range(self.TxH317['elemcenter'].shape[0]):
                u0[self.TxH317['elemdims']*n:self.TxH317['elemdims']*(n+1)]=uBack[n]
                
                  
        Att=0.0
        cwvnb_extlay=np.array(2*np.pi*self.Frequency/self.SoS+(-1j*Att)).astype(np.complex64)
        u2=ForwardSimple(cwvnb_extlay,self.TxH317['center'].astype(np.float32),
                         self.TxH317['ds'].astype(np.float32),u0,self.rf,deviceMetal='M1',MacOsPlatform='OpenCL' )
        u2=np.reshape(u2,self.xp.shape)
        
        if bUseBackProp:
            print(self.Data['Kwaveamplitude'].max())
            
            FactorBackProp=self.Data['Kwaveamplitude'].max()/np.abs(u2).max()
            u2*=FactorBackProp
            self.u0_sources*=FactorBackProp
            
        
        
        if bPlot:
            plt.figure(figsize=(12,6))
            cy=np.argmin(np.abs(self.yfield))
            plt.imshow(np.abs(u2[:,cy,:].T),extent=(self.xfmin,self.xfmax,self.zfmax,self.zfmin))
            plt.colorbar()

        return np.abs(u2)
    
    def AcousticSimToPlane(self,DistanceToFocus=0.0185+0.01,bPlot=False,FactorExpandROI=1.35):
        a=self.TxH317['Aperture']/2 #half aperture, m
        r=self.Foc#curvature radius, m
        theta=np.arcsin(a/r)
        c=np.cos(theta)*r
        h=r-c
        LocationPlane=r-DistanceToFocus
        DiameterROI=a*DistanceToFocus/c*2
        print('DiameterROI',DiameterROI,'LocationPlane',LocationPlane)
        DiameterROI*=FactorExpandROI
        Wavelength=SmallestSOS/self.Frequency
        for PPW in [6,9]:
            SpatialStep=Wavelength/PPW
            N1=int(np.ceil(DiameterROI/SpatialStep))
            if N1%2==0:
                N1+=1
            xf=np.arange(N1)*SpatialStep
            xf-=xf.mean()
            yf=xf.copy()
            nxf=xf.size
            nyf=nxf
            nzf=1
            xp,yp,zp=np.meshgrid(xf,yf,LocationPlane)
            Att=0.0
            cwvnb_extlay=np.array(2*np.pi*self.Frequency/self.SoS+(-1j*Att)).astype(np.complex64)
            CenterPlane=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
            
            u0=np.ones((self.TxH317['center'].shape[0],1),np.complex64)
            for n in range(self.TxH317['elemcenter'].shape[0]):
                u0[self.TxH317['elemdims']*n:self.TxH317['elemdims']*(n+1)]=self.u0_sources[n]
         
            Att=0.0
            cwvnb_extlay=np.array(2*np.pi*self.Frequency/self.SoS+(-1j*Att)).astype(np.complex64)
            u2=ForwardSimple(cwvnb_extlay,self.TxH317['center'].astype(np.float32),
                             self.TxH317['ds'].astype(np.float32),u0,CenterPlane.astype(np.float32),deviceMetal='M1',MacOsPlatform='OpenCL' )
            u2=np.reshape(u2,xp.shape)
            
            sname ='H317-inputplane-%3.2f-%i-PPW.npz' %(DistanceToFocus*1e3,PPW)
            np.savez_compressed(sname,u0_input=u2,xf=xf,yf=yf,zf=LocationPlane)
            
            if bPlot:
                plt.figure()
                plt.imshow(np.abs(u2[:,:,0]),extent=(xf.min(),xf.max(),yf.min(),yf.max()))
                plt.colorbar()
                plt.title('PPW =%i' % PPW)


################################################################################################################
################################################################################################################
################################################################################################################

class BabelViscoFDTD_Simulations(object):
    '''
    Meta class dealing with the specificis of each test based on the string name
    '''
    
    def __init__(self,bDisplay=True,
                 Benchmark='BM5',
                 SubType='MP2',
                 DepthFocalSpot=28.5, #mm
                 DimDomain =  np.array([0.06,0.06,0.07]),
                 OverwritePPW=None,
                 OverwriteCFL=None,
                 bSourceDisplacement=False):
        self._TESTNAME='AcField-H317-%3.2fmm' % (DepthFocalSpot)
        self._Benchmark=Benchmark # pick BM1 for water only and BM5 for curved skull
        self._SubType=SubType # MP2 for elastic, MP1 for "acoustics-like"
        self._SourceType='H317'
        if self._Benchmark in ['BM2','BM3','BM4','BM5','BM6','BM7','BM8'] and self._SubType == 'MP1':
            self._Shear=0.0
        else:
            self._Shear=1.0
            
        self._SPP=False
        self._basePPW=9
        
        self._PaddingForKArray=0
        
        self._AlphaCFL=1.0
        self._bDisplay=bDisplay
        self._DepthFocalSpot=DepthFocalSpot
        self._DimDomain=DimDomain
        
        self._bSourceDisplacement=bSourceDisplacement
        

        if not(OverwritePPW is None):
            self._basePPW=OverwritePPW
        
        if not(OverwriteCFL is None):
            self._AlphaCFL=OverwriteCFL
        
                            
    
    def GetSummary(self):
        summary=''
        summary+='### Benchmark type: '
        if self._Benchmark=='BM1':
            summary+='Homogenous medium'
            if self._SubType == 'MP1':
                summary+=' - Lossless'
            else:
                summary+=' - Absorbing'
        elif self._Benchmark=='BM2':
            summary+='Single flat cortical layer'
        elif self._Benchmark=='BM3':
            summary+='Composite flat layer'
        elif self._Benchmark=='BM4':
            summary+='Single curved cortical layer'
        elif self._Benchmark=='BM5':
            summary+='Composite curved cortical layer'
        elif self._Benchmark=='BM6':
            summary+='Half skull (homogenous) targeting V1'
        elif self._Benchmark=='BM7':
            summary+='Full skull (homogenous) targeting V1'
        elif self._Benchmark=='BM8':
            summary+='Full skull (homogenous) targeting M1'
            
        if self._Benchmark != 'BM1':
            if self._SubType == 'MP1':
                summary+='\n### Acoustic conditions (shear component disabled in elastic model)'
            else:
                summary+='\n### Elastic conditions'
        if self._SourceType =='H317':
            summary+='\n### H317 source'
        else:
            raise ValueError("Source type must be H317")
            
        summary+='\n### Resolution: %i PPW' %(self._basePPW)
        
        if self._SPP:
            summary+='\n### Special conditions: Use of superposition method'
        return summary


    def Step1_InitializeConditions(self):
        
        NumberCyclesToTrackAtEnd=2
        if self._Benchmark=='BM1':
            if self._SubType=='MP1':
                #the PPP below depend of several factors as the lowest speed of sound and PPW, the subsamplig (by the time being) is adjusted case by case
                BaseSupSamplig={6:1,9:1,12:5,15:1,18:1,24:7}
            else:
                BaseSupSamplig={6:1,9:4,12:3,15:3,18:4,24:1}
        else:
            if self._SubType=='MP1' or self._Benchmark in ['BM2','BM4','BM6','BM7','BM8']:
                BaseSupSamplig={6:4,9:9,12:8,15:10,18:9,24:19}
            else:
                BaseSupSamplig={6:2,9:3,12:3,15:8,18:11,24:6}
        SensorSubSampling=BaseSupSamplig[self._basePPW]
        
        if self._Benchmark=='BM1' and self._SubType=='MP2':
            self._SIM_SETTINGS = SimulationConditions(baseMaterial=Material['SofTissue'],
                                basePPW=self._basePPW,
                                SensorSubSampling=SensorSubSampling,  
                                NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd,                      
                                PaddingForKArray=self._PaddingForKArray,
                                bDisplay=self._bDisplay,
                                DimDomain=self._DimDomain,                     
                                DispersionCorrection=[-2285.16917671, 6925.00947243, -8007.19755945, 4388.62534545, -1032.06871257])
        else:
            self._SIM_SETTINGS = SimulationConditions(baseMaterial=Material['Water'],
                                basePPW=self._basePPW,
                                PaddingForKArray=self._PaddingForKArray,
                                bDisplay=self._bDisplay, 
                                DimDomain=self._DimDomain,
                                SensorSubSampling=SensorSubSampling,
                                NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd,
                                DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721])
        OverwriteDuration=None
        if self._Benchmark in ['BM2','BM4']:
            #single layer of crotical
            SelM=Material['Cortical']
            self._SIM_SETTINGS.AddMaterial(SelM[0],SelM[1],SelM[2]*self._Shear,SelM[3],SelM[4]*self._Shear)
            
        elif self._Benchmark in ['BM3','BM5']:
            #composite layer of skin, cortical , trabecular and brain
            for k in ['Skin','Cortical','Trabecular','Brain']:
                SelM=Material[k]
                self._SIM_SETTINGS.AddMaterial(SelM[0],SelM[1],SelM[2]*self._Shear,SelM[3],SelM[4]*self._Shear)
        else:
            assert(self._Benchmark=='BM1')
            #the default conditions are water only
        
        self._SIM_SETTINGS.UpdateConditions(AlphaCFL=self._AlphaCFL,OverwriteDuration=OverwriteDuration)
            

    def Step2_PrepMaterials(self,**kwargs):
        if platform == 'win32':
            #this is the trick, we call first the linux proxy server, as this will create the files so the Win version will just load them
            self._proxy.Step2_PrepMaterials(**kwargs)
        
        print('Material properties\n',self._SIM_SETTINGS.ReturnArrayMaterial(),'[%s]' %(self._Benchmark))
        
        if self._Benchmark =='BM2':
            #single flat skull bone
            self._SIM_SETTINGS.CreateMaterialMapSingleFlatSlice(ThicknessBone=6.5e-3,LocationBone=10e-3)
        elif self._Benchmark =='BM3':
            #multi tissue type flat material
            self._SIM_SETTINGS.CreateMaterialMapCompositeFlatSlice(ThicknessMaterials=[4e-3,1.5e-3,4e-3,1e-3],Location=26e-3)
        elif self._Benchmark =='BM4':
            
            if self._SPP:
                #SPP based
                self._SIM_SETTINGS.CreateMaterialMapSingleCurvedSliceSPP(ThicknessBone=6.5e-3,LocationBone=10e-3,SkullRadius=75e-3,
                                                                      TESTNAME=self._Phase+'-'+self._Benchmark+'-%iPPW' % (self._basePPW), **kwargs)
            else:
                 #single curved skull bonel
                self._SIM_SETTINGS.CreateMaterialMapSingleCurvedSlice(ThicknessBone=6.5e-3,LocationBone=10e-3,SkullRadius=75e-3)
             
        elif self._Benchmark =='BM5':
            LocationBone=10e-3;
            if self._basePPW >12:
                 LocationBone+=self._SIM_SETTINGS._SpatialStep*0.75;
            if self._SPP:
                #SPP based
                self._SIM_SETTINGS.CreateMaterialMapCompositeCurvedSliceSPP(ThicknessBone=6.5e-3,LocationBone=LocationBone,SkullRadius=75e-3,ThicknessTissues=[4e-3,1.5e-3,4e-3,1e-3],TESTNAME=self._Phase+'-'+self._Benchmark+'-%iPPW'% (self._basePPW),**kwargs)
            else:
                self._SIM_SETTINGS.CreateMaterialMapCompositeCurvedSlice(ThicknessBone=6.5e-3,LocationBone=LocationBone,SkullRadius=75e-3,ThicknessTissues=[4e-3,1.5e-3,4e-3,1e-3])
       
        elif self._Benchmark !='BM1':
            raise ValueError('Benchmark not supported ' + self._Benchmark)
        gc.collect()

    def Step3_PrepareSource(self):

        TxDiam = 0.02 # m, circular piston
        self._SIM_SETTINGS.CreateRayleighH317Source(bSourceDisplacement=self._bSourceDisplacement,
                                                    DepthFocalSpot=self._DepthFocalSpot)
        AA=np.where(self._SIM_SETTINGS._weight_amplitudes==1.0)
        print("index location, and location in Z = %i, %g"  % (AA[2].min(),self._SIM_SETTINGS._ZDim[AA[2].min()]))
        assert(AA[2].min()>=1.0)
        self._SIM_SETTINGS.PlotWeightedAverags()
             

    def Step4_CreateSensor(self,b3D=True):
        self._SIM_SETTINGS.CreateSensorMap(b3D=b3D)

    def Step5_Run_Simulation(self,COMPUTING_BACKEND=3,GPUName='M1',GPUNumber=0,bApplyCorrectionForDispersion=False):
        SelMapsRMSPeakList=['Pressure']
            
        if self._SPP:
            bUse_SPP=True
            nSPP=10
        else:
            bUse_SPP=False
            nSPP=1
            
        self._SIM_SETTINGS.RUN_SIMULATION(COMPUTING_BACKEND=COMPUTING_BACKEND, GPUName=GPUName,GPUNumber=GPUNumber,SelMapsRMSPeakList=SelMapsRMSPeakList,
                                         bSourceDisplacement=self._bSourceDisplacement,
                                         bApplyCorrectionForDispersion=bApplyCorrectionForDispersion,
                                         bUse_SPP=bUse_SPP,nSPP=nSPP)
        gc.collect()

    def Step6_ExtractPhaseData(self):
        b3D= True
        self._SIM_SETTINGS.CalculatePhaseData(b3D=b3D)
        gc.collect()

    def Step7_PrepAndPlotData(self):
        self._SIM_SETTINGS.PlotData3D()
        
    def Step8_ResamplingAndExport(self,bReload=False,bSkipSaveAndReturnData=False,bUseCupyToInterpolate=False):
        return self._SIM_SETTINGS.ResamplingToFocusConditions3D(TESTNAME=self._TESTNAME,
                        bReload=bReload,
                        bSkipSaveAndReturnData=bSkipSaveAndReturnData,bUseCupyToInterpolate=bUseCupyToInterpolate)
        
        gc.collect()

    def OutPutConditions(self):
        ### Usage details

        String = 'Plese see below code implementing the complete simulation.\n'+\
                'Main highlights:\n\n'+\
                'Item  | value\n'+\
                '---- | ----\n'+\
                'PML size |  %i\n' %(self._SIM_SETTINGS._PMLThickness)+\
                'Spatial step$^*$ | $\\frac{\\lambda}{%i}$ = %3.2f mm (%i PPW)\n' %(self._SIM_SETTINGS._basePPW,\
                                                                                    self._SIM_SETTINGS._SpatialStep*1e3,\
                                                                                    self._SIM_SETTINGS._basePPW) +\
                'Final Interpolation at 0.5 mm | Linear for amplitude, nearest for phase\n'+\
                'FDTD solver | $O(2)$ temporal, $O(4)$ spatial, staggered grid\n'+\
                'Temporal step | %4.3f $\mu$s, %2.1f points-per-period\n'%(self._SIM_SETTINGS._TemporalStep*1e6,self._SIM_SETTINGS._PPP)+\
                'Adjusted CFL | %3.2f \n'%(self._SIM_SETTINGS._AdjustedCFL)+\
                'Source func. | CW-pulse for %4.1f $\mu$s\n' %(self._SIM_SETTINGS._TimeSimulation*1e6)+\
                'Amplitude method | Peak\n'+\
                'Phase method | NA in library, but it it is calculated from captured sensor data and FFT\n\n'+\
                '$^*$Spatial step chosen to produce peak pressure amplitude ~2% compared to reference simulation.'\
                
        return String

################################################################################################################
################################################################################################################
################################################################################################################
class SimulationConditions(object):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,baseMaterial=Material['Water'],
                      basePPW=6,
                      PMLThickness = 12, # grid points for perect matching layer, HIGHLY RECOMMENDED DO NOT CHANGE THIS SIZE 
                      ReflectionLimit= 1e-5, #DO NOT CHANGE THIS
                      DimDomain =  np.array([0.07,0.07,0.12]),
                      SensorSubSampling=1,
                      NumberCyclesToTrackAtEnd=2,
                      SourceAmp=60e3, # kPa
                      Frequency=700e3,
                      DepthFocalSpot=28.5, #mm
                      PaddingForKArray=0,
                      QfactorCorrection=True,
                      bDisplay=True,
                      DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721]):  #coefficients to correct for values lower of CFL =1.0 in wtaer conditions.
        self._Materials=[[baseMaterial[0],baseMaterial[1],baseMaterial[2],baseMaterial[3],baseMaterial[4]]]
        self._basePPW=basePPW
        self._PMLThickness=PMLThickness
        self._ReflectionLimit=ReflectionLimit
        self._ODimDomain =DimDomain 
        self._SensorSubSampling=SensorSubSampling
        self._NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd
        self._TemporalStep=0.
        self._DepthFocalSpot=DepthFocalSpot
        self._N1=0
        self._N2=0
        self._N3=0
        self._FactorConvPtoU=baseMaterial[0]*baseMaterial[1]
        self._SourceAmpPa=SourceAmp
        self._SourceAmpDisplacement=SourceAmp/self._FactorConvPtoU
        self._Frequency=Frequency
        self._weight_amplitudes=1.0
        self._PaddingForKArray=PaddingForKArray
        self._QfactorCorrection=QfactorCorrection
        self._bDisplay=bDisplay
        self._DispersionCorrection=DispersionCorrection
        self._GMapTotal=None
        
        
    def AddMaterial(self,Density,LSoS,SSoS,LAtt,SAtt): #add material (Density (kg/m3), long. SoS 9(m/s), shear SoS (m/s), Long. Attenuation (Np/m), shear attenuation (Np/m)
        self._Materials.append([Density,LSoS,SSoS,LAtt,SAtt]);
        
        
    @property
    def Wavelength(self):
        return self._Wavelength
        
        
        
    @property
    def SpatialStep(self):
        return self._SpatialStep
        
    def UpdateConditions(self, AlphaCFL=1.0,OverwriteDuration=None,DomMaterial=0):
        '''
        Update simulation conditions
        '''
        MatArray=self.ReturnArrayMaterial()
        SmallestSOS=np.sort(MatArray[:,1:3].flatten())
        iS=np.where(SmallestSOS>0)[0]
        SmallestSOS=SmallestSOS[iS[0]]
        self._Wavelength=SmallestSOS/self._Frequency
        self._baseAlphaCFL =AlphaCFL
        print(" Wavelength, baseAlphaCFL",self._Wavelength,AlphaCFL)
        print ("smallSOS ", SmallestSOS)
        
        SpatialStep=self._Wavelength/self._basePPW
        
        dummyMaterialMap=np.zeros((10,10,MatArray.shape[0]),np.uint32)
        for n in range(MatArray.shape[0]):
            dummyMaterialMap[:,:,n]=n
        
        OTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap,MatArray,self._Frequency,self._QfactorCorrection,SpatialStep,AlphaCFL)
        
        self.DominantMediumTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap*0,MatArray[DomMaterial,:].reshape((1,5)),self._Frequency,self._QfactorCorrection,SpatialStep,1.0)

        TemporalStep=OTemporalStep

        print('"ideal" TemporalStep',TemporalStep)
        print('"ideal" DominantMediumTemporalStep',self.DominantMediumTemporalStep)

        #now we make it to be an integer division of the period
        TemporalStep=1/self._Frequency/(np.ceil(1/self._Frequency/TemporalStep)) # we make it an integer of the period
        #and back to SpatialStep
        print('"int fraction" TemporalStep',TemporalStep)
        print('"CFL fraction relative to dominant only conditions',TemporalStep/self.DominantMediumTemporalStep)
        
        self._PPP=int(np.round(1/self._Frequency/TemporalStep))
        self._AdjustedCFL=TemporalStep/OTemporalStep*AlphaCFL
        print("adjusted AlphaCL, PPP",self._AdjustedCFL,self._PPP)
        
        self._SpatialStep=SpatialStep
        self._TemporalStep=TemporalStep
        
        self._N1=int(np.ceil(self._ODimDomain[0]/self._SpatialStep)+2*self._PMLThickness)
        if self._N1%2==0:
            self._N1+=1
        self._N2=int(np.ceil(self._ODimDomain[1]/self._SpatialStep)+2*self._PMLThickness)
        if self._N2%2==0:
            self._N2+=1
        self._N3=int(np.ceil(self._ODimDomain[2]/self._SpatialStep)+2*self._PMLThickness)
        #this helps to avoid the "bleeding" from the low values of the PML in the exported files 
        self._N3+=self._PaddingForKArray+int(np.floor(0.5e-3/SpatialStep)) 
        
        print('Domain size',self._N1,self._N2,self._N3)
        self._DimDomain=np.zeros((3))
        self._DimDomain[0]=self._N1*SpatialStep
        self._DimDomain[1]=self._N2*SpatialStep
        self._DimDomain[2]=self._N3*SpatialStep
        if OverwriteDuration is not None:
            self._TimeSimulation=OverwriteDuration
        else:
            self._TimeSimulation=np.sqrt(self._DimDomain[0]**2+self._DimDomain[1]**2+self._DimDomain[2]**2)/MatArray[0,1] #time to cross one corner to another
            self._TimeSimulation=np.floor(self._TimeSimulation/self._TemporalStep)*self._TemporalStep
            
        TimeVector=np.arange(0.0,self._TimeSimulation,self._TemporalStep)
        ntSteps=(int(TimeVector.shape[0]/self._PPP)+1)*self._PPP
        self._TimeSimulation=self._TemporalStep*ntSteps
        TimeVector=np.arange(0.0,self._TimeSimulation,self._TemporalStep)
        assert(TimeVector.shape[0]==ntSteps)
        print('PPP for sensors ', self._PPP/self._SensorSubSampling)
        assert((self._PPP%self._SensorSubSampling)==0)
        
  
        nStepsBack=int(self._NumberCyclesToTrackAtEnd*self._PPP)
        self._SensorStart=int((TimeVector.shape[0]-nStepsBack)/self._SensorSubSampling)
        
        self._MaterialMap=np.zeros((self._N1,self._N2,self._N3),np.uint32) # note the 32 bit size

        
        print('PPP, Duration simulation',np.round(1/self._Frequency/TemporalStep),self._TimeSimulation*1e6)
        
        print('Number of steps sensor',np.floor(self._TimeSimulation/self._TemporalStep/self._SensorSubSampling)-self._SensorStart)
        
        self._XDim=(np.arange(self._N1)*self._SpatialStep)*1000 #mm
        self._XDim-=self._XDim.mean()
        self._YDim=(np.arange(self._N2)*self._SpatialStep)*1000 #mm
        self._YDim-=self._YDim.mean()
        self._ZDim=(np.arange(self._N3)*SpatialStep-(self._PMLThickness)*self._SpatialStep)*1000 - self._PaddingForKArray*SpatialStep*1000#mm
        
        
    def ReturnArrayMaterial(self):
        return np.array(self._Materials)
        
    def CreateMaterialMapSingleFlatSlice(self,ThicknessBone=6.5e-3,LocationBone=10e-3):
        ThicknessBoneSteps=int(np.round(ThicknessBone/self._SpatialStep))
        LocationBoneSteps=int(np.round(LocationBone/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        print("LocationBoneSteps",LocationBoneSteps,self._ZDim[LocationBoneSteps])
        self._MaterialMap[:,:,LocationBoneSteps:LocationBoneSteps+ThicknessBoneSteps]=1 #Material 1 is Cortical Bone
        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);

    def CreateMaterialMapCompositeFlatSlice(self,ThicknessMaterials=[4e-3,1.5e-3,4e-3,1e-3],Location=26e-3): 
        MatType=1
        LocationSteps=int(np.round(Location/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        for n,t in enumerate(ThicknessMaterials):
            ThicknessSteps=int(np.round(t/self._SpatialStep))
            if n==3:
                self._MaterialMap[:,:,LocationSteps:LocationSteps+ThicknessSteps]=2 
            else:
                self._MaterialMap[:,:,LocationSteps:LocationSteps+ThicknessSteps]=MatType 
                MatType+=1
            LocationSteps+=ThicknessSteps
        self._MaterialMap[:,:,LocationSteps:]=MatType

        if self._bDisplay:
            plt.figure(figsize=(14,6))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()

            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()
            plt.ylim(39,24)
            plt.xlim(-5,5)
            plt.suptitle('Material Map')
    
    
    
    def CreateMaterialMapSingleCurvedSlice(self,ThicknessBone=6.5e-3,LocationBone=10e-3,SkullRadius=75e-3):
        #Create single curved skull type interface
        LocationBoneSteps=int(np.round(LocationBone/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        
        CenterSkull=[0,0,(SkullRadius+LocationBone-self._SpatialStep*0.75)*1e3] 
        
        Mask=self.MakeSphericalMask(Radius=SkullRadius*1e3,Center=CenterSkull)
        AA=np.where(Mask)
        print(AA[2].min(), LocationBoneSteps)
        assert(AA[2].min()== LocationBoneSteps)
        BelowMask=self.MakeSphericalMask(Radius=(SkullRadius-ThicknessBone)*1e3,Center=CenterSkull)
        Mask=np.logical_xor(Mask,BelowMask)

        self._MaterialMap[Mask]=1 #Material 1 is Cortical Bone
        if self._bDisplay:
            print('Display curved')
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);

    def CreateMaterialMapCompositeCurvedSlice(self,ThicknessBone=6.5e-3,LocationBone=10e-3,SkullRadius=75e-3,ThicknessTissues=[4e-3,1.5e-3,4e-3,1e-3],adjustCenter=0.75):
        
        LocationBoneSteps=int(np.round(LocationBone/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        
        CenterSkull=[0,0,(SkullRadius+LocationBone-self._SpatialStep*adjustCenter)*1e3] 

        OuterTableROC=SkullRadius

        TissueRadius=np.array([OuterTableROC+ThicknessTissues[0], #skin
                    OuterTableROC, #outer table
                    OuterTableROC-ThicknessTissues[1], #diploe
                    OuterTableROC-ThicknessTissues[1]-ThicknessTissues[2], # inner table
                    OuterTableROC-ThicknessTissues[1]-ThicknessTissues[2]-ThicknessTissues[3]]) #brain

        print('TissueRadius before rounding in steps',TissueRadius)
        #we round each interface in terms of steps
        TissueRadius=np.round(TissueRadius/self._SpatialStep)*self._SpatialStep
        print('TissueRadius after rounding in steps',TissueRadius)

        for n in range(len(TissueRadius)-1):
            Mask=self.MakeSphericalMask(Radius=TissueRadius[n]*1e3,Center=CenterSkull)
            if n==1: #sanitycheck for the bonelocation
                AA=np.where(Mask)
                print(AA[2].min(), LocationBoneSteps)
                assert(AA[2].min()== LocationBoneSteps)
           
            BelowMask=self.MakeSphericalMask(Radius=TissueRadius[n+1]*1e3,Center=CenterSkull)

            Mask=np.logical_xor(Mask,BelowMask)
            if n==3: #the inner table
                self._MaterialMap[Mask]=2 
            else:
                self._MaterialMap[Mask]=n+1 
                
        #And we use the last below mask to make it the brain tissue
        self._MaterialMap[BelowMask]=4
        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[:,int(self._N2/2),:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()

            plt.ylim(42,25)
            plt.xlim(-10,10)

            
    def MakeSphericalMask(self, Radius=75,Center=[0,0,50]):
        #simple defintion of a focusing source centred in the domain, 
        #please note this is not a bullet-proof solution as it may not work for all cases
        XX,YY,ZZ=np.meshgrid(self._YDim,self._XDim,self._ZDim)#note we have to invert this because how meshgrid works
        Mask=(XX-Center[0])**2+(YY-Center[1])**2+(ZZ-Center[2])**2<=Radius**2
        return Mask
    
    
    def MakeCircularSource(self,Diameter):
        #simple defintion of a circular source centred in the domain
        XDim=np.arange(self._N1)*self._SpatialStep
        YDim=np.arange(self._N2)*self._SpatialStep
        XDim-=XDim.mean()
        YDim-=YDim.mean()
        XX,YY=np.meshgrid(XDim,YDim)
        MaskSource=(XX**2+YY**2)<=(Diameter/2.0)**2
        return (MaskSource*1.0).astype(np.uint32)
        
    def CreateRayleighH317Source(self,bSourceDisplacement=True,ramp_length=4,DepthFocalSpot=28.5):
        
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)
        
       
        InputPlaneData=np.load('H317-inputplane-%3.2f-%i-PPW.npz' % (DepthFocalSpot,self._basePPW))
        RadiusFace=InputPlaneData['xf'].max()
        nxf=len(self._XDim)
        nyf=len(self._YDim)
        xp,yp=np.meshgrid(self._XDim*1e-3,self._YDim*1e-3)
        self._SourceMapRayleigh=np.zeros((nxf,nyf),np.complex64)
        
        iX=np.argmin(np.abs(self._XDim*1e-3-InputPlaneData['xf'][0]))
        iY=np.argmin(np.abs(self._YDim*1e-3-InputPlaneData['yf'][0]))
        
        
        RegionMapFinal=xp**2+yp**2<=RadiusFace**2 #we select the circle on the incident field
        
        self._SourceMapRayleigh[iX:iX+InputPlaneData['xf'].size,iY:iY+InputPlaneData['yf'].size]=InputPlaneData['u0_input'][:,:,0]
        self._SourceMapRayleigh[RegionMapFinal==False]=0
        self._SourceMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        
        LocZ=self._PMLThickness
        
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
        
        self._Ox=np.zeros((self._N1,self._N2,self._N3))
        self._Oy=np.zeros((self._N1,self._N2,self._N3))
        self._Oz=np.zeros((self._N1,self._N2,self._N3))
        self._Oz[self._SourceMap>0]=1 #only Z has a value of 1
        
        self._weight_amplitudes=np.zeros(self._SourceMap.shape)
        self._weight_amplitudes[self._SourceMap>0]=1.0

        if bSourceDisplacement:
            self._Ox=np.zeros(self._SourceMap.shape)
            self._Ox[self._SourceMap>0]=1.0
            self._Oy=np.zeros(self._SourceMap.shape)
            self._Oz=np.zeros(self._SourceMap.shape)
            
        if self._bDisplay:
            plt.figure(figsize=(12,4))
            for n in range(1,4):
                plt.plot(TimeVectorSource*1e6,PulseSource[int(PulseSource.shape[0]/4)*n,:])
                plt.title('CW signal, example %i' %(n))
                
            plt.xlim(0,50)
                
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._SourceMap[:,:,LocZ],extent=[self._XDim.min(),self._XDim.max(),self._YDim.min(),self._YDim.max()])
            plt.subplot(1,2,2)
            plt.imshow(np.abs(self._SourceMapRayleigh),extent=[self._XDim.min(),self._XDim.max(),self._YDim.min(),self._YDim.max()])
            plt.colorbar()
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._SourceMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);

            plt.subplot(1,2,2)
            plt.imshow(self._SourceMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()]);

            
    def GetDisplacementFromFocusingMask(self,MaskSource,Center,LocZ=0):
        #simple defintion of a focusing source centred in the domain, 
        #please note this is not a bullet-proof solution as it may not work for all cases
        XDim=np.arange(self._N1)*self._SpatialStep
        YDim=np.arange(self._N2)*self._SpatialStep
        ZDim=np.arange(self._N3)*self._SpatialStep
        XDim-=XDim.mean()+Center[0]
        YDim-=YDim.mean()+Center[1]
        ZDim-=ZDim.mean()+Center[2]+LocZ*self._SpatialStep
            
        XX,YY,ZZ=np.meshgrid(YDim,XDim,ZDim)#note we have to invert this because how meshgrid works
        XX+=self._SpatialStep/2
        YY+=self._SpatialStep/2
        ZZ+=self._SpatialStep/2
                
        #since the sphere mask is 0-centred, the orientation vectors in each point is straighforward
        OxOyOz=np.vstack((-XX.flatten(),-YY.flatten(),-ZZ.flatten())).T
        #and we just normalize
        OxOyOz/=np.tile( np.linalg.norm(OxOyOz,axis=1).reshape(OxOyOz.shape[0],1),[1,3])
        Ox=OxOyOz[:,1].reshape(XX.shape) 
        Oy=OxOyOz[:,0].reshape(XX.shape)
        Oz=OxOyOz[:,2].reshape(XX.shape)    
        Ox[self._SourceMap==0]=0
        Oy[self._SourceMap==0]=0
        Oz[self._SourceMap==0]=0
        return Ox,Oy,Oz
        
        
    def CreateSensorMap(self,b3D=False):
        SensorMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        if b3D:
            SensorMap[self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness]=1
        else:
            SensorMap[self._PMLThickness:-self._PMLThickness,int(self._N2/2),self._PMLThickness:-self._PMLThickness]=1
        self._SensorMap=SensorMap
        print('Number of sensors ', self._SensorMap.sum())
        if self._bDisplay:
            plt.figure()
            plt.imshow(SensorMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray)
            plt.title('Sensor map location')
        
        
    def PlotWeightedAverags(self):
        if self._bDisplay:
            plt.figure(figsize=(12,12))
            plt.subplot(1,2,1)
            plt.imshow(self._weight_amplitudes[int(self._N1/2),:,:].T,cmap=plt.cm.jet,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.xlim(-10,10)
            plt.ylim(10,-10)

            plt.subplot(1,2,2)
            plt.imshow(self._weight_amplitudes[:,int(self._N2/2),:].T,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()]);
        
            
        
    def RUN_SIMULATION(self,COMPUTING_BACKEND=1,GPUName='SUPER',GPUNumber=0,SelMapsRMSPeakList=['Vx','Vy','Vz','Pressure'],bSourceDisplacement=True,bApplyCorrectionForDispersion=True,bUse_SPP=False,nSPP=1):
        MaterialList=self.ReturnArrayMaterial()

        if bSourceDisplacement:
            TypeSource=0 
            Ox=self._Ox*self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            Oy=self._Oy*self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            Oz=self._Oz*self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            PulseSource=self._PulseSource
        else:
            TypeSource=2 #stress source
            Ox=self._weight_amplitudes
            Oy=np.array([1])
            Oz=np.array([1])
            PulseSource=-self._PulseSource

        if bUse_SPP == False:
            self._Sensor,self._LastMap,self._DictPeakValue,self._InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                             self._MaterialMap,
                                                             MaterialList,
                                                             self._Frequency,
                                                             self._SourceMap,
                                                             PulseSource,
                                                             self._SpatialStep,
                                                             self._TimeSimulation,
                                                             self._SensorMap,
                                                             Ox=Ox,
                                                             Oy=Oy,
                                                             Oz=Oz,
                                                             NDelta=self._PMLThickness,
                                                             DT=self._TemporalStep,
                                                             ReflectionLimit=self._ReflectionLimit,
                                                             COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                             USE_SINGLE=True,
                                                             SelMapsRMSPeakList=SelMapsRMSPeakList,
                                                             SelMapsSensorsList=['Pressure'],
                                                             SelRMSorPeak=2,
                                                             DefaultGPUDeviceName=GPUName,
                                                             DefaultGPUDeviceNumber=GPUNumber,
                                                             AlphaCFL=1.0,
                                                             TypeSource=TypeSource,
                                                             QfactorCorrection=self._QfactorCorrection,
                                                             SensorSubSampling=self._SensorSubSampling,
                                                             SensorStart=self._SensorStart)
        else:
            t0=time.time()
           
            for N in range(nSPP):
                #ManualFraction=(self._GMapTotal>(float(N)/float(nSPP)))*1.0 # we will run for each of threshold values one by one
                CritN=-(float(N)/float(nSPP))
                SelZone=(self._GMapTotal>(float(N)/float(nSPP))) | ((self._GMapTotal>CritN) & (self._GMapTotal<0.0))
                ManualFraction=SelZone*1.0
                
                Sensor,self._LastMap,self._DictPeakValue,self._InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                         self._MaterialMap,
                                                         MaterialList,
                                                         self._Frequency,
                                                         self._SourceMap,
                                                         PulseSource,
                                                         self._SpatialStep,
                                                         self._TimeSimulation,
                                                         self._SensorMap,
                                                         Ox=Ox,
                                                         Oy=Oy,
                                                         Oz=Oz,
                                                         NDelta=self._PMLThickness,
                                                         DT=self._TemporalStep,
                                                         ReflectionLimit=self._ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelMapsRMSPeakList=SelMapsRMSPeakList,
                                                         SelMapsSensorsList=['Pressure'],
                                                         SelRMSorPeak=2,
                                                         DefaultGPUDeviceName=GPUName,
                                                         DefaultGPUDeviceNumber=GPUNumber,
                                                         AlphaCFL=1.0,
                                                         TypeSource=TypeSource,
                                                         QfactorCorrection=self._QfactorCorrection,
                                                         SensorSubSampling=self._SensorSubSampling,
                                                         SensorStart=self._SensorStart,
                                                         SPP_ZONES=1, # we run with 1 zone and with the partial zone
                                                         SPP_VolumeFraction=ManualFraction, 
                                                         SILENT=1)
                                             

                #we only need to track the sensors
                if N==0:
                    AccumZone=Sensor 
                else:
                    for k in AccumZone:
                        if k != 'time':
                            AccumZone[k]+=Sensor[k]
            for k in AccumZone:
                if k != 'time':
                    AccumZone[k]/=nSPP
            
            self._Sensor=AccumZone # Now we have finally the SPP-averaged data

            #clear_output(wait=True) #little function to clear a bit the excess of printing
            print('DONE SPP!, elapsed time (min)', (time.time()-t0)/60)
                
                

        if bApplyCorrectionForDispersion:
            CFLWater=self._TemporalStep/self.DominantMediumTemporalStep
            ExpectedError=np.polyval(self._DispersionCorrection,CFLWater)
            Correction=100.0/(100.0-ExpectedError)
            print('CFLWater only, ExpectedError, Correction', CFLWater,ExpectedError,Correction)
            for k in self._LastMap:
                self._LastMap[k]*=Correction
            for k in self._DictPeakValue:
                self._DictPeakValue[k]*=Correction
            for k in self._Sensor:
                if k=='time':
                    continue
                self._Sensor[k]*=Correction

    
    def CalculatePhaseData(self,b3D=False):
        if b3D:
            t0=time.time()
            self._PhaseMap=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapFourier=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapPeak=np.zeros((self._N1,self._N2,self._N3))
        else:
            self._PhaseMap=np.zeros((self._N1,self._N3))
            self._PressMapFourier=np.zeros((self._N1,self._N3))
            self._PressMapPeak=np.zeros((self._N1,self._N3))
        time_step = np.diff(self._Sensor['time']).mean() #remember the sensor time vector can be different from the input source
        
        assert((self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling))==0)

        freqs = np.fft.fftfreq(self._Sensor['time'].size, time_step)
        IndSpectrum=np.argmin(np.abs(freqs-self._Frequency)) # frequency entry closest to 700 kHz
        if np.isfortran(self._Sensor['Pressure']):
            self._Sensor['Pressure']=np.ascontiguousarray(self._Sensor['Pressure'])
        index=self._InputParam['IndexSensorMap']
        
        if b3D:
            nStep=100000
            for n in range(0,self._Sensor['Pressure'].shape[0],nStep):
                top=np.min([n+nStep,self._Sensor['Pressure'].shape[0]])
                FSignal=fft(self._Sensor['Pressure'][n:top,:],axis=1)
                k=np.floor(index[n:top]/(self._N1*self._N2)).astype(np.int64)
                j=index[n:top]%(self._N1*self._N2)
                i=j%self._N1
                j=np.floor(j/self._N1).astype(np.int64)
                FSignal=FSignal[:,IndSpectrum]
                if b3D==False:
                    assert(np.all(j==int(self._N2/2))) #all way up we specified the XZ plane at N2/2, this assert should pass
                pa= np.angle(FSignal)
                pp=np.abs(FSignal)
                
                self._PhaseMap[i,j,k]=pa
                self._PressMapFourier[i,j,k]=pp
                self._PressMapPeak[i,j,k]=self._Sensor['Pressure'][n:top,:].max(axis=1)
            self._InPeakValue=self._DictPeakValue['Pressure']
        else:
            FSignal=fft(self._Sensor['Pressure'],axis=1)
            index=self._InputParam['IndexSensorMap'].astype(np.int64)
            k=np.floor(index/(self._N1*self._N2)).astype(np.int64)
            j=index%(self._N1*self._N2)
            i=j%self._N1
            j=np.floor(j/self._N1).astype(np.int64)
            FSignal=FSignal[:,IndSpectrum]
            if b3D==False:
                assert(np.all(j==int(self._N2/2))) #all way up we specified the XZ plane at N2/2, this assert should pass
            pa= np.angle(FSignal)
            pp=np.abs(FSignal)
            self._PhaseMap[i,k]=pa
            self._PressMapFourier[i,k]=pp
            self._PressMapPeak[i,k]=self._Sensor['Pressure'].max(axis=1)
            self._InPeakValue=self._DictPeakValue['Pressure'][:,int(self._N2/2),:]
        
        self._PressMapFourier*=2/self._Sensor['time'].size
        if b3D:
            print('Elapsed time doing phase and amp extraction from Fourier (s)',time.time()-t0)
        
        
        
    def PlotData(self):
  
        if self._bDisplay:
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,1)
            plt.imshow(self._PressMapPeak.T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.title('BabelViscoFDTD peak amp.')
            plt.subplot(1,3,2)
            plt.imshow(self._PressMapFourier.T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.title('BabelViscoFDTD Fourier amp.')
            plt.subplot(1,3,3)
            plt.imshow(self._InPeakValue.T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()


        LineInPeak=self._InPeakValue[int(self._N1/2),:]/1e6
        LineFourierAmp=self._PressMapFourier[int(self._N1/2),:]/1e6
       

        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.plot(self._ZDim,LineFourierAmp)
            plt.plot(self._FZDim,FocusLine)
            plt.xlim(self._FZDim.min(),self._FZDim.max())
               
        print('Peak pressure BabelFDTD (MPa) = %3.2f' %  (np.max(self._PressMapPeak)/1e6))
        
    def PlotData3D(self):
        cx=np.argmin(np.abs(self._XDim))
        cy=np.argmin(np.abs(self._YDim))
        ind = np.unravel_index(np.argmax(self._PressMapPeak, axis=None), self._PressMapPeak.shape)
            
        if self._bDisplay:
            plt.figure(figsize=(12,6))
            plt.subplot(1,3,1)
            plt.imshow(self._PressMapPeak[:,cy,:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.title('BabelViscoFDTD peak amp.')
            plt.subplot(1,3,2)
            plt.imshow(self._PressMapFourier[:,cy,:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.title('BabelViscoFDTD Fourier amp.')
            plt.subplot(1,3,3)
            plt.imshow(self._InPeakValue[:,cy,:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.title('BabelViscoFDTD InPeak amp.')
            

           
        LineInPeak=self._InPeakValue[cx,cy,:]/1e6
        LineFourierAmp=self._PressMapFourier[cx,cy,:]/1e6
        

        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.plot(self._ZDim,LineFourierAmp)
                       
        print('Peak pressure BabelFDTD (MPa) = %3.2f' %  (np.max(self._PressMapPeak)/1e6))
        
    def ResamplingToFocusConditions(self,TESTNAME='test.h5',
                                    bReload=False,bSkipSaveAndReturnData=False):
        if bReload: #use with caution, mostly if some modification to the export of an existing simulation
            DictPeakValue=ReadFromH5py('..'+os.sep+'DATA'+os.sep+'ALL-'+TESTNAME+'.h5')
            PhaseMap=DictPeakValue.pop('PhaseMap')
            PressMapFourier=DictPeakValue.pop('PressMapFourier')
            PressMapPeak=DictPeakValue.pop('PressMapPeak')
        else:
            DictPeakValue=self._DictPeakValue
            PhaseMap=self._PhaseMap
            PressMapFourier=self._PressMapFourier
            PressMapPeak=self._PressMapPeak
        
        InterAllData={}
        FRefX,FRefY=np.meshgrid(self._FXDim,self._FZDim)
        FStep=self._FZDim[1]-self._FZDim[0]
        
        for k in DictPeakValue:
            if k=='Pressure':
                Interpolator=interpolate.interp2d(self._XDim,self._ZDim,PressMapFourier.T)
            else:
                Interpolator=interpolate.interp2d(self._XDim,self._ZDim,DictPeakValue[k][:,int(self._N2/2),:].T)
            InterAllData[k]=Interpolator(self._FXDim,self._FZDim).reshape(FRefX.shape)
            
        #For phase data, it is better to use nearest interpolator to avoid the phase wrap artifacts
        Xfo,Zfo=np.meshgrid(self._XDim,self._ZDim)
        InterpPhase=interpolate.NearestNDInterpolator(list(zip(Xfo.flatten(),Zfo.flatten())),PhaseMap.T.flatten())
        Xf,Zf=np.meshgrid(self._FXDim,self._FZDim)
        pInterpphase=InterpPhase(Xf.flatten(),Zf.flatten())
        pInterpphase=np.reshape(pInterpphase,FRefX.shape)
        
        if self._bDisplay:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(InterAllData['Pressure']/1e6,cmap=plt.cm.jet,
                       extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()],
                       vmax=self._REFERENCE['p_amp'].max()/1e6)
            plt.colorbar()
            plt.title('BabelViscoFDTD')
            plt.subplot(1,2,2)
            plt.imshow(self._REFERENCE['p_amp']/1e6,cmap=plt.cm.jet,extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar();
            plt.title('Reference')
            
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(pInterpphase,cmap=plt.cm.jet,extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('BabelViscoFDTD')
            plt.subplot(1,2,2)
            plt.imshow(self._REFERENCE['p_phase'],cmap=plt.cm.jet,extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar();
            plt.title('Reference')

        n=1
        if self._bDisplay:
            plt.figure(figsize=(16,8))
            for k in InterAllData:
                if k=='Pressure':
                    continue
                if len(InterAllData)>=7:
                    plt.subplot(3,3,n)
                else:
                    plt.subplot(2,3,n)
                n+=1
                plt.imshow(InterAllData[k],cmap=plt.cm.jet,extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
                plt.colorbar()
                plt.title(k)
            
            
        print('Peak pressure BabelFDTD (MPa) = %3.2f' %  (np.max(InterAllData['Pressure'])/1e6))
        
        DataToSave={}
        DataToSave['p_amp']=InterAllData['Pressure'].T
        DataToSave['p_phase']=pInterpphase.T
        for k in InterAllData:
            if k == 'Pressure':
                continue
            DataToSave[k]=InterAllData[k].T

        DataToSave['x_vec']=self._FXDim
        DataToSave['y_vec']=self._FZDim
        DataToSave['N1']=self._N1
        DataToSave['N2']=self._N2
        DataToSave['N3']=self._N3
        DataToSave['SpatialStep']=self._SpatialStep
        DataToSave['PMLThickness']=self._PMLThickness
        DataToSave['AdjustedCFL']=self._AdjustedCFL
        if not(self._GMapTotal is None):
            DataToSave['GMapTotal']=self._GMapTotal
        if  bSkipSaveAndReturnData==False:
            SaveToH5py(DataToSave,'..'+os.sep+'DATA'+os.sep+TESTNAME+'.h5')
        else:
            return DataToSave
        
        
    def ResamplingToFocusConditions3D(self,TESTNAME='test.h5',
                                    bReload=False,bSkipSaveAndReturnData=False,bUseCupyToInterpolate=True):
        
        DataToSave={}
        DataToSave['p_amp']=self._PressMapFourier
        DataToSave['p_phase']=self._PhaseMap
        DataToSave['MaterialMap']=self._MaterialMap
        DataToSave['Materials']=self.ReturnArrayMaterial()
        DataToSave['x_vec']=self._ZDim
        DataToSave['y_vec']=self._YDim
        DataToSave['z_vec']=self._XDim
        DataToSave['N1']=self._N1
        DataToSave['N2']=self._N2
        DataToSave['N3']=self._N3
        DataToSave['SpatialStep']=self._SpatialStep
        DataToSave['PMLThickness']=self._PMLThickness
        DataToSave['AdjustedCFL']=self._AdjustedCFL
       
        if  bSkipSaveAndReturnData==False:
            SaveToH5py(DataToSave,TESTNAME+'.h5')
        else:
            return DataToSave
        
