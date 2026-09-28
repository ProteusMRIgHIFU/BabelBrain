'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

'''
import os

import numpy as np
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple, SpeedofSoundWater
from stl import mesh

from TranscranialModeling.babel_integration.integration_templates.babel_integration_base import (
    RUN_SIM_BASE,
    BabelFTD_Simulations_BASE,
    Material,
    SimulationConditionsBASE,
    _rec_artifact,
)
from TranscranialModeling.tx_geometries import generate_focused_array_tx

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
  
    return ListPoints

def CreateSpreadFocus(DiameterFocalBeam=1.5e-3):
    BaseTriangle =  DiameterFocalBeam/2
    HeightTriangle = np.sin(np.pi/3)*DiameterFocalBeam
    ListPoints = [[0,HeightTriangle/2]]
    ListPoints += [[BaseTriangle,-HeightTriangle/2]]
    ListPoints += [[-BaseTriangle,-HeightTriangle/2]]
    ListPoints=np.array(ListPoints)
    return ListPoints

def shift_tx(tx,shift):
    tx['VertDisplay'][:,2] -= shift
    tx['center'][:,2] -= shift
    tx['elemcenter'][:,2] -= shift

class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self,**kargs):
        return self._BabelFTDSimClass(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                     **kargs)
        
    def RunCases(self,
                    XSteering=0.0,
                    YSteering=0.0,
                    ZSteering=0.0,
                    RotationZ=0.0,
                    MultiPoint=None,
                    **kargs):
        self._RotationZ=RotationZ
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
                 elements=[],
                 num_elements=0,
                 element_size=0,
                 Aperture=0,
                 FocalLength=0,
                 coordinate_system='',
                 **kargs):

        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._RotationZ=RotationZ
        self._elements=elements
        self._num_elements=num_elements
        self._original_element_size=element_size
        self._Aperture=Aperture
        self._focal_length=FocalLength
        self._coordinate_system=coordinate_system
        super().__init__(**kargs)

    def CreateSimConditions(self,**kargs):
        return self._SimConditionsClass(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    Aperture=self._Aperture, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                                    FocalLength=self._focal_length,
                                    elements=self._elements,
                                    num_elements=self._num_elements,
                                    element_size=self._original_element_size,
                                    coordinate_system=self._coordinate_system,
                                    **kargs)

    def AdjustMechanicalSettings(self,SkullMaskDataOrig,voxelS):
        pass

    def GenerateSTLTx(self,prefix):
        #we also export the STL of the Tx for display in Brainsight or 3D slicer
        affine=self._SkullMask.affine
        LocSpot=np.array(np.where(self._SkullMask.get_fdata(dtype=np.float32)==5.0)).flatten()

        for nt,st in enumerate(['VertDisplay','elemcenter']):
            TxVert=self._SIM_SETTINGS._TxOrig[st].T.copy()
            TxVert/=self._SIM_SETTINGS.SpatialStep
            TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])
            
            TxVert[2,:]=-TxVert[2,:]
            TxVert[0,:]+=LocSpot[0]
            TxVert[1,:]+=LocSpot[1]
            TxVert[2,:]+=LocSpot[2] - self._SIM_SETTINGS._TxMechanicalAdjustmentZ/self._SIM_SETTINGS.SpatialStep

            TxVert=np.dot(affine,TxVert)

            TxVert=TxVert.T[:,:3]

            if nt==0:
                TxStl = mesh.Mesh(np.zeros(self._SIM_SETTINGS._TxOrig['FaceDisplay'].shape[0]*2, dtype=mesh.Mesh.dtype))
                for i, f in enumerate(self._SIM_SETTINGS._TxOrig['FaceDisplay']):
                    TxStl.vectors[i*2][0] = TxVert[f[0],:]
                    TxStl.vectors[i*2][1] = TxVert[f[1],:]
                    TxStl.vectors[i*2][2] = TxVert[f[3],:]

                    TxStl.vectors[i*2+1][0] = TxVert[f[1],:]
                    TxStl.vectors[i*2+1][1] = TxVert[f[2],:]
                    TxStl.vectors[i*2+1][2] = TxVert[f[3],:]
                
                bdir=os.path.dirname(self._MASKFNAME)
                TxStl.save(bdir+os.sep+prefix+'Tx.stl')
                _rec_artifact(bdir+os.sep+prefix+'Tx.stl')
            else:
                self._TxElemCenters=TxVert
            

    def AddSaveDataSim(self,DataForSim):
        super().AddSaveDataSim(DataForSim)
        DataForSim['TransducerType']='DomePhasedArray'
        DataForSim['XSteering']=self._XSteering
        DataForSim['YSteering']=self._YSteering
        DataForSim['ZSteering']=self._ZSteering
        DataForSim['RotationZ']=self._RotationZ
        DataForSim['bDoRefocusing']=self._bDoRefocusing
        DataForSim['BasePhasedArrayProgrammingRefocusing']=self._SIM_SETTINGS.BasePhasedArrayProgrammingRefocusing
        DataForSim['BasePhasedArrayProgramming']=self._SIM_SETTINGS.BasePhasedArrayProgramming
    
class SimulationConditions(SimulationConditionsBASE):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''

    PPW_SURFACE = 9  # default points-per-wavelength for meshing the tx surface; overrideable by subclasses
    PPW_SURFACE_HIGH_RES = 20  # used for the special-cased 220 kHz high-res tx surface; overrideable by subclasses

    def __init__(self,FactorEnlarge = 1.0, #putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
                      Aperture=0.0, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                      FocalLength=0.0,
                      XSteering=0.0, #lateral steering
                      YSteering=0.0,
                      ZSteering=0.0,
                      RotationZ=0.0,#rotation of Tx over Z axis
                      elements=[],
                      num_elements=0,
                      element_size=0,
                      coordinate_system='',
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
        self._RotationZ=RotationZ
        self._elements=elements
        self._num_elements=num_elements
        self._original_element_size=element_size
        self._element_size=element_size*FactorEnlarge
        self._coordinate_system=coordinate_system

    def UpdateConditions(self, SkullMaskNii,AlphaCFL=1.0,bWaterOnly=False,
                         bForceHomogenousMedium=False,
                         BenchmarkTestFile='',
                         DomeType=False):
        super().UpdateConditions(SkullMaskNii,AlphaCFL=AlphaCFL,bWaterOnly=bWaterOnly,
                         bForceHomogenousMedium=bForceHomogenousMedium,
                         BenchmarkTestFile=BenchmarkTestFile,
                         DomeType=True)
        

    def GenTransducerGeom(self,PPWSurface=None,PPWSurfaceHighRes=None):
        if PPWSurface is None:
            PPWSurface = self.PPW_SURFACE
        if PPWSurfaceHighRes is None:
            PPWSurfaceHighRes = self.PPW_SURFACE_HIGH_RES
        if self._coordinate_system == 'spherical':
            element_positions = np.column_stack((self._elements["r"], np.deg2rad(self._elements["theta"]), np.deg2rad(self._elements["phi"])))
        else:
            element_positions = np.column_stack((self._elements["x"], self._elements["y"], self._elements["z"]))
        self._Tx = generate_focused_array_tx(element_positions, self._num_elements, self._Frequency, self._FocalLength, self._element_size, validate_elements=True, sos=SpeedofSoundWater(20.0),rotation_z=self._RotationZ, coordinate_sys=self._coordinate_system,show_plot=False,ppw_surface=PPWSurface)
        self._TxOrig = generate_focused_array_tx(element_positions, self._num_elements, self._Frequency, self._OrigFocalLength, self._original_element_size, validate_elements=True, sos=SpeedofSoundWater(20.0),rotation_z=self._RotationZ, coordinate_sys=self._coordinate_system,show_plot=False,ppw_surface=PPWSurface)

        shift_tx(self._Tx,self._FocalLength)
        shift_tx(self._TxOrig,self._OrigFocalLength)

        if self._Frequency == 220e3:
            self._TxHighRes = generate_focused_array_tx(element_positions, self._num_elements, self._Frequency, self._FocalLength, self._element_size, validate_elements=True, sos=SpeedofSoundWater(20.0),rotation_z=self._RotationZ, coordinate_sys=self._coordinate_system,show_plot=False,ppw_surface=PPWSurfaceHighRes)
            shift_tx(self._TxHighRes,self._FocalLength)
        else:
            self._TxHighRes=self._TxOrig
        
        # We use calibration per PPW to generate 1W per element    
        for tx in [self._Tx, self._TxOrig, self._TxHighRes]:
            tx["Amplitude1W"] = {
                "Rayleigh": 0.14475482330468514,
                "Visco": {
                    220000: {
                        6: 74065.04,
                        7: 79050.414,
                        8: 84021.836,
                        9: 88933.47,
                        10: 94068.0,
                        11: 91529.37,
                        12: 97344.266,
                    },
                    670000: {6: 166890.38},
                },
            }
        
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elements
        self.GenTransducerGeom()

        for k in ['center','elemcenter','VertDisplay']:
            self._Tx[k][:,0]+=self._TxMechanicalAdjustmentX
            self._Tx[k][:,1]+=self._TxMechanicalAdjustmentY
            self._Tx[k][:,2]+=self._TxMechanicalAdjustmentZ
            self._TxHighRes[k][:,0]+=self._TxMechanicalAdjustmentX
            self._TxHighRes[k][:,1]+=self._TxMechanicalAdjustmentY
            self._TxHighRes[k][:,2]+=self._TxMechanicalAdjustmentZ

     
        #we apply an homogeneous pressure 
       
        print('min,max Tx Z',self._Tx['center'][:,2].min(),self._Tx['center'][:,2].max())

        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        #we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming=np.zeros(self._Tx['NumberElems'],np.complex64)
        self.BasePhasedArrayProgrammingRefocusing=np.zeros(self._Tx['NumberElems'],np.complex64)

        Amplitude=1.0
        if 'Amplitude1W' in self._Tx:
            Amplitude=self._Tx['Amplitude1W']['Rayleigh']
            print('using 1W Rayleigh per element ampltiude',Amplitude)
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
                u0[nBase:nBase+self._Tx['elemdims']]=np.exp(1j*phi).astype(np.complex64)
                nBase+=self._Tx['elemdims']
        else:
             u0=(np.ones((self._Tx['center'].shape[0],1),np.float32)+ 1j*np.zeros((self._Tx['center'].shape[0],1),np.float32))
        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        yp,xp,zp=np.meshgrid(self._YDim,self._XDim,self._ZDim)
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u0*= self.AdjustWeightAmplitudes()*Amplitude

        u2=ForwardSimple(cwvnb_extlay,self._Tx['center'].astype(np.float32),self._Tx['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)*1.5e6 # in Pa
        
        self._u2RayleighField=u2

        
    def CreateSources(self,ramp_length=8):
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

        nBase=0
        nBaseVert=0
        Orig=[self._XDim[0],self._YDim[0],self._ZDim[0]]

        PulseSource = np.zeros((self._TxHighRes['NumberElems'],TimeVectorSource.shape[0]))

        AmplitudeCal=1.0
        if 'Amplitude1W' in self._Tx:
            print(self._Tx['Amplitude1W']['Visco'])
            AmplitudeCal=self._Tx['Amplitude1W']['Visco'][int(self._Frequency)][self._basePPW]
            print('Using amplitude for 1W',AmplitudeCal)

        for n in range(self._TxHighRes['NumberElems']):
            SelCenters=self._TxHighRes['center'][nBase:nBase+self._TxHighRes['elemdims'],:]
            SelCenters=np.vstack((self._TxHighRes['center'][nBase:nBase+self._TxHighRes['elemdims'],:],
                                self._TxHighRes['VertDisplay'][nBaseVert:nBase+self._TxHighRes['elemdims']*4,:]))
            
            IndX=np.round((SelCenters[:,0]-Orig[0])/self._SpatialStep).astype(int)
            IndY=np.round((SelCenters[:,1]-Orig[1])/self._SpatialStep).astype(int)
            IndZ=np.round((SelCenters[:,2]-Orig[2])/self._SpatialStep).astype(int)
            assert(np.all(IndX>=self._PMLThickness))
            assert(np.all(IndX<(self._N1-self._PMLThickness)))
            assert(np.all(IndY>=self._PMLThickness))
            assert(np.all(IndY<(self._N2-self._PMLThickness)))
            assert(np.all(IndZ>=self._PMLThickness))
            assert(np.all(IndZ<(self._N3-self._PMLThickness)))
            assert(np.all(self._SourceMap[IndX,IndY,IndZ]==0))
            self._SourceMap[IndX,IndY,IndZ]=n+1

            nBase+=self._TxHighRes['elemdims']
            nBaseVert+=self._TxHighRes['elemdims']*4

            PulseSource[n,:] = np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(self.BasePhasedArrayProgramming[n]))*AmplitudeCal
            PulseSource[n,:int(ramp_length_points)]*=ramp
            PulseSource[n,-int(ramp_length_points):]*=np.flip(ramp)

            
        self._PulseSource=PulseSource
        self._PulseAmplitude=AmplitudeCal
        
        ## Now we create the sources for back propagation
        
        self._PunctualSource=np.sin(2*np.pi*self._Frequency*TimeVectorSource).reshape(1,len(TimeVectorSource))
        self._PunctualSource[0,:int(ramp_length_points)]*=ramp
        self._PunctualSource[0,-int(ramp_length_points):]*=np.flip(ramp)
        
        self._SourceMapPunctual=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        LocForRefocusing=self._FocalSpotLocation.copy()
        self._SourceMapPunctual[LocForRefocusing[0],LocForRefocusing[1],LocForRefocusing[2]]=1

    def CreateSensorMap(self):
        '''
        Create the sensor map and back-propagation sensor map for the simulation.
        '''
        self._SensorMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        # for the back propagation, we only use the entering face
        self._SensorMapBackPropagation=np.zeros((self._N1,self._N2,self._N3),np.uint32)    
    
        self._SensorMap[self._PMLThickness:-self._PMLThickness,
                        self._PMLThickness:-self._PMLThickness,
                        self._ZSourceLocation+1:-self._PMLThickness]=1
        
        self._SensorMapBackPropagation=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        Orig=[self._XDim[0],self._YDim[0],self._ZDim[0]]
        self._IndexSensorsBack=[]
        for n in range(self._Tx['NumberElems']):
            center=self._Tx['elemcenter'][n,:]
            IndX=np.round((center[0]-Orig[0])/self._SpatialStep).astype(int)
            IndY=np.round((center[1]-Orig[1])/self._SpatialStep).astype(int)
            IndZ=np.round((center[2]-Orig[2])/self._SpatialStep).astype(int)
            self._SensorMapBackPropagation[IndX,IndY,IndZ]=1
            self._IndexSensorsBack.append((IndX,IndY,IndZ))


        
    def CalculatePhaseData(self,bRefocused=False,bDoRefocusing=True,bDoRefocusingVolume=False):
        #we overwrite to use a volume
        super().CalculatePhaseData(bRefocused=bRefocused,bDoRefocusing=bDoRefocusing,bDoRefocusingVolume=True)
        
    def BackPropagationRayleigh(self,deviceName='6800'):
        for n in range(self._Tx['NumberElems']):
            IndX,IndY,IndZ=self._IndexSensorsBack[n]
            u2back=self._PressMapFourierBack[IndX,IndY,IndZ]
            self.BasePhasedArrayProgrammingRefocusing[n]=np.conjugate(u2back)
            
        
    def CreateSourcesRefocus(self,ramp_length=8):
        #we create the list of functions sources taken from the Rayliegh incident field
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)

        PulseSource = np.zeros((self._TxHighRes['NumberElems'],TimeVectorSource.shape[0]))

        for n in range(self._TxHighRes['NumberElems']):

            PulseSource[n,:] = np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(self.BasePhasedArrayProgrammingRefocusing[n]))*self._PulseAmplitude
            PulseSource[n,:int(ramp_length_points)]*=ramp
            PulseSource[n,-int(ramp_length_points):]*=np.flip(ramp)
        
        self._PulseSourceRefocus=PulseSource

    def ReturnResults(self,bDoRefocusing=True,bUseRayleighForWater=False,bDoRefocusingVolume=False):
        return super().ReturnResults(bDoRefocusing=bDoRefocusing,bUseRayleighForWater=bUseRayleighForWater,bDoRefocusingVolume=True)
    
         
    def RUN_SIMULATION(self,bDoStressSource=False,SelRMSorPeak=1,bApplyCorrectionForDispersion=True,**kargs):
        super().RUN_SIMULATION(bDoStressSource=True,bApplyCorrectionForDispersion=False,SelRMSorPeak=1,**kargs)

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
