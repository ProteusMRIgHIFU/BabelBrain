'''
Pipeline to execute viscoleastic simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - May 19, 2022

'''
import numpy as np
from sys import platform
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from BabelViscoFDTD.H5pySimple import ReadFromH5py,SaveToH5py
from BabelViscoFDTD.PropagationModel import PropagationModel
from BabelViscoFDTD.tools.RayleighAndBHTE import InitCuda,InitOpenCL,SpeedofSoundWater,ForwardSimple
from scipy import ndimage
import nibabel
from nibabel import processing
from scipy import interpolate
from skimage.draw import circle_perimeter,disk
from skimage.transform import rotate
from skimage.measure import regionprops, regionprops_table, label
from scipy.io import loadmat, savemat
from stl import mesh
from pprint import pprint
import warnings
import time
import gc
import os
import pickle
try:
    import mkl_fft as fft
except:
    print('mkl_fft not available')
    from numpy import fft
    
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

PModel=PropagationModel()


## Global definitions

DbToNeper=1/(20*np.log10(np.exp(1)))

def FitSpeedCorticalShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. 
    FRef=np.array([270e3,836e3])
    Cs270=np.array([1577.0,1498.0,1313.0]).mean()
    Cs836=np.array([1758.0,1674.0,1545.0]).mean()
    CsRef=np.array([Cs270,Cs836])
    p=np.polyfit(FRef, CsRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitSpeedTrabecularShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. 
    FRef=np.array([270e3,836e3])
    Cs270=np.array([1227.0,1365.0,1200.0]).mean()
    Cs836=np.array([1574.0,1252.0,1327.0]).mean()
    CsRef=np.array([Cs270,Cs836])
    p=np.polyfit(FRef, CsRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitAttCorticalShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. 
    PichardoData=(57.0/.27 +373/0.836)/2
    return np.round(PichardoData*(frequency/1e6)*0.6) #temporary fix to test

def FitAttTrabecularShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. 
    PichardoData=(57.0/.27+373/0.836)/2
    return np.round(PichardoData*(frequency/1e6)*0.6) #temporary fix to test

def FitSpeedCorticalLong(frequency):
    #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. 
    FRef=np.array([270e3,836e3])
    ClRef=np.array([2448.0,2516])
    p=np.polyfit(FRef, ClRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitSpeedTrabecularLong(frequency):
    #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. 
    FRef=np.array([270e3,836e3])
    ClRef=np.array([2140.0,2300])
    p=np.polyfit(FRef, ClRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitAttCorticalLong(frequency):
    #from J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    JasaAtt1MHz=(2.15+1.67)/2*100
    return np.round(JasaAtt1MHz*(frequency/1e6)*0.6) #temporary fix to test

def FitAttTrabecularLong(frequency):
    #from J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    JasaAtt1MHz=1.5*100
    return np.round(JasaAtt1MHz*(frequency/1e6)*0.6) #temporary fix to test


MatFreq={}
for f in [250e3,500e3,700e3]:
    Material={}
    #Density (kg/m3), LongSoS (m/s), ShearSoS (m/s), Long Att (Np/m), Shear Att (Np/m)
    Material['Water']=     np.array([1000.0, 1500.0, 0.0   ,   0.0,                   0.0] )
    Material['SofTissue']= np.array([1000.0, 1500.0, 0.0   ,   1.0 *f/500e3,  0.0] )
    Material['Cortical']=  np.array([1896.5, FitSpeedCorticalLong(f), 
                                             FitSpeedCorticalShear(f),  
                                             FitAttCorticalLong(f)  , 
                                             FitAttCorticalShear(f)])
    Material['Trabecular']=np.array([1738.0, FitSpeedTrabecularLong(f),
                                             FitSpeedTrabecularShear(f),
                                             FitAttTrabecularLong(f) , 
                                             FitAttTrabecularShear(f)])
    Material['Skin']=      np.array([1090.0, 1610.0, 0.0   ,  2.3*f/500e3 , 0])
    Material['Brain']=     np.array([1040.0, 1546.0, 0.0   ,  3.45*f/500e3 , 0])

    MatFreq[f]=Material

pprint(MatFreq)

def SettingSmallestSOS(frequency):
    SelFreq=MatFreq[frequency]
    SoS=SelFreq['Water'][1]
    for k in SelFreq:
        if  SelFreq[k][1]<SoS:
            SoS=SelFreq[k][1]
        if SelFreq[k][2]>0 and SelFreq[k][2] < SoS:
            SoS=SelFreq[k][2]
    return SoS


def primeCheck(n):
    # 0, 1, even numbers greater than 2 are NOT PRIME
    if n==1 or n==0 or (n % 2 == 0 and n > 2):
        return False
    else:
        # Not prime if divisable by another number less
        # or equal to the square root of itself.
        # n**(1/2) returns square root of n
        for i in range(3, int(n**(1/2))+1, 2):
            if n%i == 0:
                return False
        return True
    
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
    

    BetaC = np.arange(Beta1+BetaStep/2,Beta1+BetaStep*(1/2 + nrstep),BetaStep)
    
    Ind=0

    SingElem = np.zeros((0,3))
    N = np.zeros((0,3))
    ds = np.zeros((0,1))

    VertDisplay=  np.zeros((0,3))
    FaceDisplay= np.zeros((0,4),np.int)

    for nr in range(len(BetaC)):

        Perim = np.sin(BetaC[nr])*Foc*2*np.pi

        nAlpha = np.ceil(Perim/lstep)
        sAlpha = 2*np.pi/nAlpha

        AlphaC = np.arange(sAlpha/2,sAlpha*(1/2 + nAlpha ),sAlpha)


        SingElem=np.vstack((SingElem,np.zeros((len(AlphaC),3))))
        N  = np.vstack((N,np.zeros((len(AlphaC),3))))
        ds = np.vstack((ds,np.zeros((len(AlphaC),1))))

        VertDisplay= np.vstack((VertDisplay,np.zeros((len(AlphaC)*4,3))))
        FaceDisplay= np.vstack((FaceDisplay,np.zeros((len(AlphaC),4),np.int)))


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
        TxStl = mesh.Mesh(np.zeros(TxP['FaceDisplay'].shape[0]*2, dtype=mesh.Mesh.dtype))
        for i, f in enumerate(TxP['FaceDisplay']):
            TxStl.vectors[i*2][0] = TxVert[f[0],:]
            TxStl.vectors[i*2][1] = TxVert[f[1],:]
            TxStl.vectors[i*2][2] = TxVert[f[3],:]

            TxStl.vectors[i*2+1][0] = TxVert[f[1],:]
            TxStl.vectors[i*2+1][1] = TxVert[f[2],:]
            TxStl.vectors[i*2+1][2] = TxVert[f[3],:]
        
#        TxStl.save('Tx_Ring_%i.stl' %(n))
            
            
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

def TWExpPhaseDistance(distance): #distance in mm
    from scipy.interpolate import interp1d
    ExpData=np.loadtxt('TWExperimentalPhaseProgramming.csv',skiprows=1,delimiter=',')
    assert(distance>=ExpData[:,0].min()  and distance<=ExpData[:,0].max())
    P1=interp1d(ExpData[:,0],ExpData[:,2])
    P2=interp1d(ExpData[:,0],ExpData[:,3])
    P3=interp1d(ExpData[:,0],ExpData[:,4])
    return np.array([0.0, P1(distance),P2(distance),P3(distance)])
##########################################

def SaveNiftiFlirt(nii,fn):
    nii.to_filename(fn)
    res = '%6.5f' % (np.array(nii.header.get_zooms()).mean())
    cmd='flirt -in "'+fn + '" -ref "'+ fn + '" -applyisoxfm ' +res + ' -nosearch -out "' +fn.split('__.nii.gz')[0]+'.nii.gz'+'"'
    print(cmd)
    assert(os.system(cmd)==0)
    os.remove(fn)

def RunCases(targets,deviceName='A6000',COMPUTING_BACKEND=1,
             ID='LIFU1-01',
             basedir='../LIFU Clinical Trial Data/Participants/',
             bTightNarrowBeamDomain=True,
             TxMechanicalAdjustmentX=0,
             TxMechanicalAdjustmentY=0,
             TxMechanicalAdjustmentZ=0,
             extrasuffix='',bDoRefocusing=False,
             ZSteering=0.0,
             bDisplay=True,
             bMinimalSaving=False,
             bForceRecalc=False,
             bRunThroughFile=False,
             bWaterOnly=False,
             basePPW=[9],
             Frequencies=[500e3]):
    OutNames=[]
    for target in targets:
        subsamplingFactor=2 # Brainsight can't handle very large 3D files with high res, so we need to 
        #sub sample when save the final results.
        for Frequency in Frequencies:
            alignments=['']
            fstr='_%ikHz_' %(int(Frequency/1e3))
           
            AlphaCFL=0.5
            for PPW in basePPW:
                ppws='%iPPW_' % PPW
                for alignment in alignments:
                    if target=='V1' and alignment=='_unaligned':
                        OrientationTx='Y'
                    else:
                        OrientationTx='Z'
                    prefix=basedir+ID+os.sep
                    MASKFNAME=prefix+target+alignment+fstr+ppws+ 'BabelViscoInput.nii.gz'
                    print (MASKFNAME)
                    for noshear in [False]:
                        listAdjust=[0.0]
                        for AdjustDepthFocalSpot in listAdjust:

                            if noshear:
                                snoshear='_NoShear_'
                            else:
                                snoshear=''
                            if AdjustDepthFocalSpot==7.0:
                                sadjust='_FocalSpotMoved_'
                            else:
                                sadjust=''
                            sFreq='%ikHz_' %(int(Frequency/1e3))
                            outName=target+alignment+fstr+ppws+sadjust+snoshear+extrasuffix
                            bdir=os.path.dirname(MASKFNAME)
                            cname=bdir+os.sep+outName+'DataForSim.h5'
                            print(cname)
                            if os.path.isfile(cname)and not bForceRecalc:
                                print('*'*50)
                                print (' Skipping '+ cname)
                                print('*'*50)
                                OutNames.append(bdir+os.sep+outName+'DataForSim.h5')
                                continue
                                
                            if bRunThroughFile:
                                inparams={}
                                inparams['targets']=targets
                                inparams['deviceName']=deviceName
                                inparams['COMPUTING_BACKEND']=COMPUTING_BACKEND
                                inparams['ID']=ID
                                inparams['bTightNarrowBeamDomain']=bTightNarrowBeamDomain
                                inparams['TxMechanicalAdjustmentX']=TxMechanicalAdjustmentX
                                inparams['TxMechanicalAdjustmentY']=TxMechanicalAdjustmentY
                                inparams['TxMechanicalAdjustmentZ']=TxMechanicalAdjustmentZ
                                inparams['extrasuffix']=extrasuffix
                                inparams['bDoRefocusing']=bDoRefocusing
                                inparams['ZSteering']=ZSteering
                                inparams['bDisplay']=bDisplay
                                inparams['bMinimalSaving']=bMinimalSaving
                                inparams['bForceRecalc']=bForceRecalc
                                inparams['bRunThroughFile']=False
                                
                                with open('inputforsim','wb') as f:
                                    pickle.dump(inparams,f)
                                cmd='python RunSimViaFile.py'
                                res=os.system(cmd)
                                if res !=0:
                                    raise SystemError("Something didn't work when trying to run simulation")
                                with open('outputrunfromfile','rb') as f:
                                    result=pickle.load(f)
                                print('Done for ',result['ofname'])
                                OutNames+=result['ofname']
                            else:

                                print('*'*50)
                                print(target,'noshear',noshear,'AdjustDepthFocalSpot',AdjustDepthFocalSpot)
                                TestClass=BabelFTD_Simulations(MASKFNAME=MASKFNAME,
                                                               bNoShear=noshear,
                                                               bTightNarrowBeamDomain=bTightNarrowBeamDomain,
                                                               Frequency=Frequency,
                                                               basePPW=PPW,
                                                               AlphaCFL=AlphaCFL,
                                                               bWaterOnly=bWaterOnly,
                                                               TxMechanicalAdjustmentX=TxMechanicalAdjustmentX,
                                                               TxMechanicalAdjustmentY=TxMechanicalAdjustmentY,
                                                               TxMechanicalAdjustmentZ=TxMechanicalAdjustmentZ,
                                                               bDoRefocusing=bDoRefocusing,
                                                               ZSteering=ZSteering,
                                                               bDisplay=bDisplay)
                                print('  Step 1')

                                #with suppress_stdout():
                                TestClass.Step1_InitializeConditions(AdjustDepthFocalSpot=AdjustDepthFocalSpot,
                                                                     OrientationTx=OrientationTx)
                                print('  Step 2')
                                TestClass.Step2_CalculateRayleighFieldsForward(prefix=outName,
                                                                               deviceName=deviceName,
                                                                               bSkipSavingSTL= bMinimalSaving)
                                
                                print('  Step 3')
                                TestClass.Step3_CreateSourceSignal_and_Sensor()
                                print('  Step 4')
                                TestClass.Step4_Run_Simulation(GPUName=deviceName,COMPUTING_BACKEND=COMPUTING_BACKEND)
                                print('  Step 5')
                                TestClass.Step5_ExtractPhaseDataForwardandBack()
                                if bDoRefocusing:

                                    print('  Step 6')
                                    TestClass.Step6_BackPropagationRayleigh(deviceName=deviceName)
                                    print('  Step 7')
                                    TestClass.Step7_Run_Simulation_Refocus(GPUName=deviceName,COMPUTING_BACKEND=COMPUTING_BACKEND)
                                    print('  Step 8')
                                    TestClass.Step8_ExtractPhaseDataRefocus()
                                print('  Step 9')
                                TestClass.Step9_PrepAndPlotData()
                                print('  Step 10')
                                oname=TestClass.Step10_GetResults(prefix=outName,subsamplingFactor=subsamplingFactor,
                                                                 bMinimalSaving=bMinimalSaving)
                                OutNames.append(oname)
    return OutNames
 
    
def PlotFinalAnalysis(basedir='../LIFU Clinical Trial Data/Participants/',
                      Target='RSTN_Skin',
                      ID='LIFU1-01',
                      PPW=9,
                      DistanceFromSkin=50):
    plt.close('all')
    Water=ReadFromH5py(basedir+ID+os.sep+Target+'_500kHz_%iPPW__Water_DataForSim.h5' % (PPW))
    Skull=ReadFromH5py(basedir+ID+os.sep+Target+'_500kHz_%iPPW_DataForSim.h5' % (PPW))

    LocTarget=Skull['TargetLocation']
    print(LocTarget)
    
    for d in [Water,Skull]:
        for t in ['p_amp','MaterialMap']:
            d[t]=np.ascontiguousarray(np.flip(d[t],axis=2))
    
    Water['z_vec']-=0.14
    Skull['z_vec']-=0.14
    Water['z_vec']*=1e3
    Skull['z_vec']*=1e3
    Skull['x_vec']*=1e3
    Skull['y_vec']*=1e3
    Skull['MaterialMap'][Skull['MaterialMap']==3]=2
    Skull['MaterialMap'][Skull['MaterialMap']==4]=3
    
    DensityMap=Water['Material'][:,0][Water['MaterialMap']]
    SoSMap=    Water['Material'][:,1][Water['MaterialMap']]
    IWater=Water['p_amp']**2/2/DensityMap/SoSMap/1e4
    
    DensityMap=Skull['Material'][:,0][Skull['MaterialMap']]
    SoSMap=    Skull['Material'][:,1][Skull['MaterialMap']]
    ISkull=Skull['p_amp']**2/2/DensityMap/SoSMap/1e4
    
    ISkull/=IWater[Skull['MaterialMap']==3].max()
    IWater/=IWater[Skull['MaterialMap']==3].max()

    #these are the dimensions of the Tx
    Foc=63.2
    Diam=64

    DistanceToTarget=DistanceFromSkin*1e3

    Factor=IWater[Skull['MaterialMap']==3].max()/ISkull[Skull['MaterialMap']==3].max()
    print('*'*40+'\n'+'*'*40+'\n'+'Correction Factor for Isppa',Factor,'\n'+'*'*40+'\n'+'*'*40+'\n')

    plt.figure(figsize=(8,4))
    dz=np.diff(Skull['z_vec']).mean()
    Zvec=Skull['z_vec'].copy()
    Zvec-=Zvec[LocTarget[2]]
    Zvec+=DistanceToTarget
    XX,ZZ=np.meshgrid(Skull['x_vec'],Zvec)
    ax=plt.subplot(1,2,1)
    plt.contourf(XX,ZZ,ISkull[:,LocTarget[1],:].T*Factor*20,np.arange(2,22,2),cmap=plt.cm.jet)
    h=plt.colorbar()
    h.set_label('$I_{\mathrm{SPPA}}$ (W/cm$^2$)')
    plt.contour(XX,ZZ,Skull['MaterialMap'][:,LocTarget[1],:].T,[0,1,2,3], cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.set_xlabel('X mm')
    ax.set_ylabel('Z mm')
    ax.invert_yaxis()
    plt.plot(0,DistanceToTarget,'+k',markersize=18)
    ax=plt.subplot(1,2,2)
    plt.contourf(XX,ZZ,ISkull[LocTarget[0],:,:].T*Factor*20,np.arange(2,22,2),cmap=plt.cm.jet)
    plt.colorbar()
    plt.contour(XX,ZZ,Skull['MaterialMap'][LocTarget[0],:,:].T,[0,1,2,3], cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.set_xlabel('Y mm')
    ax.set_ylabel('Z mm')
    ax.invert_yaxis()
    plt.plot(0,DistanceToTarget,'+k',markersize=18)
    #plt.savefig('BeamProfileSkinRight.png',dpi=150,bbox_inches='tight')
    plt.title('MAIN SIMULATION RESULTS, adjust steering and lateral mechanical corrections') 
    
    IWaterBrain=IWater.copy()
    IWaterBrain[Skull['MaterialMap']<3]=0
    LocInWater=np.array(np.where(IWaterBrain==IWaterBrain.max())).flatten()
    print('LocInWater',LocInWater)

    plt.figure(figsize=(8,4))
    ax=plt.subplot(1,2,1)
    plt.contourf(XX,ZZ,IWater[:,LocInWater[1],:].T*20,np.arange(2,22,2),cmap=plt.cm.jet)
    h=plt.colorbar()
    h.set_label('$I_{\mathrm{SPPA}}$ (W/cm$^2$)')
    plt.contour(XX,ZZ,Skull['MaterialMap'][:,LocInWater[1],:].T,[0,1,2,3], cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.set_xlabel('X mm')
    ax.set_ylabel('Z mm')
    ax.invert_yaxis()
    plt.plot(0,DistanceToTarget,'+k',markersize=18)
    ax=plt.subplot(1,2,2)
    plt.contourf(XX,ZZ,IWater[LocInWater[1],:,:].T*20,np.arange(2,22,2),cmap=plt.cm.jet)
    h=plt.colorbar()
    h.set_label('$I_{\mathrm{SPPA}}$ (W/cm$^2$)')
    plt.contour(XX,ZZ,Skull['MaterialMap'][LocInWater[0],:,:].T,[0,1,2,3], cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.set_xlabel('Y mm')
    ax.set_ylabel('Z mm')
    ax.invert_yaxis()
    plt.plot(0,DistanceToTarget,'+k',markersize=18)
    #plt.savefig('BeamProfileWaterSkinRight.png',dpi=150,bbox_inches='tight')

    IWaterLine=((Skull['MaterialMap']==3)*IWater)[LocTarget[0],LocTarget[1],:]
    ISkullLine=((Skull['MaterialMap']==3)*ISkull)[LocInWater[0],LocInWater[1],:]

    plt.rcParams['font.size']=14
    plt.figure(figsize=(8,4))
    plt.plot(Zvec,IWaterLine*20)
    plt.plot(Zvec,ISkullLine*20)
    plt.plot(Zvec,ISkullLine*Factor*20)
    plt.legend(['Water','Skull','Skull-Compensated'])
    plt.title('Correction Factor ' +str(Factor),fontsize=14) 
    plt.plot([DistanceToTarget,DistanceToTarget],[0,20],':')
    plt.xlabel('Distance from out plane (mm)')
    plt.ylabel('$I_{\mathrm{SPPA}}$ (W/cm$^2$)')
    plt.xlim(0,90)
########################################################
########################################################

class BabelFTD_Simulations(object):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,MASKFNAME='4007/4007_keep/m2m_4007_keep/BabelViscoInput.nii.gz',
                 Frequency=250e3,
                 bDisplay=True,
                 basePPW=9,
                 AlphaCFL=1.0,
                 bNoShear=False,
                 pressure=50e3,
                 PadForRayleigh=12,
                 bTightNarrowBeamDomain=False, #if this set, simulations will be done only accross a section area that follows the acoustic beam, this is useful to reduce computational costs
                 zLengthBeyonFocalPointWhenNarrow=4e-2,
                 ZSteering=0.0,
                 TxMechanicalAdjustmentX=0,
                 TxMechanicalAdjustmentY=0,
                 TxMechanicalAdjustmentZ=0,
                 bDoRefocusing=True,
                 bWaterOnly=False):
        self._MASKFNAME=MASKFNAME
        
        if bNoShear:
            self._Shear=0.0
        else:
            self._Shear=1.0

        self._basePPW=basePPW
        
        self._AlphaCFL=AlphaCFL
        self._bDisplay=bDisplay
        
        self._Frequency=Frequency
        self._pressure=pressure
        self._PadForRayleigh=PadForRayleigh
        self._bWaterOnly=bWaterOnly
        self._bTightNarrowBeamDomain=bTightNarrowBeamDomain
        self._zLengthBeyonFocalPointWhenNarrow=zLengthBeyonFocalPointWhenNarrow
        self._TxMechanicalAdjustmentX=TxMechanicalAdjustmentX
        self._TxMechanicalAdjustmentY=TxMechanicalAdjustmentY
        self._TxMechanicalAdjustmentZ=TxMechanicalAdjustmentZ
        self._ZSteering=ZSteering
        self._bDoRefocusing=bDoRefocusing


    def Step1_InitializeConditions(self,AdjustDepthFocalSpot=0.,OrientationTx='Z'): #in case it is desired to move up or down in the Z direction the focal spot
        self._SkullMask=nibabel.load(self._MASKFNAME)
        SkullMaskDataOrig=self._SkullMask.get_fdata()
        voxelS=np.array(self._SkullMask.header.get_zooms())*1e-3
        Dims=np.array(SkullMaskDataOrig.shape)*voxelS
        if self._Frequency==250e3:
            SensorSubSampling=4
        else:
            #SensorSubSampling=4
            #SensorSubSampling=9
            SensorSubSampling=1
            
        self._SIM_SETTINGS = SimulationConditions(baseMaterial=Material['Water'],
                                basePPW=self._basePPW,
                                Frequency=self._Frequency,
                                PaddingForKArray=0,
                                bDisplay=self._bDisplay, 
                                DimDomain=Dims,
                                SensorSubSampling=SensorSubSampling,
                                SourceAmp=self._pressure,
                                AdjustDepthFocalSpot=AdjustDepthFocalSpot,
                                OrientationTx=OrientationTx,
                                bTightNarrowBeamDomain=self._bTightNarrowBeamDomain,
                                zLengthBeyonFocalPointWhenNarrow=self._zLengthBeyonFocalPointWhenNarrow,
                                TxMechanicalAdjustmentX=self._TxMechanicalAdjustmentX,
                                TxMechanicalAdjustmentY=self._TxMechanicalAdjustmentY,
                                TxMechanicalAdjustmentZ=self._TxMechanicalAdjustmentZ,
                                ZSteering=self._ZSteering,
                                DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721])
        for k in ['Skin','Cortical','Trabecular','Brain']:
            SelM=MatFreq[self._Frequency][k]
            self._SIM_SETTINGS.AddMaterial(SelM[0], #den
                                           SelM[1],
                                           SelM[2]*self._Shear,
                                           SelM[3],
                                           SelM[4]*self._Shear) #the attenuation came for 500 kHz, so we adjust with the one being used
        
        self._SIM_SETTINGS.UpdateConditions(self._SkullMask,AlphaCFL=self._AlphaCFL,bWaterOnly=self._bWaterOnly)
        gc.collect()
        
    def Step2_CalculateRayleighFieldsForward(self,prefix='',deviceName='6800',bSkipSavingSTL=False):
        #we use Rayliegh to forward propagate until a plane on top the skull, this plane will be used as a source in BabelVisco
        self._SIM_SETTINGS.CalculateRayleighFieldsForward(deviceName=deviceName)
        if bSkipSavingSTL ==False:
            n=1
            for VertDisplay,FaceDisplay in zip(self._SIM_SETTINGS._TxRC['RingVertDisplay'],
                                    self._SIM_SETTINGS._TxRC['RingFaceDisplay']):
                #we also export the STL of the Tx for display in Brainsight or 3D slicer
                TxVert=VertDisplay.T.copy()
                TxVert/=self._SIM_SETTINGS.SpatialStep
                TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])
                affine=self._SkullMask.affine

                LocSpot=np.array(np.where(self._SkullMask.get_fdata()==5.0)).flatten()

                TxVert[2,:]=-TxVert[2,:]
                TxVert[0,:]+=LocSpot[0]
                TxVert[1,:]+=LocSpot[1]
                TxVert[2,:]+=LocSpot[2]+self._SIM_SETTINGS._FocalLength/self._SIM_SETTINGS.SpatialStep

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
        
        gc.collect()
        

    def Step3_CreateSourceSignal_and_Sensor(self):
        self._SIM_SETTINGS.CreateSources()
        gc.collect()
        self._SIM_SETTINGS.CreateSensorMap()
        gc.collect()

    def Step4_Run_Simulation(self,GPUName='GP100',bApplyCorrectionForDispersion=True,COMPUTING_BACKEND=1):
        SelMapsRMSPeakList=['Pressure']
        self._SIM_SETTINGS.RUN_SIMULATION(GPUName=GPUName,SelMapsRMSPeakList=SelMapsRMSPeakList,
                                          bApplyCorrectionForDispersion=bApplyCorrectionForDispersion,
                                          COMPUTING_BACKEND=COMPUTING_BACKEND,
                                          bDoRefocusing=self._bDoRefocusing)
        gc.collect()

    def Step5_ExtractPhaseDataForwardandBack(self):
        self._SIM_SETTINGS.CalculatePhaseData(bDoRefocusing=self._bDoRefocusing)
        gc.collect()
        #self._SIM_SETTINGS.PlotResultsPlanePartial()
        
        
    def Step6_BackPropagationRayleigh(self,deviceName='6800'):
        self._SIM_SETTINGS.BackPropagationRayleigh(deviceName=deviceName)
        gc.collect()
        self._SIM_SETTINGS.CreateSourcesRefocus()
        gc.collect()
        
    def Step7_Run_Simulation_Refocus(self,GPUName='GP100',COMPUTING_BACKEND=1,bApplyCorrectionForDispersion=True):
        SelMapsRMSPeakList=['Pressure']
        self._SIM_SETTINGS.RUN_SIMULATION(GPUName=GPUName,
                                          SelMapsRMSPeakList=SelMapsRMSPeakList,
                                          bApplyCorrectionForDispersion=bApplyCorrectionForDispersion,
                                          bRefocused=True,COMPUTING_BACKEND=COMPUTING_BACKEND)
        gc.collect()
    def Step8_ExtractPhaseDataRefocus(self):
        self._SIM_SETTINGS.CalculatePhaseData(bRefocused=True)
        gc.collect()
        
    def Step9_PrepAndPlotData(self):
        self._SIM_SETTINGS.PlotResultsPlane(bDoRefocusing=self._bDoRefocusing)
        gc.collect()
        
    def Step10_GetResults(self,prefix='',subsamplingFactor=1,bMinimalSaving=False):
        ss=subsamplingFactor
        if self._bWaterOnly:
            waterPrefix='_Water_'
        else:
            waterPrefix=''
        RayleighWater,RayleighWaterOverlay,\
            FullSolutionPressure,\
            FullSolutionPressureRefocus,\
            DataForSim= self._SIM_SETTINGS.ReturnResults(bDoRefocusing=self._bDoRefocusing)
        affine=self._SkullMask.affine.copy()
        affine[0:3,0:3]=affine[0:3,0:3] @ (np.eye(3)*subsamplingFactor)
        bdir=os.path.dirname(self._MASKFNAME)
        if bMinimalSaving==False:
            nii=nibabel.Nifti1Image(RayleighWaterOverlay[::ss,::ss,::ss],affine=affine)
            SaveNiftiFlirt(nii,bdir+os.sep+prefix+waterPrefix+'RayleighFreeWaterWOverlay__.nii.gz')

            nii=nibabel.Nifti1Image(RayleighWater[::ss,::ss,::ss],affine=affine)
            SaveNiftiFlirt(nii,bdir+os.sep+prefix+waterPrefix+'RayleighFreeWater__.nii.gz')
            if self._bDoRefocusing:
                nii=nibabel.Nifti1Image(FullSolutionPressureRefocus[::ss,::ss,::ss],affine=affine)
                SaveNiftiFlirt(nii,bdir+os.sep+prefix+waterPrefix+'FullElasticSolutionRefocus__.nii.gz')
                
        nii=nibabel.Nifti1Image(FullSolutionPressure[::ss,::ss,::ss],affine=affine)
        SaveNiftiFlirt(nii,bdir+os.sep+prefix+waterPrefix+'FullElasticSolution__.nii.gz')
        
        if subsamplingFactor>1:
            kt = ['p_amp','MaterialMap']
            if self._bDoRefocusing:
                kt.append('p_amp_refocus')
            for k in kt:
                DataForSim[k]=DataForSim[k][::ss,::ss,::ss]
            for k in ['x_vec','y_vec','z_vec']:
                DataForSim[k]=DataForSim[k][::ss]
            DataForSim['SpatialStep']*=ss
            DataForSim['TargetLocation']=np.round(DataForSim['TargetLocation']/ss).astype(int)
            
        DataForSim['BasePhasedArrayProgrammingRefocusing']=self._SIM_SETTINGS.BasePhasedArrayProgrammingRefocusing
        DataForSim['BasePhasedArrayProgramming']=self._SIM_SETTINGS.BasePhasedArrayProgramming
        
        DataForSim['TxMechanicalAdjustmentX']=self._TxMechanicalAdjustmentX
        DataForSim['TxMechanicalAdjustmentY']=self._TxMechanicalAdjustmentY
        DataForSim['TxMechanicalAdjustmentZ']=self._TxMechanicalAdjustmentZ
        DataForSim['ZSteering']=self._ZSteering
        
        DataForSim['affine']=affine
        
        ###
        
        FocIJK=np.ones((4,1))
        DataMaskOrig=self._SkullMask.get_fdata()
        DataMask=np.flip(DataMaskOrig,axis=2)
        FocIJK[:3,0]=np.array(np.where(DataMask==5)).flatten()
        
        VoxelSize=self._SkullMask.header.get_zooms()[0]*1e-3
        LineOfSight=DataMask[int(FocIJK[0,0]),int(FocIJK[1,0]),:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (FocIJK[2,0]-StartSkin)*VoxelSize
        print('DistanceFromSkin',np.round(DistanceFromSkin*1e3,2))
        
        DataForSim['DistanceFromSkin']=DistanceFromSkin
        
        FocIJK[:3,0]=np.array(np.where(DataMaskOrig==5)).flatten()
        
        FocXYZ=self._SkullMask.affine@FocIJK
        FocIJKAdjust=FocIJK.copy()
        #we adjust in steps
        FocIJKAdjust[0,0]+=self._TxMechanicalAdjustmentX*1e3/self._SkullMask.header.get_zooms()[0]
        FocIJKAdjust[1,0]+=self._TxMechanicalAdjustmentY*1e3/self._SkullMask.header.get_zooms()[1]
        #
        FocXYZAdjust=self._SkullMask.affine@FocIJKAdjust
        AdjustmentInRAS=(FocXYZ-FocXYZAdjust).flatten()[:3]
        DataForSim['AdjustmentInRAS']=AdjustmentInRAS
        print('Adjustment in RAS - T1W space',AdjustmentInRAS)
        
        sname=bdir+os.sep+prefix+waterPrefix+'DataForSim.h5'
        SaveToH5py(DataForSim,sname)
        gc.collect()
        
        return sname

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
                '$^*$Spatial step chosen to produce peak pressure amplitude ~2% compared to reference simulation (FOCUS).'\
                
        return String
    
########################################################
########################################################
class SimulationConditions(object):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,baseMaterial=Material['Water'],
                      basePPW=9,
                      PMLThickness = 12, # grid points for perect matching layer, HIGHLY RECOMMENDED DO NOT CHANGE THIS SIZE 
                      ReflectionLimit= 1e-5, #DO NOT CHANGE THIS
                      DimDomain =  np.array([0.07,0.07,0.12]),
                      SensorSubSampling=6,
                      NumberCyclesToTrackAtEnd=2,
                      SourceAmp=60e3, # kPa
                      Frequency=500e3,
                      FactorEnlarge = 2, #putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
                      Aperture=64e-3, # m, aperture of the Tx, used to calculated cross section area entering the domain
                      FocalLength=63.2e-3,
                      NaturalOutPlaneDistance =  52.4e-3, #distance from out of plane to focal spot
                      InDiameters= np.array([0.0    , 32.8e-3, 46e-3,   55.9e-3]), #inner diameter of rings
                      OutDiameters=np.array([32.8e-3, 46e-3,   55.9e-3, 64e-3]), #outer diameter of rings
                      OrientationTx='Z',
                      PaddingForKArray=0,
                      PaddingForRayleigh=12,
                      QfactorCorrection=True,
                      bDisplay=True,
                      bTightNarrowBeamDomain = False,
                      zLengthBeyonFocalPointWhenNarrow=4e-2,
                      AdjustDepthFocalSpot=0., #in case it is desired to move up or down in the Z direction the focal spot, useful when targeting very superficial targets, in mm
                      TxMechanicalAdjustmentX =0, # in case we want to move mechanically the Tx (useful when targeting shallow locations such as M1 and we want to evaluate if an small mechnical adjustment can ensure focusing)
                      TxMechanicalAdjustmentY =0, # in case we want to move mechanically the Tx (useful when targeting shallow locations such as M1 and we want to evaluate if an small mechnical adjustment can ensure focusing)
                      TxMechanicalAdjustmentZ =0, # in case we want to move mechanically the Tx (useful when targeting shallow locations such as M1 and we want to evaluate if an small mechnical adjustment can ensure focusing)
                      ZSteering=0.0, # steering
                      DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721]):  #coefficients to correct for values lower of CFL =1.0 in wtaer conditions.
        self._Materials=[[baseMaterial[0],baseMaterial[1],baseMaterial[2],baseMaterial[3],baseMaterial[4]]]
        self._basePPW=basePPW
        self._PMLThickness=PMLThickness
        self._ReflectionLimit=ReflectionLimit
        self._ODimDomain =DimDomain 
        self._SensorSubSampling=SensorSubSampling
        self._NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd
        self._TemporalStep=0.
        self._N1=0
        self._N2=0
        self._N3=0
        self._FactorConvPtoU=baseMaterial[0]*baseMaterial[1]
        self._SourceAmpPa=SourceAmp
        self._SourceAmpDisplacement=SourceAmp/self._FactorConvPtoU
        self._Frequency=Frequency
        self._weight_amplitudes=1.0
        self._PaddingForKArray=PaddingForKArray
        self._PaddingForRayleigh=PaddingForRayleigh
        self._QfactorCorrection=QfactorCorrection
        self._bDisplay=bDisplay
        self._DispersionCorrection=DispersionCorrection
        self._AdjustDepthFocalSpot=AdjustDepthFocalSpot
        self._FactorEnlarge=FactorEnlarge
        self._OrigAperture=Aperture
        self._OrigFocalLength=FocalLength
        self._OrigInDiameters=InDiameters
        self._OrigOutDiameters=OutDiameters
        self._Aperture=Aperture*FactorEnlarge
        self._FocalLength=FocalLength*FactorEnlarge
        self._InDiameters=InDiameters*FactorEnlarge
        self._OutDiameters=OutDiameters*FactorEnlarge
        self._NaturalOutPlaneDistance=NaturalOutPlaneDistance
        self._OrientationTx=OrientationTx
        self._bTightNarrowBeamDomain=bTightNarrowBeamDomain
        self._zLengthBeyonFocalPointWhenNarrow=zLengthBeyonFocalPointWhenNarrow
        self._TxMechanicalAdjustmentX=TxMechanicalAdjustmentX
        self._TxMechanicalAdjustmentY=TxMechanicalAdjustmentY
        self._TxMechanicalAdjustmentZ=TxMechanicalAdjustmentZ
        self._ZSteering=ZSteering
  
        
        
    def AddMaterial(self,Density,LSoS,SSoS,LAtt,SAtt): #add material (Density (kg/m3), long. SoS 9(m/s), shear SoS (m/s), Long. Attenuation (Np/m), shear attenuation (Np/m)
        self._Materials.append([Density,LSoS,SSoS,LAtt,SAtt]);
        
        
    @property
    def Wavelength(self):
        return self._Wavelength
        
        
    @property
    def SpatialStep(self):
        return self._SpatialStep
        
    def UpdateConditions(self, SkullMaskNii,AlphaCFL=1.0,bWaterOnly=False):
        '''
        Update simulation conditions
        '''
        MatArray=self.ReturnArrayMaterial()
        print('MatArray\n',MatArray)
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
        
        self.DominantMediumTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap*0,MatArray[0,:].reshape((1,5)),self._Frequency,self._QfactorCorrection,SpatialStep,1.0)

        TemporalStep=OTemporalStep

        print('"ideal" TemporalStep',TemporalStep)
        print('"ideal" DominantMediumTemporalStep',self.DominantMediumTemporalStep)

        #now we make it to be an integer division of the period
        self._PPP=np.ceil(1/self._Frequency/TemporalStep)
        #we add to catch the weird case it ends in 23, to avoid having a sensor that needs so many points
        if self._PPP==31:
            self._PPP=32
        if self._PPP==34:
            self._PPP=35
        if self._PPP==23:
            self._PPP=24
        if self._PPP==71:
            self._PPP=72
        TemporalStep=1/self._Frequency/self._PPP # we make it an integer of the period
        self._AdjustedCFL=TemporalStep/OTemporalStep*AlphaCFL
        
        #and back to SpatialStep
        print('"int fraction" TemporalStep',TemporalStep)
        print('"CFL fraction relative to water only conditions',TemporalStep/self.DominantMediumTemporalStep)
        
        print("adjusted AlphaCL, PPP",self._AdjustedCFL,self._PPP)
        
        self._SpatialStep=SpatialStep
        self._TemporalStep=TemporalStep
        
        #we save the mask array and flipped
        self._SkullMaskDataOrig=np.flip(SkullMaskNii.get_fdata(),axis=2)
        voxelS=np.array(SkullMaskNii.header.get_zooms())*1e-3
        print('voxelS, SpatialStep',voxelS,SpatialStep)
        assert(np.allclose(np.round(np.ones(voxelS.shape)*SpatialStep,6),np.round(voxelS,6)))
        
        #default offsets , this can change if the Rayleigh field does not fit
        if self._OrientationTx=='Z':
            self._XLOffset=self._PMLThickness 
            self._YLOffset=self._PMLThickness
            self._ZLOffset=self._PMLThickness+self._PaddingForRayleigh+self._PaddingForKArray
        elif self._OrientationTx=='X':
            self._XLOffset=self._PMLThickness+self._PaddingForRayleigh+self._PaddingForKArray 
            self._YLOffset=self._PMLThickness
            self._ZLOffset=self._PMLThickness
        else:
            self._XLOffset=self._PMLThickness 
            self._YLOffset=self._PMLThickness+self._PaddingForRayleigh+self._PaddingForKArray
            self._ZLOffset=self._PMLThickness
        self._XROffset=self._PMLThickness 
        self._YROffset=self._PMLThickness
        self._ZROffset=self._PMLThickness
        
        
        #swe will adjust size of domain until be sure the incident Rayleigh field fits in
        self._XShrink_L=0
        self._XShrink_R=0
        self._YShrink_L=0
        self._YShrink_R=0
        self._ZShrink_L=0
        self._ZShrink_R=0
        self.bMapFit=False
        bCompleteForShrinking=False
        self._nCountShrink=0
        while not self.bMapFit or not bCompleteForShrinking:
            self.bMapFit=True
            self._N1=self._SkullMaskDataOrig.shape[0]+self._XLOffset+self._XROffset -self._XShrink_L-self._XShrink_R
            self._N2=self._SkullMaskDataOrig.shape[1]+self._YLOffset+self._YROffset -self._YShrink_L-self._YShrink_R
            self._N3=self._SkullMaskDataOrig.shape[2]+self._ZLOffset+self._ZROffset -self._ZShrink_L-self._ZShrink_R


            self._FocalSpotLocationOrig=np.array(np.where(self._SkullMaskDataOrig==5.0)).flatten()
            self._FocalSpotLocationOrig[2]+=int(np.round((self._AdjustDepthFocalSpot/1e3)/SpatialStep))
            self._FocalSpotLocation=self._FocalSpotLocationOrig.copy()
            self._FocalSpotLocation+=np.array([self._XLOffset,self._YLOffset,self._ZLOffset])
            self._FocalSpotLocation-=np.array([self._XShrink_L,self._YShrink_L,self._ZShrink_L])
            print('self._FocalSpotLocation',self._FocalSpotLocation)
            
            xfield = np.arange(self._N1)*SpatialStep
            yfield = np.arange(self._N2)*SpatialStep
            zfield = np.arange(self._N3)*SpatialStep
            
            
            xfield-=xfield[self._FocalSpotLocation[0]]
            yfield-=yfield[self._FocalSpotLocation[1]]
            zfield-=zfield[self._FocalSpotLocation[2]]
            
            if self._OrientationTx=='Z':
                zfield+=self._FocalLength
                TopZ=zfield[self._PMLThickness]
            elif self._OrientationTx=='X':
                xfield+=self._FocalLength
                TopZ=xfield[self._PMLThickness]
            else:
                yfield+=self._FocalLength
                TopZ=yfield[self._PMLThickness]
            
            
            DistanceToFocus=self._FocalLength-TopZ+self._TxMechanicalAdjustmentZ
            Alpha=np.arcsin(self._Aperture/2/self._FocalLength)
            RadiusFace=DistanceToFocus*np.tan(Alpha)*1.05 # we make a bit larger to be sure of covering all incident beam            
            
            if self._OrientationTx=='Z':
                ypp,xpp=np.meshgrid(yfield,xfield)
            elif self._OrientationTx=='X':
                ypp,xpp=np.meshgrid(zfield,yfield)
            else:
                ypp,xpp=np.meshgrid(zfield,xfield)
            
            RegionMap=xpp**2+ypp**2<=RadiusFace**2 #we select the circle on the incident field
            IndXMap,IndYMap=np.nonzero(RegionMap)
            print('self._OrientationTx',self._OrientationTx)
            #any of this conditions will force to recalculate dimensions
            
            def fgen(var):
                sn={'X':'1','Y':'2','Z':'3'}
                pcode=\
'''

if np.any(Ind{0}Map<self._PMLThickness): 
    print('** Rayleigh map not fitting in the low part of N{1}, increasing it ...', self._{0}LOffset)
    self._{0}LOffset+=self._PMLThickness-Ind{0}Map.min()
    print('{0}LOffset',self._{0}LOffset)
    self.bMapFit=False
elif self._bTightNarrowBeamDomain:
    if self._{0}LOffset==self._PMLThickness:
        self._{0}Shrink_L+=Ind{0}Map.min()-self._{0}LOffset
        print('{0}Shrink_L',self._{0}Shrink_L)
    self._nCountShrink+=1
if np.any(Ind{0}Map>=self._N{1}-self._PMLThickness):
    print('** Rayleigh map not fitting in the upper part of N{1}, increasing it ...',self._{0}ROffset)
    self._{0}ROffset+=Ind{0}Map.max()-(self._N{1}-self._PMLThickness)+1
    print('{0}Offset',self._{0}ROffset)
    self.bMapFit=False
elif self._bTightNarrowBeamDomain:
    if self._{0}ROffset==self._PMLThickness:
        self._{0}Shrink_R+=self._N{1}-self._{0}ROffset-Ind{0}Map.max()-1
        print('{0}Shrink_R',self._{0}Shrink_R)
    self._nCountShrink+=1
'''.format(var,sn[var])
                return pcode
            
            if self._OrientationTx=='Z':
                exec(fgen('X'))
                exec(fgen('Y'))
              
                if self._bTightNarrowBeamDomain:
                    nStepsZReduction=int(self._zLengthBeyonFocalPointWhenNarrow/self._SpatialStep)
                    self._ZShrink_R+=self._N3-(self._FocalSpotLocation[2]+nStepsZReduction)
                    if self._ZShrink_R<0:
                        self._ZShrink_R=0
                    print('ZShrink_R',self._ZShrink_R,self._nCountShrink)
                    
            elif self._OrientationTx=='X':
                exec(fgen('Z'),{},{'self':self,'nCountShrink':nCountShrink,'np':np,'IndXMap':IndXMap,'IndYMap':IndXMap,'IndZMap':IndYMap})
                exec(fgen('Y'),{},{'self':self,'nCountShrink':nCountShrink,'np':np,'IndXMap':IndXMap,'IndYMap':IndXMap,'IndZMap':IndYMap})
                if self._bTightNarrowBeamDomain:
                    nStepsZReduction=int(self._zLengthBeyonFocalPointWhenNarrow/self._SpatialStep)
                    self._XShrink_R+=self._N1-(self._FocalSpotLocation[0]+nStepsZReduction)
                    if self._XShrink_R<0:
                        self._XShrink_R=0
                    print('XShrink_R',self._XShrink_R)
            else:
                exec(fgen('X'),{},{'self':self,'nCountShrink':nCountShrink,'np':np,'IndXMap':IndXMap,'IndYMap':IndXMap,'IndZMap':IndYMap})
                exec(fgen('Z'),{},{'self':self,'nCountShrink':nCountShrink,'np':np,'IndXMap':IndXMap,'IndYMap':IndXMap,'IndZMap':IndYMap})
                if self._bTightNarrowBeamDomain:
                    nStepsZReduction=int(self._zLengthBeyonFocalPointWhenNarrow/self._SpatialStep)
                    self._YShrink_R+=self._N2-(self._FocalSpotLocation[1]+nStepsZReduction)
                    if self._YShrink_R<0:
                        self._YShrink_R=0
                    print('YShrink_R',self._YShrink_R)
                
            if self.bMapFit:
                if self._bTightNarrowBeamDomain==False:
                    bCompleteForShrinking=True
                elif self._nCountShrink >=8:
                    bCompleteForShrinking=True
        
        self._XDim=xfield
        self._YDim=yfield
        self._ZDim=zfield
        
        
        print('Domain size',self._N1,self._N2,self._N3)
        self._DimDomain=np.zeros((3))
        self._DimDomain[0]=self._N1*SpatialStep
        self._DimDomain[1]=self._N2*SpatialStep
        self._DimDomain[2]=self._N3*SpatialStep
        
        self._TimeSimulation=np.sqrt(self._DimDomain[0]**2+self._DimDomain[1]**2+self._DimDomain[2]**2)/MatArray[0,1] #time to cross one corner to another
        self._TimeSimulation=np.floor(self._TimeSimulation/self._TemporalStep)*self._TemporalStep
        
        TimeVector=np.arange(0.0,self._TimeSimulation,self._TemporalStep)
        ntSteps=(int(TimeVector.shape[0]/self._PPP)+1)*self._PPP
        self._TimeSimulation=self._TemporalStep*ntSteps
        TimeVector=np.arange(0,ntSteps)*self._TemporalStep
        print('PPP, Subsampling, PPP for sensors ', self._PPP,self._SensorSubSampling, self._PPP/self._SensorSubSampling)
        assert((self._PPP%self._SensorSubSampling)==0)
        nStepsBack=int(self._NumberCyclesToTrackAtEnd*self._PPP)
        self._SensorStart=int((TimeVector.shape[0]-nStepsBack)/self._SensorSubSampling)

        self._MaterialMap=np.zeros((self._N1,self._N2,self._N3),np.uint32) # note the 32 bit size
        if bWaterOnly==False:
            if self._XShrink_R==0:
                upperXR=self._SkullMaskDataOrig.shape[0]
            else:
                upperXR=-self._XShrink_R
            if self._YShrink_R==0:
                upperYR=self._SkullMaskDataOrig.shape[1]
            else:
                upperYR=-self._YShrink_R
            if self._ZShrink_R==0:
                upperZR=self._SkullMaskDataOrig.shape[2]
            else:
                upperZR=-self._ZShrink_R
                
            self._MaterialMap[self._XLOffset:-self._XROffset,
                              self._YLOffset:-self._YROffset,
                              self._ZLOffset:-self._ZROffset]=\
                                self._SkullMaskDataOrig.astype(np.uint32)[self._XShrink_L:upperXR,
                                                                         self._YShrink_L:upperYR,
                                                                         self._ZShrink_L:upperZR]

            self._MaterialMap[self._MaterialMap==5]=4 # this is to make the focal spot location as brain tissue
        
        print('PPP, Duration simulation',np.round(1/self._Frequency/TemporalStep),self._TimeSimulation*1e6)
        
        print('Number of steps sensor',np.floor(self._TimeSimulation/self._TemporalStep/self._SensorSubSampling)-self._SensorStart)
        

    def GenTxSimplified(self):
        TxRC=GenerateFocusTx(self._Frequency,self._FocalLength, self._Aperture, SpeedofSoundWater(20.0))
        TxRC['Aperture']=self._Aperture
        TxRC['NumberElems']=1
        TxRC['center'][:,2]+=self._FocalLength
        TxRC['elemcenter'][:,2]+=self._FocalLength
        TxRC['VertDisplay'][:,2]+=self._FocalLength
        return TxRC
    
    def GenTx(self):
        print('self._InDiameters, self._OutDiameters,self._FocalLength',self._InDiameters, self._OutDiameters,self._FocalLength)
        TxRC=GeneratedRingArrayTx(self._Frequency,self._FocalLength, 
                             self._InDiameters, self._OutDiameters, SpeedofSoundWater(20.0))
        TxRC['Aperture']=self._Aperture
        TxRC['NumberElems']=len(self._InDiameters)
        TxRC['center'][:,2]+=self._FocalLength
        TxRC['elemcenter'][:,2]+=self._FocalLength
        for n in range(len(TxRC['RingVertDisplay'])):
            TxRC['RingVertDisplay'][n][:,2]+=self._FocalLength
        return TxRC
    
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        if platform != "darwin":
            InitCuda()
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elemens
        self._TxRC=self.GenTx()
        
        if self._bDisplay:
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = Axes3D(fig)
            
            for VertDisplay, FaceDisplay in zip(self._TxRC['RingVertDisplay'],
                                                self._TxRC['RingFaceDisplay']):
            
                ax.add_collection3d(Poly3DCollection(VertDisplay[FaceDisplay]*1e3)) #we plot the units in mm
                #3D display are not so smart as regular 2D, so we have to adjust manually the limits so we can see the figure correctly
            ax.set_xlim(-self._TxRC['Aperture']/2*1e3-5,self._TxRC['Aperture']/2*1e3+5)
            ax.set_ylim(-self._TxRC['Aperture']/2*1e3-5,self._TxRC['Aperture']/2*1e3+5)
            ax.set_zlim(0,135)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.show()
        
        for k in ['center','RingVertDisplay','elemcenter']:
            if k == 'RingVertDisplay':
                for n in range(len(self._TxRC[k])):
                    self._TxRC[k][n][:,0]+=self._TxMechanicalAdjustmentX
                    self._TxRC[k][n][:,1]+=self._TxMechanicalAdjustmentY
                    self._TxRC[k][n][:,2]+=self._TxMechanicalAdjustmentZ
                    if self._OrientationTx=='X':
                        p=self._TxRC[k][n][:,0].copy()
                        self._TxRC[k][n][:,0]=self._TxRC[k][:,2].copy()
                        self._TxRC[k][n][:,2]=p
                    elif self._OrientationTx=='Y':
                        p=self._TxRC[k][n][:,1].copy()
                        self._TxRC[k][n][:,1]=self._TxRC[k][:,2].copy()
                        self._TxRC[k][n][:,2]=p
                    
            else:
                self._TxRC[k][:,0]+=self._TxMechanicalAdjustmentX
                self._TxRC[k][:,1]+=self._TxMechanicalAdjustmentY
                self._TxRC[k][:,2]+=self._TxMechanicalAdjustmentZ
            if self._OrientationTx=='X':
                p=self._TxRC[k][:,0].copy()
                self._TxRC[k][:,0]=self._TxRC[k][:,2].copy()
                self._TxRC[k][:,2]=p
            elif self._OrientationTx=='Y':
                p=self._TxRC[k][:,1].copy()
                self._TxRC[k][:,1]=self._TxRC[k][:,2].copy()
                self._TxRC[k][:,2]=p
        
      
        #we apply an homogeneous pressure 
       
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        #we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming=np.zeros(self._TxRC['NumberElems'],np.complex64)
        self.BasePhasedArrayProgrammingRefocusing=np.zeros(self._TxRC['NumberElems'],np.complex64)
        
        if self._ZSteering!=0.0:
            print('Running Steering')
            ds=np.ones((1))*self._SpatialStep**2

            u0=np.zeros((1),np.complex64)
            u0[0]=1+0j
            center=np.zeros((1,3),np.float32)
            #to avoid adding an erroneus steering to the calculations, we need to discount the mechanical motion 
            center[0,0]=self._XDim[self._FocalSpotLocation[0]]+self._TxMechanicalAdjustmentX
            center[0,1]=self._YDim[self._FocalSpotLocation[1]]+self._TxMechanicalAdjustmentY
            center[0,2]=self._ZDim[self._FocalSpotLocation[2]]+self._ZSteering+self._TxMechanicalAdjustmentZ
            
            u2back=ForwardSimple(cwvnb_extlay,center,ds.astype(np.float32),
                                 u0,self._TxRC['elemcenter'].astype(np.float32),deviceMetal=deviceName)

            AllPhi=np.zeros(self._TxRC['NumberElems'])
            for n in range(self._TxRC['NumberElems']):
                self.BasePhasedArrayProgramming[n]=np.conjugate(u2back[n])
                phi=np.angle(np.conjugate(u2back[n]))
                AllPhi[n]=phi

            AllPhi-=AllPhi[0] # we made all phases relative to elem 0

    #         Distance=self._NaturalOutPlaneDistance+self._ZSteering #we do the steering relative to the natural out of plance distance
    #         print('Running Steering',Distance,TWExpPhaseDistance(Distance*1e3))

    #         AllPhi=np.deg2rad(TWExpPhaseDistance(Distance*1e3))
            self.BasePhasedArrayProgramming=np.exp(1j*AllPhi)
            print('Phase for array: [',np.rad2deg(AllPhi).tolist(),']')
            u0=np.zeros((self._TxRC['center'].shape[0],1),np.complex64)
            nBase=0
            for n in range(self._TxRC['NumberElems']):
                u0[nBase:nBase+self._TxRC['elemdims'][n][0]]=(self._SourceAmpPa*np.exp(1j*AllPhi[n])).astype(np.complex64)
                nBase+=self._TxRC['elemdims'][n][0]
        else:
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
        
        TopZ=self._ZDim[self._PMLThickness]
        DistanceToFocus=self._FocalLength-TopZ
        Alpha=np.arcsin(self._Aperture/2/self._FocalLength)
        RadiusFace=DistanceToFocus*np.tan(Alpha)*1.05 # we make a bit larger to be sure of covering all incident beam
        
        if self._OrientationTx=='Z':
            self._SourceMapRayleigh=u2[:,:,self._PMLThickness].copy()
            ypp,xpp=np.meshgrid(self._YDim,self._XDim)
        elif self._OrientationTx=='X':
            self._SourceMapRayleigh=u2[self._PMLThickness,:,:].copy()
            ypp,xpp=np.meshgrid(self._ZDim,self._YDim)
        else:
            self._SourceMapRayleigh=u2[:,self._PMLThickness,:].copy()
            ypp,xpp=np.meshgrid(self._ZDim,self._XDim)
        
        RegionMap=xpp**2+ypp**2<=RadiusFace**2 #we select the circle on the incident field
        self._SourceMapRayleigh[RegionMap==False]=0+1j*0
        
        if self._bDisplay:
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(np.abs(self._SourceMapRayleigh)/1e6,
                       vmin=np.abs(self._SourceMapRayleigh[RegionMap]).min()/1e6,cmap=plt.cm.jet)
            plt.colorbar()
            plt.title('Incident map to be forwarded propagated (MPa)')

            plt.subplot(1,2,2)
            
            if self._OrientationTx=='Z':
                plt.imshow((np.abs(u2[:,self._FocalSpotLocation[1],:]).T+
                                       self._MaterialMap[self._FocalSpotLocation[0],:,:].T*
                                       np.abs(u2[self._FocalSpotLocation[0],:,:]).max()/10)/1e6,
                                       extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()],
                                       cmap=plt.cm.jet)
            elif self._OrientationTx=='X':
                plt.imshow((np.abs(u2[:,:,self._FocalSpotLocation[2]]).T+
                                       self._MaterialMap[:,:,self._FocalSpotLocation[2]].T*
                                       np.abs(u2[:,:,self._FocalSpotLocation[2]]).max()/10)/1e6,
                                       extent=[self._XDim.min(),self._XDim.max(),self._YDim.max(),self._YDim.min()],
                                       cmap=plt.cm.jet)
            else:
                plt.imshow((np.abs(u2[:,:,self._FocalSpotLocation[2]]).T+
                                       self._MaterialMap[:,:,self._FocalSpotLocation[2]].T*
                                       np.abs(u2[:,:,self._FocalSpotLocation[2]]).max()/10)/1e6,
                                       extent=[self._XDim.min(),self._XDim.max(),self._YDim.max(),self._YDim.min()],
                                       cmap=plt.cm.jet)
            plt.colorbar()
            plt.title('Acoustic field with Rayleigh with skull and brain (MPa)')

          
        
    def ReturnArrayMaterial(self):
        return np.array(self._Materials)

        
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
        LocZ=self._PMLThickness
        
        SourceMaskIND=np.where(np.abs(self._SourceMapRayleigh)>0)
        if self._OrientationTx=='Z':
            SourceMask=np.zeros((self._N1,self._N2),np.uint32)
        elif self._OrientationTx=='X':
            SourceMask=np.zeros((self._N2,self._N3),np.uint32)
        else:
            SourceMask=np.zeros((self._N1,self._N3),np.uint32)
        
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
        if self._OrientationTx=='Z':
            self._SourceMap[:,:,LocZ]=SourceMask 
        elif self._OrientationTx=='X':
            self._SourceMap[LocZ,:,:]=SourceMask 
        else:
            self._SourceMap[:,LocZ,:]=SourceMask 
            
        self._PulseSource=PulseSource
        
        ## Now we create the sources for back propagation
        
        self._PunctualSource=np.sin(2*np.pi*self._Frequency*TimeVectorSource).reshape(1,len(TimeVectorSource))
        self._SourceMapPunctual=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        self._SourceMapPunctual[self._FocalSpotLocation[0],self._FocalSpotLocation[1],self._FocalSpotLocation[2]]=1
        

        if self._bDisplay:
            plt.figure(figsize=(6,3))
            for n in range(1,4):
                plt.plot(TimeVectorSource*1e6,PulseSource[int(PulseSource.shape[0]/4)*n,:])
                plt.title('CW signal, example %i' %(n))
                
            plt.xlim(0,50)
                
            plt.figure(figsize=(3,2))
            if self._OrientationTx=='Z':
                plt.imshow(self._SourceMap[:,:,LocZ])
            elif self._OrientationTx=='Y':
                plt.imshow(self._SourceMap[:,LocZ,:])
            else:
                plt.imshow(self._SourceMap[LocZ,:,:])
            plt.title('source map - source ids')

                
    def CreateSensorMap(self):
        
        self._SensorMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        # for the back propagation, we only use the entering face
        self._SensorMapBackPropagation=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        
        if self._OrientationTx=='Z':
            self._SensorMapBackPropagation[self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness,self._PMLThickness]=1
            self._SensorMap[self._PMLThickness:-self._PMLThickness,self._FocalSpotLocation[1],self._PMLThickness:-self._PMLThickness]=1
        elif self._OrientationTx=='X':
            self._SensorMapBackPropagation[self._PMLThickness,self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness]=1
            self._SensorMap[self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness,self._FocalSpotLocation[2]]=1
        else:
            self._SensorMapBackPropagation[self._PMLThickness:-self._PMLThickness,self._PMLThickness,self._PMLThickness:-self._PMLThickness]=1
            self._SensorMap[self._FocalSpotLocation[0],self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness]=1
            
        if self._bDisplay:
            plt.figure()
            plt.imshow(self._SensorMap[:,self._FocalSpotLocation[1],:].T,cmap=plt.cm.gray)
            plt.title('Sensor map location')
        
            
        
    def RUN_SIMULATION(self,GPUName='SUPER',SelMapsRMSPeakList=['Pressure'],bRefocused=False,
                       bApplyCorrectionForDispersion=True,COMPUTING_BACKEND=1,bDoRefocusing=True):
        MaterialList=self.ReturnArrayMaterial()

        TypeSource=2 #stress source
        Ox=np.ones(self._MaterialMap.shape) #we do not do weigthing for a forwardpropagated source
        Oy=np.array([1])
        Oz=np.array([1])
        
        if bRefocused==False:
            self._Sensor,LastMap,self._DictPeakValue,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                             self._MaterialMap,
                                                             MaterialList,
                                                             self._Frequency,
                                                             self._SourceMap,
                                                             self._PulseSource,
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
                                                             SelRMSorPeak=1,
                                                             DefaultGPUDeviceName=GPUName,
                                                             AlphaCFL=1.0,
                                                             TypeSource=TypeSource,
                                                             QfactorCorrection=self._QfactorCorrection,
                                                             SensorSubSampling=self._SensorSubSampling,
                                                             SensorStart=self._SensorStart)
            
            self._InputParam=InputParam['IndexSensorMap']
            gc.collect()

            if bDoRefocusing:
            #now backpropagation
                self._SensorBack,_,_,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                                 self._MaterialMap,
                                                                 MaterialList,
                                                                 self._Frequency,
                                                                 self._SourceMapPunctual,
                                                                 self._PunctualSource,
                                                                 self._SpatialStep,
                                                                 self._TimeSimulation,
                                                                 self._SensorMapBackPropagation,
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
                                                                 SelRMSorPeak=1,
                                                                 DefaultGPUDeviceName=GPUName,
                                                                 AlphaCFL=1.0,
                                                                 TypeSource=TypeSource,
                                                                 QfactorCorrection=self._QfactorCorrection,
                                                                 SensorSubSampling=self._SensorSubSampling,
                                                                 SensorStart=self._SensorStart)
                self._InputParamBack=InputParam['IndexSensorMap']
        else:
            self._SensorRefocus,_,self._DictPeakValueRefocus,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                             self._MaterialMap,
                                                             MaterialList,
                                                             self._Frequency,
                                                             self._SourceMap,
                                                             self._PulseSourceRefocus,
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
                                                             SelRMSorPeak=1,
                                                             DefaultGPUDeviceName=GPUName,
                                                             AlphaCFL=1.0,
                                                             TypeSource=TypeSource,
                                                             QfactorCorrection=self._QfactorCorrection,
                                                             SensorSubSampling=self._SensorSubSampling,
                                                             SensorStart=self._SensorStart)
            self._InputParamRefocus=InputParam['IndexSensorMap']

        if bApplyCorrectionForDispersion:
            CFLWater=self._TemporalStep/self.DominantMediumTemporalStep
            ExpectedError=np.polyval(self._DispersionCorrection,CFLWater)
            Correction=100.0/(100.0-ExpectedError)
            print('CFLWater only, ExpectedError, Correction', CFLWater,ExpectedError,Correction)
            if bRefocused==False:
                for k in self._DictPeakValue:
                    self._DictPeakValue[k]*=Correction*np.sqrt(2)
                for k in self._Sensor:
                    if k=='time':
                        continue
                    self._Sensor[k]*=Correction
                if bDoRefocusing:
                    for k in self._SensorBack:
                        if k=='time':
                            continue
                        self._SensorBack[k]*=Correction
            else:
                for k in self._DictPeakValueRefocus:
                    self._DictPeakValueRefocus[k]*=Correction*np.sqrt(2)
                for k in self._SensorRefocus:
                    if k=='time':
                        continue
                    self._SensorRefocus[k]*=Correction
                    
        gc.collect()
    
    def CalculatePhaseData(self,bRefocused=False,bDoRefocusing=True):
        
        t0=time.time()
        if bRefocused==False:
            self._PhaseMap=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapFourier=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapPeak=np.zeros((self._N1,self._N2,self._N3))
            if bDoRefocusing:
                if self._OrientationTx=='Z':
                    self._PressMapFourierBack=np.zeros((self._N1,self._N2),np.complex64)
                elif self._OrientationTx=='X':
                    self._PressMapFourierBack=np.zeros((self._N2,self._N3),np.complex64)
                else:
                    self._PressMapFourierBack=np.zeros((self._N1,self._N3),np.complex64)
        else:
            self._PhaseMapRefocus=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapFourierRefocus=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapPeakRefocus=np.zeros((self._N1,self._N2,self._N3))
            
   
        time_step = np.diff(self._Sensor['time']).mean() #remember the sensor time vector can be different from the input source
        if self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling) !=0: #because some roundings, we may get
            print('Rounding of time vector was not exact multiple of PPP, truncating time vector a little')
            nDiff=int(self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling))
            print(' Cutting %i entries from sensor from length %i to %i' %(nDiff,self._Sensor['time'].shape[0],self._Sensor['time'].shape[0]-nDiff))
            self._Sensor['time']=self._Sensor['time'][:-nDiff]
            self._Sensor['Pressure']=self._Sensor['Pressure'][:-nDiff]
        assert((self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling))==0)

        freqs = np.fft.fftfreq(self._Sensor['time'].size, time_step)
        IndSpectrum=np.argmin(np.abs(freqs-self._Frequency)) # frequency entry closest to 500 kHz
        if bRefocused==False:
            self._Sensor['Pressure']=np.ascontiguousarray(self._Sensor['Pressure'])
            
            index=self._InputParam
            nStep=100000
            for n in range(0,self._Sensor['Pressure'].shape[0],nStep):
                top=np.min([n+nStep,self._Sensor['Pressure'].shape[0]])
                FSignal=fft.fft(self._Sensor['Pressure'][n:top,:],axis=1)
                k=np.floor(index[n:top]/(self._N1*self._N2)).astype(np.int64)
                j=index[n:top]%(self._N1*self._N2)
                i=j%self._N1
                j=np.floor(j/self._N1).astype(np.int64)
                FSignal=FSignal[:,IndSpectrum]
                pa= np.angle(FSignal)
                pp=np.abs(FSignal)

                self._PhaseMap[i,j,k]=pa
                self._PressMapFourier[i,j,k]=pp
                self._PressMapPeak[i,j,k]=self._Sensor['Pressure'][n:top,:].max(axis=1)
            self._InPeakValue=self._DictPeakValue['Pressure']
            self._PressMapFourier*=2/self._Sensor['time'].size
            print('Elapsed time doing phase and amp extraction from Fourier (s)',time.time()-t0)
            
            if bDoRefocusing:
                self._SensorBack['Pressure']=np.ascontiguousarray(self._SensorBack['Pressure'])
                index=self._InputParamBack
                for n in range(0,self._SensorBack['Pressure'].shape[0],nStep):
                    top=np.min([n+nStep,self._SensorBack['Pressure'].shape[0]])
                    FSignal=fft.fft(self._SensorBack['Pressure'][n:top,:],axis=1)
                    k=np.floor(index[n:top]/(self._N1*self._N2)).astype(np.int64)
                    j=index[n:top]%(self._N1*self._N2)
                    i=j%self._N1
                    j=np.floor(j/self._N1).astype(np.int64)
                    FSignal=FSignal[:,IndSpectrum]
                    if self._OrientationTx=='Z':
                        assert(np.all(k==self._PMLThickness))
                        self._PressMapFourierBack[i,j]=FSignal
                    elif self._OrientationTx=='X':
                        assert(np.all(i==self._PMLThickness))
                        self._PressMapFourierBack[j,k]=FSignal
                    else:
                        assert(np.all(j==self._PMLThickness))
                        self._PressMapFourierBack[i,k]=FSignal
            
        else:
            self._SensorRefocus['Pressure']=np.ascontiguousarray(self._SensorRefocus['Pressure'])
            index=self._InputParamRefocus
            nStep=100000
            for n in range(0,self._SensorRefocus['Pressure'].shape[0],nStep):
                top=np.min([n+nStep,self._SensorRefocus['Pressure'].shape[0]])
                FSignal=fft.fft(self._SensorRefocus['Pressure'][n:top,:],axis=1)
                k=np.floor(index[n:top]/(self._N1*self._N2)).astype(np.int64)
                j=index[n:top]%(self._N1*self._N2)
                i=j%self._N1
                j=np.floor(j/self._N1).astype(np.int64)
                FSignal=FSignal[:,IndSpectrum]
                pa= np.angle(FSignal)
                pp=np.abs(FSignal)

                self._PhaseMapRefocus[i,j,k]=pa
                self._PressMapFourierRefocus[i,j,k]=pp
                self._PressMapPeakRefocus[i,j,k]=self._SensorRefocus['Pressure'][n:top,:].max(axis=1)
            self._InPeakValueRefocus=self._DictPeakValueRefocus['Pressure']
            self._PressMapFourierRefocus*=2/self._SensorRefocus['time'].size
            print('Elapsed time doing phase and amp extraction from Fourier (s)',time.time()-t0)
            
        
            
    def BackPropagationRayleigh(self,deviceName='6800'):
        assert(np.all(np.array(self._SourceMapRayleigh.shape)==np.array(self._PressMapFourierBack.shape)))
        SelRegRayleigh=np.abs(self._SourceMapRayleigh)>0
        if self._OrientationTx=='Z':
            ypp,xpp=np.meshgrid(self._YDim,self._XDim)
            ypp=ypp[SelRegRayleigh]
            xpp=xpp[SelRegRayleigh]
            center=np.zeros((ypp.size,3),np.float32)
            center[:,0]=xpp.flatten()
            center[:,1]=ypp.flatten()
            center[:,2]=self._ZDim[self._PMLThickness]
        elif self._OrientationTx=='X':
            ypp,xpp=np.meshgrid(self._ZDim,self._YDim)
            ypp=ypp[SelRegRayleigh]
            xpp=xpp[SelRegRayleigh]
            center=np.zeros((ypp.size,3),np.float32)
            center[:,1]=xpp.flatten()
            center[:,2]=ypp.flatten()
            center[:,0]=self._XDim[self._PMLThickness]
        else:
            ypp,xpp=np.meshgrid(self._ZDim,self._XDim)
            ypp=ypp[SelRegRayleigh]
            xpp=xpp[SelRegRayleigh]
            center=np.zeros((ypp.size,3),np.float32)
            center[:,0]=xpp.flatten()
            center[:,2]=ypp.flatten()
            center[:,1]=self._YDim[self._PMLThickness]
            
        ds=np.ones((center.shape[0]))*self._SpatialStep**2
        
        if platform != "darwin":
            InitCuda()

        #we apply an homogeneous pressure 
        u0=self._PressMapFourierBack[SelRegRayleigh]
        
        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)

        u2back=ForwardSimple(cwvnb_extlay,center.astype(np.float32),ds.astype(np.float32),
                             u0,self._TxRC['elemcenter'].astype(np.float32),deviceMetal=deviceName)
        
        #now we calculate forward back
        
        u0=np.zeros((self._TxRC['center'].shape[0],1),np.complex64)
        nBase=0
        for n in range(self._TxRC['NumberElems']):
            phi=np.angle(np.conjugate(u2back[n]))
            self.BasePhasedArrayProgrammingRefocusing[n]=np.conjugate(u2back[n])
            u0[nBase:nBase+self._TxRC['elemdims'][n][0]]=(self._SourceAmpPa*np.exp(1j*phi)).astype(np.complex64)
            nBase+=self._TxRC['elemdims'][n][0]

        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        yp,xp,zp=np.meshgrid(self._YDim,self._XDim,self._ZDim)
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u2=ForwardSimple(cwvnb_extlay,self._TxRC['center'].astype(np.float32),self._TxRC['ds'].astype(np.float32),
                         u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)
        
        TopZ=self._ZDim[self._PMLThickness]
        DistanceToFocus=self._FocalLength-TopZ
        Alpha=np.arcsin(self._Aperture/2/self._FocalLength)
        RadiusFace=DistanceToFocus*np.tan(Alpha)*1.05 # we make a bit larger to be sure of covering all incident beam
        
        if self._OrientationTx=='Z':
            self._SourceMapRayleighRefocus=u2[:,:,self._PMLThickness].copy()
            ypp,xpp=np.meshgrid(self._YDim,self._XDim)
        elif self._OrientationTx=='X':
            self._SourceMapRayleighRefocus=u2[self._PMLThickness,:,:].copy()
            ypp,xpp=np.meshgrid(self._ZDim,self._YDim)
        else:
            self._SourceMapRayleighRefocus=u2[:,self._PMLThickness,:].copy()
            ypp,xpp=np.meshgrid(self._ZDim,self._XDim)
        
        RegionMap=xpp**2+ypp**2<=RadiusFace**2 #we select the circle on the incident field
        self._SourceMapRayleighRefocus[RegionMap==False]=0+1j*0  
        
        
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
        
        LocZ=self._PMLThickness
        
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
         
        
    def PlotResultsPlanePartial(self):
  
        if self._bDisplay:
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(self._PressMapPeak[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD peak amp. (MPa)')
            plt.subplot(1,3,2)
            plt.imshow(self._PressMapFourier[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD Fourier amp. (MPa)')
            plt.subplot(1,3,3)
            plt.imshow(self._InPeakValue[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD InPeak amp. (MPa)')
            
    def PlotResultsPlane(self,bDoRefocusing=True):
  
        if self._bDisplay:
            if bDoRefocusing:
                plt.figure(figsize=(9,6))
                nRows=2
            else:
                plt.figure(figsize=(9,6))
                nRows=1
            plt.subplot(nRows,3,1)
            plt.imshow(self._PressMapPeak[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD peak amp. (MPa)')
            plt.subplot(nRows,3,2)
            plt.imshow(self._PressMapFourier[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD Fourier amp. (MPa)')
            plt.subplot(nRows,3,3)
            plt.imshow(self._InPeakValue[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD InPeak amp. (MPa)')
            
            if bDoRefocusing:
                plt.subplot(nRows,3,4)
                plt.imshow(self._PressMapPeakRefocus[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
                plt.xlim(self._XDim.min(),self._XDim.max())
                plt.ylim(self._ZDim.max(),self._ZDim.min())
                plt.colorbar()
                plt.title('BabelViscoFDTD peak refocus (MPa)')
                plt.subplot(nRows,3,5)
                plt.imshow(self._PressMapFourierRefocus[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
                plt.xlim(self._XDim.min(),self._XDim.max())
                plt.ylim(self._ZDim.max(),self._ZDim.min())
                plt.colorbar()
                plt.title('BabelViscoFDTD Fourier Refocus (MPa)')
                plt.subplot(nRows,3,6)
                plt.imshow(self._InPeakValueRefocus[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
                plt.xlim(self._XDim.min(),self._XDim.max())
                plt.ylim(self._ZDim.max(),self._ZDim.min())
                plt.colorbar()
                plt.title('BabelViscoFDTD InPeak amp. Refocus (MPa)')
            


        LineInPeak=self._InPeakValue[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/2/1.5e6/1e4
        LineFourierAmp=self._PressMapFourier[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/2/1.5e6/1e4
        LineRayleigh=np.abs(self._u2RayleighField[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:])/2/1.5e6/1e4
        # if bDoRefocusing:
        #     LineInPeakRefocus=self._InPeakValueRefocus[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]
        #     LinePeakRefocus=self._PressMapPeakRefocus[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]
        #     LineFourierAmpRefocus=self._PressMapFourierRefocus[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]

        if self._bDisplay:
            Z=self._ZDim*1e3
            Z-=Z[self._FocalSpotLocation[2]]
            Z+=self._NaturalOutPlaneDistance*1e3+self._TxMechanicalAdjustmentZ*1e3
            fig, ax = plt.subplots(1,1,figsize=(6,4))
            ax.plot(Z,LineInPeak)
            ax.plot(Z,LineFourierAmp)
            ax.plot(Z,LineRayleigh)
            if bDoRefocusing:
                ax.plot(Z,LinePeakRefocus)
                ax.plot(Z,LineInPeakRefocus)
                ax.plot(Z,LineFourierAmpRefocus)
            if bDoRefocusing:
                plt.legend(['Inpeak','Fourier','RayleighWater','PeakRefocus','InpeakRefocus','FourierRefocus'])
            else:
                plt.legend(['Inpeak','Fourier','RayleighWater'])
            ax.plot([0.0,0.0],
                     [0,np.max(LineInPeak)],':')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.set_xlim(10,Z.max())
        
    def ReturnResults(self,bDoRefocusing=True):
        if self._XShrink_R==0:
            upperXR=self._SkullMaskDataOrig.shape[0]
        else:
            upperXR=-self._XShrink_R
        if self._YShrink_R==0:
            upperYR=self._SkullMaskDataOrig.shape[1]
        else:
            upperYR=-self._YShrink_R
        if self._ZShrink_R==0:
            upperZR=self._SkullMaskDataOrig.shape[2]
        else:
            upperZR=-self._ZShrink_R

        #we return the region not including the PML and padding
        RayleighWater=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        RayleighWater[self._XShrink_L:upperXR,
                      self._YShrink_L:upperYR,
                      self._ZShrink_L:upperZR]=\
                      np.abs(self._u2RayleighField[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset])
        
        RayleighWater=np.flip(RayleighWater,axis=2)
        #this one creates an overlay of skull and brain tissue that helps to show it Slicer or other visualization tools
        MaskSkull=np.flip(self._SkullMaskDataOrig.astype(np.float32),axis=2)
        RayleighWaterOverlay=RayleighWater+MaskSkull*RayleighWater.max()/10
        
        FullSolutionPressure=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        FullSolutionPressure[self._XShrink_L:upperXR,
                      self._YShrink_L:upperYR,
                      self._ZShrink_L:upperZR]=\
                      self._InPeakValue[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset]
        FullSolutionPressure=np.flip(FullSolutionPressure,axis=2)
        FullSolutionPressureRefocus=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        if bDoRefocusing:
            FullSolutionPressureRefocus[self._XShrink_L:upperXR,
                          self._YShrink_L:upperYR,
                          self._ZShrink_L:upperZR]=\
                          self._InPeakValueRefocus[self._XLOffset:-self._XROffset,
                                       self._YLOffset:-self._YROffset,
                                       self._ZLOffset:-self._ZROffset]
            FullSolutionPressureRefocus=np.flip(FullSolutionPressureRefocus,axis=2)
        
        DataForSim ={}
        DataForSim['p_amp']=self._InPeakValue[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset].copy()
        if bDoRefocusing:
            DataForSim['p_amp_refocus']=self._InPeakValueRefocus[self._XLOffset:-self._XROffset,
                                       self._YLOffset:-self._YROffset,
                                       self._ZLOffset:-self._ZROffset].copy()
            
        MaterialMap=self._MaterialMap.copy()
        MaterialMap[self._FocalSpotLocation[0],self._FocalSpotLocation[1],self._FocalSpotLocation[2]]=5.0
        
        DataForSim['MaterialMap']=MaterialMap[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset].copy()
        TargetLocation=np.array(np.where(DataForSim['MaterialMap']==5.0)).flatten()
        DataForSim['MaterialMap'][DataForSim['MaterialMap']==5.0]=4.0 #we switch it back to soft tissue
        
        
        for k in DataForSim:
            DataForSim[k]=np.flip(DataForSim[k],axis=2)
        DataForSim['Material']=self.ReturnArrayMaterial()
        DataForSim['x_vec']=self._XDim[self._XLOffset:-self._XROffset]
        DataForSim['y_vec']=self._YDim[self._YLOffset:-self._YROffset]
        DataForSim['z_vec']=self._ZDim[self._ZLOffset:-self._ZROffset]
        DataForSim['SpatialStep']=self._SpatialStep
        DataForSim['TargetLocation']=TargetLocation
        
        
        assert(np.all(np.array(RayleighWaterOverlay.shape)==np.array(FullSolutionPressure.shape)))
        return  RayleighWater,\
                RayleighWaterOverlay,\
                FullSolutionPressure,\
                FullSolutionPressureRefocus,\
                DataForSim
        