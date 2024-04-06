import sys
import platform
import traceback
import numpy as np
import nibabel

def CalculateFieldProcess(queue,Target,TxSystem,**kargs):
    
    class InOutputWrapper(object):
       
        def __init__(self, queue, stdout=True):
            self.queue=queue
            if stdout:
                self._stream = sys.stdout
                sys.stdout = self
            else:
                self._stream = sys.stderr
                sys.stderr = self
            self._stdout = stdout

        def write(self, text):
            self.queue.put(text)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def __del__(self):
            try:
                if self._stdout:
                    sys.stdout = self._stream
                else:
                    sys.stderr = self._stream
            except AttributeError:
                pass

    if TxSystem in ['Single','BSonix']:
        from TranscranialModeling.BabelIntegrationSingle import RUN_SIM 
    elif TxSystem =='CTX_500':
        from TranscranialModeling.BabelIntegrationCTX500 import RUN_SIM 
    elif TxSystem =='H317':
        from TranscranialModeling.BabelIntegrationH317 import RUN_SIM
    elif TxSystem =='H246':
        from TranscranialModeling.BabelIntegrationH246 import RUN_SIM 
    elif TxSystem =='REMOPD':
        from TranscranialModeling.BabelIntegrationREMOPD import RUN_SIM
    elif TxSystem =='I12378':
        from TranscranialModeling.BabelIntegrationI12378 import RUN_SIM
    elif TxSystem =='ATAC':
        from TranscranialModeling.BabelIntegrationATAC import RUN_SIM
    else:
        raise ValueError("TX system " + TxSystem + " is not yet supported")

    if TxSystem in ['H317','REMOPD','I12378','ATAC']:
        if kargs['bDryRun']==False:
            stdout = InOutputWrapper(queue,True)
    else:
        stdout = InOutputWrapper(queue,True)
    try:
        R=RUN_SIM()
        FilesSkull=R.RunCases(targets=Target, 
                        bTightNarrowBeamDomain=True,
                        bForceRecalc=True,
                        bDisplay=False,
                        **kargs)
        bDryRun = False
        
        if 'bDryRun' in kargs:
            bDryRun=kargs['bDryRun']
        if kargs['bUseRayleighForWater']==False or bDryRun:
            if TxSystem in ['H317','REMOPD','I12378','ATAC']:
                kargs['bDoRefocusing']=False
                if kargs['XSteering']==0.0:
                    kargs['XSteering']=1e-6
            FilesWater=R.RunCases(targets=Target, 
                            bTightNarrowBeamDomain=True,
                            bForceRecalc=True,
                            bWaterOnly=True,
                            bDisplay=False,
                            **kargs)
        if TxSystem in ['H317','I12378','ATAC']:
            #we need to combine ac field files for display if using multipoint
            if kargs['MultiPoint'] is not None and kargs['bDryRun'] == False: 
                kargs['bDryRun'] = True
                FilesWater=R.RunCases(targets=Target, 
                            bTightNarrowBeamDomain=True,
                            bForceRecalc=True,
                            bWaterOnly=True,
                            bDisplay=False,
                            **kargs)
                #now we combine the individual Nifti files into a single one , this is required mainly for proper visualization in Brainsight
                nSub=[]
                nRefocus=[]
                for fnames in [FilesSkull,FilesWater]:
                    for f in fnames:
                        fsub=f.replace('DataForSim.h5','FullElasticSolution_Sub_NORM.nii.gz')
                        nSub.append(nibabel.load(fsub))    
                        if kargs['bDoRefocusing']:
                            fsubrefocus=f.replace('DataForSim.h5','FullElasticSolutionRefocus_Sub_NORM.nii.gz')
                            nRefocus.append(nibabel.load(fsubrefocus))    
                    
                    for ss,sub in zip(['','Refocus'],[nSub,nRefocus]):
                        if len(sub)>0:
                            AllpData=np.zeros((len(sub),sub[0].shape[0],sub[0].shape[1],sub[0].shape[2]))
                            for n,entry in enumerate(sub):
                                AllpData[n,:,:,:]=entry.get_fdata()
                            AllpData=AllpData.max(axis=0)
                            combinedNifti=nibabel.Nifti1Image(AllpData,sub[0].affine,header=sub[0].header)
                            if 'Water_DataForSim.h5' in fnames[0] :
                                send = '_Water_FullElasticSolution'+ss+'_Sub_NORM.nii.gz'
                            else:
                                send = '_FullElasticSolution'+ss+'_Sub_NORM.nii.gz'
                            finalName=fnames[0].split('__Steer_X')[0]+send
                            combinedNifti.to_filename(finalName)

        if TxSystem in ['H317','REMOPD','I12378','ATAC']:
            kargs['bDryRun'] = True
            FilesWater=R.RunCases(targets=Target, 
                            bTightNarrowBeamDomain=True,
                            bForceRecalc=True,
                            bWaterOnly=True,
                            bDisplay=False,
                            **kargs)
            outFiles={'FilesSkull':FilesSkull,'FilesWater':FilesWater}
            if bDryRun==False:
                queue.put(outFiles)
            else:
                return outFiles
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))
