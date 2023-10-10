import sys
import platform
import traceback
from BabelViscoFDTD.tools.RayleighAndBHTE import  InitOpenCL, InitCuda, InitMetal
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
from scipy.io import savemat
import numpy as np

from ThermalModeling.CalculateTemperatureEffects import CalculateTemperatureEffects

def CalculateThermalProcess(queue,case,AllDC_PRF_Duration,**kargs):
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
    stdout = InOutputWrapper(queue,True)

    try:
        if kargs['COMPUTING_BACKEND']==1:
            InitCuda()
            Backend='CUDA'
        elif kargs['COMPUTING_BACKEND']==2:
            InitOpenCL(kargs['deviceName'])
            Backend='OpenCL'
        else:
            InitMetal(kargs['deviceName'])
            Backend='Metal'

        AllCases=[]
        lf =['MaxBrainPressure','MaxIsppa', 'MaxIspta','MonitorSlice','TI','TIC','TIS','TempProfileTarget',\
            'TimeProfileTarget','p_map_central','Isppa','Ispta','MI']
        Index=[]
        for combination in AllDC_PRF_Duration:
            SubData={}
            fname=CalculateTemperatureEffects(case,
                                                DutyCycle=combination['DC'],
                                                PRF=combination['PRF'],
                                                DurationUS=combination['Duration'],
                                                DurationOff=combination['DurationOff'],
                                                Isppa=kargs['Isppa'],
                                                sel_p=kargs['sel_p'],
                                                bPlot=False,
                                                bCalculateLosses=True,
                                                bForceRecalc=True,
                                                Backend=Backend)
            Data=ReadFromH5py(fname+'.h5')
            for f in lf:
                if 'p_map_central'==f:
                    SubData['p_map']=Data[f] #this will make it compatible for other purposes
                SubData[f]=Data[f]
            AllCases.append(SubData)
            Index.append([combination['DC'],combination['PRF'],combination['Duration'],combination['DurationOff'],np.round(kargs['Isppa'],1)])
        for f in lf:
            Data.pop(f)
        Index=np.array(Index)

        Data['AllData']=AllCases
        Data['Index']=Index
        ConsolodidateName=fname.split('-Duration-')[0]+'_AllCombinations'
        savemat(ConsolodidateName+'.mat',Data)
        SaveToH5py(Data,ConsolodidateName+'.h5')

    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))