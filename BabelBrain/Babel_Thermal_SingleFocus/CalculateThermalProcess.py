import sys
import platform
import traceback
from BabelViscoFDTD.tools.RayleighAndBHTE import  InitOpenCL, InitCuda, InitMetal
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
from scipy.io import savemat
import numpy as np

from ThermalModeling.CalculateTemperatureEffects import CalculateTemperatureEffects
from multiprocessing import Process,Queue


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

#subprocess , useful to deal with large number of cases
def SubProcess(queueMsg,queueResult,case,deviceName,**kargs):
    stdout = InOutputWrapper(queueMsg,True)
    if kargs['Backend']=='CUDA':
        InitCuda(deviceName)
    elif kargs['Backend']=='OpenCL':
        InitOpenCL(deviceName)
    else:
        InitMetal(deviceName)
    fname=CalculateTemperatureEffects(case,**kargs)
    queueResult.put(fname)
    

def CalculateThermalProcess(queueMsg,case,AllDC_PRF_Duration,**kargs):

    try:
        Backend = ['CUDA','OpenCL','Metal'][kargs['COMPUTING_BACKEND']-1]
        deviceName=kargs['deviceName']
        AllCases=[]
        #These fields will preseved individually per sonication regime
        lf =['MaxBrainPressure','MaxIsppa', 'MaxIspta','MonitorSlice','TI','TIC','TIS','TempProfileTarget',\
            'TimeProfileTarget','p_map_central','Isppa','Ispta','MI','DurationUS','DurationOff','DutyCycle','PRF']
        Index=[]
        for combination in AllDC_PRF_Duration:
            SubData={}

            queueResult=Queue()
            kargsSub={}
            kargsSub['DutyCycle']=combination['DC']
            kargsSub['PRF']=combination['PRF']
            kargsSub['DurationUS']=combination['Duration']
            kargsSub['DurationOff']=combination['DurationOff']
            kargsSub['Isppa']=kargs['Isppa']
            kargsSub['sel_p']=kargs['sel_p']
            kargsSub['bPlot']=False
            kargsSub['bForceRecalc']=True
            kargsSub['Backend']=Backend
            fieldWorkerProcess = Process(target=SubProcess, 
                                    args=(queueMsg,queueResult,case,deviceName),
                                    kwargs=kargsSub)

            fieldWorkerProcess.start()
            fieldWorkerProcess.join()
            fname=queueResult.get()
            Data=ReadFromH5py(fname+'.h5')
            for f in lf:
                if 'p_map_central'==f:
                    SubData['p_map']=Data[f] #this will make it compatible for other purposes
                SubData[f]=Data[f]
            AllCases.append(SubData)
            Index.append([combination['DC'],combination['PRF'],combination['Duration'],combination['DurationOff'],np.round(kargs['Isppa'],1)])
        for f in lf:
            if f in Data: #this will prevent failing with fields that are specific to phase arrays such as steering
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