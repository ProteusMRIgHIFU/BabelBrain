import sys
import platform
import traceback
from BabelViscoFDTD.tools.RayleighAndBHTE import  InitOpenCL, InitCuda, InitMetal

from IntegrationBrainsightTW.CalculateTemperatureEffects import CalculateTemperatureEffects

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
        if sys.platform not in ['linux','win32']:
            if 'arm64' in platform.platform():
                InitMetal(kargs['deviceName'])
                Backend='Metal'
            else:
                InitOpenCL(kargs['deviceName'])
                Backend='OpenCL'
        else:
            if kargs['COMPUTING_BACKEND']==1:
                InitCuda()
                Backend='CUDA'
            elif kargs['COMPUTING_BACKEND']==2:
                InitOpenCL(kargs['deviceName'])
                Backend='OpenCL'
            

        for combination in AllDC_PRF_Duration:
            fname=CalculateTemperatureEffects(case,
                                                DutyCycle=combination['DC'],
                                                PRF=combination['PRF'],
                                                DurationUS=combination['Duration'],
                                                Isppa=kargs['Isppa'],
                                                sel_p=kargs['sel_p'],
                                                bPlot=False,
                                                bCalculateLosses=True,
                                                bForceRecalc=True,
                                                Backend=Backend)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))