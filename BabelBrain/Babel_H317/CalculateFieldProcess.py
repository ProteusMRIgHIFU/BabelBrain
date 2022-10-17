import sys
import platform
import traceback

from TranscranialModeling.BabelIntegrationUC import RUN_SIM

def CalculateFieldProcess(queue,Target,**kargs):
    
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
        COMPUTING_BACKEND=kargs['COMPUTING_BACKEND']
        R=RUN_SIM()
        R.RunCases(targets=Target,ID=kargs['ID'],
                        deviceName=kargs['deviceName'],
                        COMPUTING_BACKEND=COMPUTING_BACKEND,
                        bTightNarrowBeamDomain=True,
                        bForceRecalc=True,
                        basePPW=kargs['basePPW'],
                        basedir=kargs['basedir'],
                        TxMechanicalAdjustmentZ=kargs['TxMechanicalAdjustmentZ'],
                        TxMechanicalAdjustmentX=kargs['TxMechanicalAdjustmentX'],
                        TxMechanicalAdjustmentY=kargs['TxMechanicalAdjustmentY'],
                        ZSteering=kargs['ZSteering'],
                        RotationZ=kargs['RotationZ'],
                        Frequencies=kargs['Frequencies'],
                        XSteering=kargs['XSteering'],
                        bDoRefocusing=kargs['bDoRefocusing'],
                        DistanceConeToFocus=kargs['DistanceConeToFocus'],
                        bUseCT=kargs['bUseCT'],
                        bDisplay=False)
                        
        R.RunCases(targets=Target,ID=kargs['ID'],
                        deviceName=kargs['deviceName'],
                        COMPUTING_BACKEND=COMPUTING_BACKEND,
                        bTightNarrowBeamDomain=True,
                        bForceRecalc=True,
                        basePPW=kargs['basePPW'],
                        basedir=kargs['basedir'],
                        TxMechanicalAdjustmentZ=kargs['TxMechanicalAdjustmentZ'],
                        TxMechanicalAdjustmentX=kargs['TxMechanicalAdjustmentX'],
                        TxMechanicalAdjustmentY=kargs['TxMechanicalAdjustmentY'],
                        ZSteering=kargs['ZSteering'],
                        RotationZ=kargs['RotationZ'],
                        Frequencies=kargs['Frequencies'],
                        XSteering=1e-6,
                        bWaterOnly=True,
                        bDoRefocusing=False,
                        DistanceConeToFocus=kargs['DistanceConeToFocus'],
                        bUseCT=kargs['bUseCT'],
                        bDisplay=False)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))

