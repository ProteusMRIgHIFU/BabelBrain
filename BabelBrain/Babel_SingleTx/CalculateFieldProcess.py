import sys
import platform
from TranscranialModeling.BabelIntegrationSingle import RUN_SIM
import traceback

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
        R.RunCases(targets=Target, 
                        bTightNarrowBeamDomain=True,
                        bForceRecalc=True,
                        bDisplay=False,
                        **kargs)

        R.RunCases(targets=Target, 
                        bTightNarrowBeamDomain=True,
                        bForceRecalc=True,
                        bWaterOnly=True,
                        bDisplay=False,
                        **kargs)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))
