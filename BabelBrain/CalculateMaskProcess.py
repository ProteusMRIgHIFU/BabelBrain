import sys
import platform
import traceback

def CalculateMaskProcess(queue,TxSystem,COMPUTING_BACKEND,devicename,**kargs):
    
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

        if TxSystem =='CTX_500':
            from IntegrationBrainsightTW import BabelDatasetPrepsTW as DataPreps


        elif TxSystem =='H317':
            from IntegrationBrainsightUC import BabelDatasetPreps as DataPreps
        
        if sys.platform not in ['linux','win32']:   
            from GPUMedianFilter import  MedianFilter
            if 'arm64' in platform.platform():
                #we will favour OpenCL implementation for this task
                MedianFilter.InitOpenCL(DeviceName= devicename)
                COMPUTING_BACKEND=2
            else:
                MedianFilter.InitMetal(DeviceName= devicename)
                COMPUTING_BACKEND=3
            DataPreps.InitMedianGPUCallback(MedianFilter.MedianFilterSize7,COMPUTING_BACKEND)

            from GPUVoxelize import Voxelize
            if 'arm64' in platform.platform():
                Voxelize.InitOpenCL(DeviceName= devicename)
                COMPUTING_BACKEND=2
            else:
                Voxelize.InitMetal(DeviceName= devicename)
                COMPUTING_BACKEND=3
            DataPreps.InitVoxelizeGPUCallback(Voxelize.Voxelize,COMPUTING_BACKEND)

        DataPreps.GetSkullMaskFromSimbNIBSSTL(**kargs)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))