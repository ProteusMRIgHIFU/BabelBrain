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

        import BabelDatasetPreps as DataPreps
        
        if sys.platform not in ['linux','win32']:   
            from GPUMedianFilter import  MedianFilter
            from GPUVoxelize import Voxelize
            from GPUMapping import MappingFilter
        
            if COMPUTING_BACKEND==2:
                MedianFilter.InitOpenCL(DeviceName= devicename)
                Voxelize.InitOpenCL(DeviceName= devicename)
                MappingFilter.InitOpenCL(DeviceName= devicename)
            else:
                MedianFilter.InitMetal(DeviceName= devicename)
                Voxelize.InitMetal(DeviceName= devicename)
                MappingFilter.InitMetal(DeviceName= devicename)
            
            DataPreps.InitMedianGPUCallback(MedianFilter.MedianFilterSize7,COMPUTING_BACKEND)
            DataPreps.InitVoxelizeGPUCallback(Voxelize.Voxelize,COMPUTING_BACKEND)
            DataPreps.InitMappingGPUCallback(MappingFilter.MapFilter,COMPUTING_BACKEND)
        else:
            from GPUVoxelize import Voxelize
            if COMPUTING_BACKEND==1:
                Voxelize.InitCUDA(DeviceName= devicename)
            else:
                Voxelize.InitOpenCL(DeviceName= devicename)
            DataPreps.InitVoxelizeGPUCallback(Voxelize.Voxelize,COMPUTING_BACKEND)
                    
        DataPreps.GetSkullMaskFromSimbNIBSSTL(**kargs)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))