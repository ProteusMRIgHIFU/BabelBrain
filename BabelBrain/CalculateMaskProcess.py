import sys
import traceback

def CalculateMaskProcess(queue,COMPUTING_BACKEND,devicename,**kargs):
    
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
        try:
            import BabelDatasetPreps as DataPreps
            from GPUFunctions.GPUVoxelize import Voxelize
            from GPUFunctions.GPUMapping import MappingFilter
            from GPUFunctions.GPUResample import Resample
            from GPUFunctions.GPUBinaryClosing import BinaryClosing
            from GPUFunctions.GPULabel import LabelImage
            from GPUFunctions.GPUMedianFilter import MedianFilter
        except:
            from . import BabelDatasetPreps as DataPreps
            from .GPUFunctions.GPUVoxelize import Voxelize
            from .GPUFunctions.GPUMapping import MappingFilter
            from .GPUFunctions.GPUResample import Resample
            from .GPUFunctions.GPUBinaryClosing import BinaryClosing
            from .GPUFunctions.GPULabel import LabelImage
            from .GPUFunctions.GPUMedianFilter import MedianFilter
        print('sys.platform',sys.platform)

        if COMPUTING_BACKEND == 1:
            gpu_backend = 'CUDA'
        elif COMPUTING_BACKEND == 2:
            gpu_backend = 'OpenCL'
        elif COMPUTING_BACKEND == 3:
            gpu_backend = 'Metal'
        else:
            raise ValueError('Non valid computing backend was given')

        MedianFilter.InitMedianFilter(DeviceName=devicename,GPUBackend=gpu_backend)
        Voxelize.InitVoxelize(DeviceName=devicename,GPUBackend=gpu_backend)
        MappingFilter.InitMapFilter(DeviceName=devicename,GPUBackend=gpu_backend)
        Resample.InitResample(DeviceName=devicename,GPUBackend=gpu_backend)
        BinaryClosing.InitBinaryClosing(DeviceName=devicename,GPUBackend=gpu_backend)
        
        DataPreps.InitMedianGPUCallback(MedianFilter.MedianFilter,COMPUTING_BACKEND)
        DataPreps.InitVoxelizeGPUCallback(Voxelize.Voxelize,COMPUTING_BACKEND)
        DataPreps.InitMappingGPUCallback(MappingFilter.MapFilter,COMPUTING_BACKEND)
        DataPreps.InitResampleGPUCallback(Resample.ResampleFromTo,COMPUTING_BACKEND)
        DataPreps.InitBinaryClosingGPUCallback(BinaryClosing.BinaryClose,COMPUTING_BACKEND)

        # Metal version not ready
        if gpu_backend != 'Metal':
            LabelImage.InitLabel(DeviceName=devicename,GPUBackend=gpu_backend)
            DataPreps.InitLabelImageGPUCallback(LabelImage.LabelImage,COMPUTING_BACKEND)
                    
        DataPreps.GetSkullMaskFromSimbNIBSSTL(**kargs)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))