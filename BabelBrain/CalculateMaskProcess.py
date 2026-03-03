import sys
import traceback

def calculate_mask_process(queue,COMPUTING_BACKEND,devicename,**kargs):
    
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
            from GPUFunctions.GPUVoxelize import voxelize
            from GPUFunctions.GPUMapping import MappingFilter
            from GPUFunctions.GPUResample import Resample
            from GPUFunctions.GPUBinaryClosing import BinaryClosing
            from GPUFunctions.GPULabel import label_image
            from GPUFunctions.GPUMedianFilter import median_filter
        except:
            from . import BabelDatasetPreps as DataPreps
            from .GPUFunctions.GPUVoxelize import voxelize
            from .GPUFunctions.GPUMapping import MappingFilter
            from .GPUFunctions.GPUResample import Resample
            from .GPUFunctions.GPUBinaryClosing import BinaryClosing
            from .GPUFunctions.GPULabel import label_image
            from .GPUFunctions.GPUMedianFilter import median_filter
        print('sys.platform',sys.platform)

        if COMPUTING_BACKEND == 1:
            gpu_backend = 'CUDA'
        elif COMPUTING_BACKEND == 2:
            gpu_backend = 'OpenCL'
        elif COMPUTING_BACKEND == 3:
            gpu_backend = 'Metal'
        elif COMPUTING_BACKEND == 4:
            gpu_backend = 'MLX'
        else:
            raise ValueError('Non valid computing backend was given')

        median_filter.init_median_filter(DeviceName=devicename,GPUBackend=gpu_backend)
        voxelize.init_voxelize(DeviceName=devicename,GPUBackend=gpu_backend)
        MappingFilter.init_map_filter(DeviceName=devicename,GPUBackend=gpu_backend)
        Resample.init_resample(DeviceName=devicename,GPUBackend=gpu_backend)
        BinaryClosing.init_binary_closing(DeviceName=devicename,GPUBackend=gpu_backend)
        
        DataPreps.init_median_gpu_callback(median_filter.median_filter,COMPUTING_BACKEND)
        if COMPUTING_BACKEND != 2: #something got broken with the OpenCL version , #disabling temporarily
            DataPreps.init_voxelize_gpu_callback(voxelize.voxelize,COMPUTING_BACKEND)
        DataPreps.init_mapping_gpu_callback(MappingFilter.map_filter,COMPUTING_BACKEND)
        DataPreps.init_resample_gpu_callback(Resample.resample_from_to,COMPUTING_BACKEND)
        DataPreps.init_binary_closing_gpu_callback(BinaryClosing.binary_close,COMPUTING_BACKEND)

        # Metal version not ready
        if gpu_backend not in ['Metal','MLX']:
            label_image.init_label(DeviceName=devicename,GPUBackend=gpu_backend)
            DataPreps.init_label_image_gpu_callback(label_image.label_image,COMPUTING_BACKEND)
                    
        DataPreps.get_skull_mask_from_simbnibs_stl(**kargs)
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))