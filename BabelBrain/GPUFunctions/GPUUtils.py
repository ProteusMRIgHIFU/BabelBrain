import logging
logger = logging.getLogger()
import os
from pathlib import Path
import platform
import sys

_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir


def get_step_size(gpu_device,num_large_buffers,data_type,GPUBackend):

    # Set default step size value
    step = 240000000
    
    # Find optimal step size based on GPU device limitations
    if GPUBackend == 'CUDA':
        import cupy as cp

        # Get available memory
        gpu_available_memory = cp.cuda.Device(gpu_device).mem_info[0]
        logger.info(f"GPU available memory: {gpu_available_memory} bytes")

        # Check for valid response
        if gpu_available_memory == 0:
            print(f"Queried GPU available memory returned 0, most likely a communication issue, will try using default step size ({step})")
            return step

        # Get GPU max buffer size
        max_buffer_size = gpu_available_memory // num_large_buffers
        logger.info(f"GPU max buffer size for {num_large_buffers} array(s): {max_buffer_size} bytes")

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl

        ''' Need a better way to determine max buffer size using pyopencl, default to 240000000 for now'''
        # # Get GPU max buffer size for a single array
        # max_buffer_size_1 = gpu_device.get_info(pocl.device_info.MAX_MEM_ALLOC_SIZE)
        # logger.info(f"GPU max single buffer size: {max_buffer_size_1} bytes")

        # # Get max buffer size for n large arrays
        # global_mem_size = gpu_device.get_info(pocl.device_info.GLOBAL_MEM_SIZE)
        # logger.info(f"GPU total memory: {global_mem_size} bytes")
        # max_buffer_size_2 = global_mem_size // num_large_buffers
        # logger.info(f"GPU max buffer size for {num_large_buffers} array(s): {max_buffer_size_2} bytes")

        # # Select lesser value
        # max_buffer_size = min(max_buffer_size_1,max_buffer_size_2)
        return step

    elif GPUBackend == 'Metal':
        ''' 
        Need a way to determine max buffer size using metalcompute, default to 240000000 for now
        
        * Note that step can't be too large as issues arise with metalcompute where longer
        GPU calls can return incomplete data. Furthermore, systems where AMD GPU is used for both display and
        calculation can also run into watchdog timeout issues and cause the system to crash.
        '''
        
        return step
    
    # Determine largest safe buffer size
    max_buffer_size = int(max_buffer_size * 0.8)  # Use 80% to be safe
    logger.info(f"GPU max safe buffer size: {max_buffer_size} bytes")

    # Determine appropriate step size
    step = max_buffer_size//data_type.itemsize # Accounting for array dtype size
    logger.info(f"Step size: {step}")

    return step


def InitCUDA(preamble=None,kernel_files=None,DeviceName='A6000',build_later=False):
    import cupy as cp

    if preamble is None:
        preamble = ''

    if kernel_files is None:
        kernel_files = ''

    # Obtain list of gpu devices
    devCount = cp.cuda.runtime.getDeviceCount()
    print("Number of CUDA devices found:", devCount)
    if devCount == 0:
        raise SystemError("There are no CUDA devices.")
        
    selDevice = None

    # Select device that matches specified name
    for deviceID in range(0, devCount):
        d=cp.cuda.runtime.getDeviceProperties(deviceID)
        if DeviceName in d['name'].decode('UTF-8'):
            selDevice=cp.cuda.Device(deviceID)
            break

    if selDevice is None:
        raise SystemError("There are no devices supporting CUDA or that matches selected device.")
      
    ctx=selDevice

    # Build program from source code
    kernel_codes = [preamble]
    for k_file in kernel_files:
        with open(k_file, 'r') as f:
            kernel_code = f.read()
            kernel_codes.append(kernel_code)

    complete_kernel = '\n'.join(kernel_codes)
    if build_later:
        prgcl = complete_kernel
    else:
        # Windows sometimes has issues finding CUDA
        if platform.system()=='Windows':
            sys.executable.split('\\')[:-1]
            options=('-I',os.path.join(os.getenv('CUDA_PATH'),'Library','Include'),
                        '-I',str(resource_path()),
                        '--ptxas-options=-v')
        else:
            options=('-I',str(resource_path()))
        
        prgcl = cp.RawModule(code=complete_kernel,options=options)

    return ctx,prgcl,selDevice


def InitOpenCL(preamble=None,kernel_files=None,DeviceName='A6000',build_later=False):
    import pyopencl as pocl
    
    if preamble is None:
        preamble = ''

    if kernel_files is None:
        kernel_files = ''

    # Obtain list of openCL platforms
    Platforms=pocl.get_platforms()
    if len(Platforms)==0:
        raise SystemError("No OpenCL platforms")
    
    # Obtain list of available devices and select one 
    SelDevice=None
    for device in Platforms[0].get_devices():
        print(device.name)
        if DeviceName in device.name:
            SelDevice=device
    if SelDevice is None:
        raise SystemError("No OpenCL device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', SelDevice.name)

        # Print device information
        logger.info(f"  Type: {pocl.device_type.to_string(SelDevice.type)}")
        logger.info(f"  Max Compute Units: {SelDevice.max_compute_units}")
        logger.info(f"  Max Work Group Size: {SelDevice.max_work_group_size}")
        logger.info(f"  Max Work Item Dimensions: {SelDevice.max_work_item_dimensions}")
        logger.info(f"  Max Work Item Sizes: {SelDevice.max_work_item_sizes}")
        logger.info(f"  Global Memory Size: {SelDevice.global_mem_size / (1024 ** 2)} MB")
        logger.info(f"  Local Memory Size: {SelDevice.local_mem_size / 1024} KB")
        logger.info(f"  Max Memory Allocation Size: {SelDevice.max_mem_alloc_size / (1024 ** 2)} MB")
        logger.info(f"  OpenCL C Version: {SelDevice.opencl_c_version}\n")

    # Create context for selected device
    ctx = pocl.Context([SelDevice])
    
    # Build program from source code
    kernel_codes = [preamble]
    for k_file in kernel_files:
        with open(k_file, 'r') as f:
            kernel_code = f.read()
            kernel_codes.append(kernel_code)

    complete_kernel = '\n'.join(kernel_codes)
    if build_later:
        prgcl = complete_kernel
    else:
        prgcl = pocl.Program(ctx,complete_kernel).build()

    # Create command queue for selected device
    queue = pocl.CommandQueue(ctx)

    # Allocate device memory
    mf = pocl.mem_flags

    return queue, prgcl, SelDevice, ctx, mf


def InitMetal(preamble=None,kernel_files=None,DeviceName='A6000',build_later=False):
    import metalcomputebabel as mc

    if preamble is None:
        preamble = ''

    if kernel_files is None:
        kernel_files = ''

    devices = mc.get_devices()
    SelDevice=None
    for n,dev in enumerate(devices):
        if DeviceName in dev.deviceName:
            SelDevice=dev
            break
    if SelDevice is None:
        raise SystemError("No Metal device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', dev.deviceName)
    
    ctx = mc.Device(n)
    if 'arm64' not in platform.platform():
        ctx.set_external_gpu(1) 

    # Build program from source code
    kernel_codes = [preamble]
    for k_file in kernel_files:
        with open(k_file, 'r') as f:
            kernel_code = f.read()
            kernel_codes.append(kernel_code)

    complete_kernel = '\n'.join(kernel_codes)
    if build_later:
        prgcl = complete_kernel
    else:
        prgcl = ctx.kernel(complete_kernel)

    return prgcl, SelDevice, ctx