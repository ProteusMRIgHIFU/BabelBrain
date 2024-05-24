import numpy as np

def get_step_size(gpu_device,num_buffers,data,GPUBackend):

    # Get GPU device limitations
    if GPUBackend == 'CUDA':
        pass
    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        max_alloc_size = gpu_device.get_info(pocl.device_info.MAX_MEM_ALLOC_SIZE)
        global_mem_size = gpu_device.get_info(pocl.device_info.GLOBAL_MEM_SIZE)
    elif GPUBackend == 'Metal':
        pass

    # Determine largest safe buffer size
    max_buffer_size = min(global_mem_size // num_buffers, max_alloc_size)
    max_buffer_size = int(max_buffer_size * 0.8)  # use 80% to be safe

    # Use array size if less than largest safe buffer size
    # buffer_size = min(max_buffer_size, data.nbytes)
    # step = buffer_size//data.dtype.itemsize
    max_buffer_size = max_buffer_size//data.dtype.itemsize
    max_work_item_size = gpu_device.max_work_item_sizes[0] * gpu_device.max_work_item_sizes[1] * gpu_device.max_work_item_sizes[2]
    step = min(max_buffer_size, max_work_item_size)
    return step