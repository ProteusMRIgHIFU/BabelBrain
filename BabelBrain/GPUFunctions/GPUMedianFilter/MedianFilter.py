import logging
logger = logging.getLogger()
import os
import platform
import sys

import numpy as np

from pathlib import Path

try:
    from GPUUtils import InitCUDA,InitOpenCL,InitMetal,get_step_size
except:
    from ..GPUUtils import InitCUDA,InitOpenCL,InitMetal,get_step_size

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

def InitMedianFilter(DeviceName='A6000',GPUBackend='OpenCL'):
    global queue
    global prgcl
    global sel_device
    global ctx
    global knl
    global mf
    global clp
    global cndimage

    kernel_files = [os.path.join(resource_path(), 'median_filter.cpp')]

    if GPUBackend == 'CUDA':
        import cupy as cp
        from cupyx.scipy import ndimage
        clp = cp
        cndimage = ndimage

        ctx,_,sel_device = InitCUDA(DeviceName=DeviceName)

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        clp = pocl

        preamble = '#define _OPENCL\ntypedef unsigned char PixelType;\n'
        queue,prgcl,sel_device,ctx,mf = InitOpenCL(preamble,kernel_files=kernel_files,DeviceName=DeviceName)
        
        # Create kernel from program function
        knl=prgcl.median_reflect

    elif GPUBackend == 'Metal':
        import metalcomputebabel as mc

        clp = mc
        preamble = '#define _METAL\ntypedef unsigned char PixelType;\n' 
        prgcl, sel_device, ctx = InitMetal(preamble,DeviceName=DeviceName,kernel_files=kernel_files)
       
        # Create kernel from program function
        knl=prgcl.function('median_reflect')
    
    
def MedianFilter(data,size,GPUBackend='OpenCL'):

    logger.info(f"\nStarting Median Filter")

    # Check data type and format 
    assert(data.dtype==np.uint8)
    assert(np.isfortran(data)==False) 

    # Determine median filter footprint
    if isinstance(size,int):
        footprint = np.ones((size,size,size),dtype=bool)
    else:
        footprint = np.ones(size,dtype=bool)

    if footprint.shape[0] > 7 or footprint.shape[1] > 7 or footprint.shape[2] > 7:
        raise ValueError(f"GPU Median filter can only handles sizes up to 7x7x7, current size is {footprint.shape}")
    
    output = np.zeros_like(data)
    totalPoints = output.size
    logger.info(f"Total points: {totalPoints}")
    step = get_step_size(sel_device,num_large_buffers=2,data_type=data.dtype,GPUBackend=GPUBackend)

    # Handle array in chunks
    for point in range(0,totalPoints,step):
        # Grab indices for current output section
        slice_start = (point // (output.shape[0] * output.shape[1]))
        slice_end = min(((point + step) // (output.shape[0] * output.shape[1])), output.shape[2])
        logger.info(f"Working on slices {slice_start } to {slice_end} out of {output.shape[2]}")

        # Need slightly larger array to account for median filter size
        padding = footprint.shape[2] // 2 
        actual_start = max(0, slice_start - padding)
        actual_end = min(output.shape[2], slice_end + padding)

        # Grab section of data
        data_section = np.copy(data[:,:,actual_start:actual_end])
        output_section = np.zeros_like(data_section)

        if GPUBackend == 'CUDA':
            with ctx:
                data_section_gpu = clp.asarray(data_section)

                try:
                    output_section_gpu = cndimage.median_filter(data_section_gpu,size)
                except clp.cuda.memory.OutOfMemoryError as e:
                    raise MemoryError(f"{e}\nRan out of GPU memory, suggest lowering PPW")

                output_section = clp.asnumpy(output_section_gpu)
        elif GPUBackend == 'OpenCL':
            
            # Transfer data to gpu
            data_section_pr = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_section)
            output_section_pr = clp.Buffer(ctx, mf.WRITE_ONLY, output_section.nbytes)
            
            # Kernel call
            knl(queue, output_section.shape, 
                None,
                data_section_pr,
                output_section_pr,
                np.int32(data_section.shape[0]),
                np.int32(data_section.shape[1]),
                np.int32(data_section.shape[2]),
                np.int32(footprint.shape[0]),
                np.int32(footprint.shape[1]),
                np.int32(footprint.shape[2]),
                g_times_l=False).wait()
            queue.finish()
            
            # Transfer data back to host
            clp.enqueue_copy(queue,output_section,output_section_pr)
            queue.finish()

            # Release buffers
            data_section_pr.release()
            output_section_pr.release()
            queue.finish()

        elif GPUBackend == 'Metal':
            int_params=np.zeros(6,np.int32)
            int_params[0] = output_section.shape[0]
            int_params[1] = output_section.shape[1]
            int_params[2] = data_section.shape[2]
            int_params[3] = footprint.shape[0]
            int_params[4] = footprint.shape[1]
            int_params[5] = footprint.shape[2]

            data_section_pr = ctx.buffer(data_section)
            output_section_pr = ctx.buffer(output_section)
            int_params_pr = ctx.buffer(int_params)
            
            ctx.init_command_buffer()
            handle=knl(output_section.size,data_section_pr,output_section_pr,int_params_pr)
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if 'arm64' not in platform.platform():
                ctx.sync_buffers((data_section_pr,output_section_pr))
            
            output_section = np.frombuffer(output_section_pr,dtype=np.uint8).reshape(output_section.shape)

        # Record results in output array
        if slice_end == output.shape[2]:
            output[:,:,slice_start:slice_end] = output_section[:,:,slice_start-actual_start:]
        else:
            output[:,:,slice_start:slice_end] = output_section[:,:,slice_start-actual_start:slice_end-actual_end]

    return output