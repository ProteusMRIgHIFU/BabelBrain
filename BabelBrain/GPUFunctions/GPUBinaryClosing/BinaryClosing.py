import logging
logger = logging.getLogger()
import operator
import os
os.environ['PYOPENCL_NO_CACHE']='1'

import numpy as np
from pathlib import Path
import platform
import sys
from scipy.ndimage._morphology import generate_binary_structure, _center_is_true
from scipy.ndimage._ni_support import _get_output

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

def InitBinaryClosing(DeviceName='A6000',GPUBackend='OpenCL'):
    global queue 
    global prgcl
    global sel_device
    global ctx
    global knl
    global mf
    global clp
    global cndimage

    kernel_files = [os.path.join(resource_path(), 'binary_closing.cpp')
    ]

    if GPUBackend == 'CUDA':
        import cupy as cp
        from cupyx.scipy import ndimage
        clp = cp
        cndimage = ndimage

        preamble = '#define _CUDA'
        ctx,prgcl,sel_device = InitCUDA(preamble,kernel_files,DeviceName)

        # Create kernels from program function
        knl = prgcl.get_function('binary_erosion')

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        clp = pocl
        preamble = '#define _OPENCL'
        queue,prgcl,sel_device,ctx,mf = InitOpenCL(preamble,kernel_files,DeviceName)
        
        # Create kernels from program function
        knl=prgcl.binary_erosion
    elif GPUBackend == 'Metal':
        import metalcomputebabel as mc
        clp = mc
        preamble = '#define _METAL'
        prgcl, sel_device, ctx = InitMetal(preamble,kernel_files,DeviceName)

        # Create kernels from program function
        knl=prgcl.function('binary_erosion')

def erode_kernel(input, structure, output, offsets, border_value, center_is_true, invert, GPUBackend='OpenCL'):

    if invert:
        border_value = int(not border_value)
        true_val = 0
        false_val = 1
    else:
        true_val = 1
        false_val = 0

    padding = structure.shape[0] - offsets[0]

    input = input.astype(np.uint8)
    structure = structure.astype(np.uint8)
    output = np.zeros_like(output, dtype=np.uint8)
    int_params=np.zeros(20,np.int32)
    int_params[0]  = input.shape[0]
    int_params[1]  = input.shape[1]
    int_params[2]  = input.shape[2]
    int_params[3]  = input.strides[0]
    int_params[4]  = input.strides[1]
    int_params[5]  = input.strides[2]
    int_params[6]  = true_val
    int_params[7]  = false_val
    int_params[8]  = border_value
    int_params[9]  = center_is_true
    int_params[10] = structure.shape[0] 
    int_params[11] = structure.shape[1] 
    int_params[12] = structure.shape[2] 
    int_params[13] = offsets[0]
    int_params[14] = offsets[1]
    int_params[15] = offsets[2]
    int_params[18] = padding

    assert(np.isfortran(output)==False)
    assert(np.isfortran(input)==False)
    assert(np.isfortran(structure)==False)
    assert(np.isfortran(int_params)==False)

    totalPoints = output.size
    logger.info(f"Total points: {totalPoints}")
    step = get_step_size(sel_device,num_large_buffers=2,data_type=output.dtype,GPUBackend=GPUBackend)

    for point in range(0,totalPoints,step):

        # Grab indices for current output section
        slice_start = (point // (output.shape[1] * output.shape[2]))
        slice_end = min(((point + step) // (output.shape[1] * output.shape[2])),output.shape[0])
        logger.info(f"\nWorking on slices {slice_start} to {slice_end} out of {output.shape[0]}")
        
        # Need slightly larger array to account for binary closing structure size
        actual_start = max(0,slice_start-padding)
        actual_end = min(output.shape[0],slice_end+padding)
        logger.debug(f"Actual start: {actual_start}, Actual end {actual_end}")

        # Grab sections of data
        input_section = np.copy(input[actual_start:actual_end,:,:])
        output_section = np.copy(output[actual_start:actual_end,:,:])
        logger.debug(f"section shape: {input_section.shape}")

        current_position = actual_start * output.shape[1] * output.shape[2]
        base_32 = current_position // (2**31)
        if base_32 > 0:
            logger.debug(f"Curren position initially: {current_position}")
        current_position = current_position - (base_32 * (2**31))
        logger.debug(f"Current position: {current_position}")
        logger.debug(f"base32: {base_32}")
    
        int_params[16] = current_position
        int_params[17] = base_32
        int_params[19] = output_section.size

        if GPUBackend=='CUDA':
            with ctx:
                # Move input data from host to device memory
                input_section_gpu = clp.asarray(input_section)
                structure_gpu = clp.asarray(structure)
                int_params_gpu = clp.asarray(int_params)
                output_section_gpu = clp.asarray(output_section)

                # Define block and grid sizes
                block_size = (8, 8, 8)
                grid_size = ((output_section.shape[0] + block_size[0] - 1) // block_size[0],
                             (output_section.shape[1] + block_size[1] - 1) // block_size[1],
                             (output_section.shape[2] + block_size[2] - 1) // block_size[2])
                
                # Deploy kernel
                knl(grid_size,block_size,
                    (input_section_gpu,
                    structure_gpu,
                    int_params_gpu,
                    output_section_gpu)
                    )

                # Move kernel output data back to host memory
                output_section = output_section_gpu.get()

        elif GPUBackend=='OpenCL':

            # Move input data from host to device memory
            input_section_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_section)
            structure_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=structure)
            int_params_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int_params)
            output_section_gpu = clp.Buffer(ctx, mf.WRITE_ONLY, output_section.nbytes)

            # Deploy kernel
            knl(queue, output_section.shape,
                None,
                input_section_gpu,
                structure_gpu,
                int_params_gpu,
                output_section_gpu,
                ).wait()
            queue.finish()

            # Move kernel output data back to host memory
            clp.enqueue_copy(queue, output_section, output_section_gpu).wait()
            queue.finish()

            # Release GPU memory
            input_section_gpu.release()
            structure_gpu.release()
            int_params_gpu.release()
            output_section_gpu.release()
            queue.finish()
        elif GPUBackend=='Metal':
            
            # Move input data from host to device memory
            input_section_gpu = ctx.buffer(input_section)
            structure_gpu = ctx.buffer(structure)
            int_params_gpu = ctx.buffer(int_params)
            output_section_gpu = ctx.buffer(output_section)
            
            # Deploy kernel
            ctx.init_command_buffer()
            handle=knl(output_section.size,input_section_gpu,structure_gpu,int_params_gpu,output_section_gpu)
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if 'arm64' not in platform.platform():
                ctx.sync_buffers((int_params_gpu,output_section_gpu))

            # Move kernel output data back to host memory
            output_section = np.frombuffer(output_section_gpu,dtype=np.uint8).reshape(output_section.shape)
        else:
            raise ValueError("Unknown gpu backend was selected")
        
        if slice_end == output.shape[0]:
            output[slice_start:slice_end,:,:] = output_section[slice_start-actual_start:,:,:]
        else:
            output[slice_start:slice_end,:,:] = output_section[slice_start-actual_start:slice_end-actual_end,:,:]
    
    return output


def _origins_to_offsets(origins, w_shape):
    return tuple(x//2+o for x, o in zip(w_shape, origins))


def _get_inttype(input):
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    return 'int' if nbytes < (1 << 31) else 'ptrdiff_t'


def _center_is_true(structure, origin):
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape, origin)])
    return bool(structure[coor])  # device synchronization


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if isinstance(arg, str):
        return [conv(arg)] * ndim
    try:
        arg = iter(arg)
    except TypeError:
        return [conv(arg)] * ndim
    lst = [conv(x) for x in arg]
    if len(lst) != ndim:
        msg = "{} must have length equal to input rank".format(name)
        raise RuntimeError(msg)
    return lst


def _binary_erosion_modified(input, structure, iterations, mask, output, border_value,
                             origin, invert, brute_force=True, GPUBackend='OpenCL'):
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise TypeError('iterations parameter should be an integer')

    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
        all_weights_nonzero = input.ndim == 1
        center_is_true = True
        default_structure = True
    else:
        structure = structure.astype(dtype=bool, copy=False)
        # transfer to CPU for use in determining if it is fully dense
        # structure_cpu = cupy.asnumpy(structure)
        default_structure = False
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.c_contiguous:
        structure = np.ascontiguousarray(structure)
    if structure.size < 1:
        raise RuntimeError('structure must not be empty')

    if mask is not None:
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
        if not mask.flags.c_contiguous:
            ''' Changed to numpy '''
            # mask = cupy.ascontiguousarray(mask)
            mask = np.ascontiguousarray(mask)
        masked = True
    else:
        masked = False
    ''' Copied function to this file '''
    # origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    origin = _fix_sequence_arg(origin, input.ndim, 'origin', int)

    ''' Changed to numpy '''
    # if isinstance(output, cupy.ndarray):
    if isinstance(output, np.ndarray):
        if output.dtype.kind == 'c':
            raise TypeError('Complex output type not supported')
    else:
        output = bool

    ''' Call scipy's version of _get_output instead of cupy's '''
    # output = _util._get_output(output, input)
    output = _get_output(output, input)

    ''' Changed to numpy '''
    # temp_needed = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    temp_needed = np.may_share_memory(output, input)

    if temp_needed:
        # input and output arrays cannot share memory
        temp = output

        ''' Call scipy's version of _get_output instead of cupy's '''
        # output = _util._get_output(output.dtype, input)
        output = _get_output(output.dtype, input)
    if structure.ndim == 0:
        # kernel doesn't handle ndim=0, so special case it here
        if float(structure):
            ''' Changed to numpy '''
            # output[...] = cupy.asarray(input, dtype=bool)
            output[...] = np.asarray(input, dtype=bool)
        else:
            ''' Changed to numpy '''
            # output[...] = ~cupy.asarray(input, dtype=bool)
            output[...] = ~np.asarray(input, dtype=bool)
        return output
    
    origin = tuple(origin)

    ''' Copied functions to this file '''
    # int_type = _util._get_inttype(input)
    # offsets = _filters_core._origins_to_offsets(origin, structure.shape)
    int_type = _get_inttype(input)
    offsets = _origins_to_offsets(origin, structure.shape)

    if not default_structure:
        # synchronize required to determine if all weights are non-zero
        ''' Changed to numpy '''
        # nnz = int(cupy.count_nonzero(structure))
        nnz = int(np.count_nonzero(structure))

        all_weights_nonzero = nnz == structure.size
        if all_weights_nonzero:
            center_is_true = True
        else:
            ''' Call scipy's version of _center_is_true instead of cupy's'''
            center_is_true = _center_is_true(structure, origin)

    ''' Removed '''
    # erode_kernel = _get_binary_erosion_kernel(
    #     structure.shape, int_type, offsets, center_is_true, border_value,
    #     invert, masked, all_weights_nonzero,
    # )

    if iterations == 1:
        ''' Removed since mask is always false '''
        # if masked:
        #     output = erode_kernel(input, structure, mask, output)
        # else:
        #     output = erode_kernel(input, structure, output)
        ''' Call our custom kernel instead '''
        output = erode_kernel(input, structure, output, offsets, border_value, center_is_true, invert, GPUBackend)
    elif center_is_true and not brute_force:
        raise NotImplementedError(
            'only brute_force iteration has been implemented'
        )
    else:
        ''' Changed to numpy '''
        # if cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS'):
        #     raise ValueError('output and input may not overlap in memory')
        # tmp_in = cupy.empty_like(input, dtype=output.dtype)
        if np.may_share_memory(output, input):
            raise ValueError('output and input may not overlap in memory')
        tmp_in = np.empty_like(input, dtype=output.dtype)
        
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in

        ''' Removed since mask is always false '''
        # if masked:
        #     tmp_out = erode_kernel(input, structure, mask, tmp_out)
        # else:
        #     tmp_out = erode_kernel(input, structure, tmp_out)
        tmp_out = erode_kernel(input, structure, output, offsets, border_value, center_is_true, invert, GPUBackend)
        
        # TODO: kernel doesn't return the changed status, so determine it here
        changed = not (input == tmp_out).all()  # synchronize!
        ii = 1
        while ii < iterations or ((iterations < 1) and changed):
            tmp_in, tmp_out = tmp_out, tmp_in

            ''' Removed since mask is always false '''
            # if masked:
            #     tmp_out = erode_kernel(tmp_in, structure, mask, tmp_out)
            # else:
            #     tmp_out = erode_kernel(tmp_in, structure, tmp_out)
            tmp_out = erode_kernel(input, structure, output, offsets, border_value, center_is_true, invert, GPUBackend)
            
            changed = not (tmp_in == tmp_out).all()
            ii += 1
            if not changed and (not ii & 1):  # synchronize!
                # can exit early if nothing changed
                # (only do this after even number of tmp_in/out swaps)
                break
        output = tmp_out
    if temp_needed:
        ''' Changed to numpy '''
        # from cupy import _core
        # _core.elementwise_copy(output, temp)
        np.copyto(temp,output)

        output = temp
    return output

def binary_erosion_modified(input, structure=None, iterations=1, mask=None, output=None,
                   border_value=0, origin=0, brute_force=False, GPUBackend='OpenCL'):
    """Multidimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.
    """
    return _binary_erosion_modified(input, structure, iterations, mask, output,
                           border_value, origin, 0, brute_force, GPUBackend)


def binary_dilation_modified(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0, brute_force=False, GPUBackend='OpenCL'):
    """Multidimensional binary dilation with the given structuring element."""

    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    
    ''' Copied function to this file'''
    # origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    origin = _fix_sequence_arg(origin, input.ndim, 'origin', int)

    structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1
    
    ''' Call modified version of function'''
    # return _binary_erosion(input, structure, iterations, mask, output,
    #                        border_value, origin, 1, brute_force)
    return _binary_erosion_modified(input, structure, iterations, mask, output,
                                    border_value, origin, 1, brute_force, GPUBackend)


def BinaryClose(input, structure, iterations=1, output=None, origin=0,
                   mask=None, border_value=0, brute_force=False, GPUBackend='OpenCL'):
    """
    Modified from cupy's binary_closing function to work for OpenCL and Metal 
    backends in addition to CUDA.

    see cupyx.scipy.nd_image._morphology.binary_closing for reference
    """

    logger.info("\nStarting Binary Closing")
    
    # If using cuda, we try to complete using cupy's cndimage.binary_closing function
    # since it is more robust than our custom kernel. Switch to looping method if GPU
    # memory is not sufficient for array size.
    if GPUBackend=='CUDA':
        try:
            with ctx:
                input_gpu = clp.asarray(input)
                structure_gpu = clp.asarray(structure)

                output_gpu = cndimage.binary_closing(input_gpu,structure=structure_gpu)

                output = output_gpu.get()
            return output
        except clp.cuda.memory.OutOfMemoryError as e:
            print(f"{e}\nNot enough memory to complete binary closing in one go. Switching to looping method")

    ''' Removed since we always pass a structure'''
    # if structure is None:
    #     rank = input.ndim
    #     ''' Call scipy's version of generate_binary_structure instead of cupy's'''
    #     structure = generate_binary_structure(rank, 1)

    ''' Call modified versions of functions'''
    # tmp = binary_dilation_modified(input, structure, iterations, mask, None,
    #                                border_value, origin, brute_force, GPUBackend)
    # return binary_erosion_modified(tmp, structure, iterations, mask, output,
    #                                border_value, origin, brute_force, GPUBackend)
    tmp = binary_dilation_modified(input, structure, iterations, mask, None,
                                    border_value, origin, brute_force, GPUBackend)
    return binary_erosion_modified(tmp, structure, iterations, mask, output,
                                    border_value, origin, brute_force, GPUBackend)