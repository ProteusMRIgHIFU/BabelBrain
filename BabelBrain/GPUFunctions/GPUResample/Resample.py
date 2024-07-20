import logging
logger = logging.getLogger()
import os
from pathlib import Path
import platform
import sys
import warnings

from nibabel import processing
from nibabel.affines import AffineError, to_matvec
from nibabel.imageclasses import spatial_axes_first
from nibabel.nifti1 import Nifti1Image
import numpy as np
import numpy.linalg as npl
from scipy.ndimage._interpolation import spline_filter, _prepad_for_spline_filter
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

def InitResample(DeviceName='A6000',GPUBackend='OpenCL'):
    global queue 
    global prgcl
    global sel_device
    global ctx
    global knl_at
    global knl_sf
    global mf
    global clp
    global cndimage

    kernel_files = [
       os.path.join(resource_path(), 'affine_transform.cpp'),
        # base_path + os.sep + 'BabelBrain' + os.sep + 'GPUFunctions' + os.sep + 'GPUResample' + os.sep + 'spline_filter.cpp'
    ]

    if GPUBackend == 'CUDA':
        import cupy as cp
        from cupyx.scipy import ndimage
        clp = cp
        cndimage = ndimage

        preamble = '#define _CUDA'
        ctx,prgcl,sel_device = InitCUDA(preamble,kernel_files=kernel_files,DeviceName=DeviceName)

        # Create kernels from program function
        knl_at = prgcl.get_function('affine_transform')

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        clp = pocl

        preamble = '#define _OPENCL'
        queue,prgcl,sel_device,ctx,mf = InitOpenCL(preamble,kernel_files=kernel_files,DeviceName=DeviceName)
        
        # Create kernels from program function
        knl_at=prgcl.affine_transform
        # knl_sf=prgcl.spline_filter_3d

    elif GPUBackend == 'Metal':
        import metalcomputebabel as mc
        clp = mc
        
        preamble = '#define _METAL'
        prgcl, sel_device, ctx = InitMetal(preamble,kernel_files=kernel_files,DeviceName=DeviceName)
       
        # Create kernels from program function
        knl_at=prgcl.function('affine_transform')
        # knl_sf=prgcl.function('spline_filter_3d')


def _check_cval_modified(mode, cval, integer_output):
    ''' Changed to numpy '''
    if mode == 'constant' and integer_output and not np.isfinite(cval):
        raise NotImplementedError("Non-finite cval is not supported for "
                                  "outputs with integer dtype.")
    

def _filter_input_modified(image, prefilter, mode, cval, order):
    """Perform spline prefiltering when needed.

    Spline orders > 1 need a prefiltering stage to preserve resolution.

    For boundary modes without analytical spline boundary conditions, some
    prepadding of the input with cupy.pad is used to maintain accuracy.
    ``npad`` is an integer corresponding to the amount of padding at each edge
    of the array.
    """

    if not prefilter or order < 2:
        ''' Changed to numpy'''
        return (np.ascontiguousarray(image), 0)
    
    ''' Call scipy version instead of cupy version '''
    padded, npad = _prepad_for_spline_filter(image, mode, cval)

    ''' Changed to numpy '''
    float_dtype = np.promote_types(image.dtype, np.float32)

    ''' Call scipy version instead of cupy version '''
    filtered = spline_filter(padded, order, output=float_dtype, mode=mode)
    
    ''' Changed to numpy'''
    return np.ascontiguousarray(filtered), npad


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    ''' Untouched from cupy version '''
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


def _check_parameter(func_name, order, mode):
    ''' Untouched from cupy version '''
    if order is None:
        warnings.warn(f'Currently the default order of {func_name} is 1. In a '
                      'future release this may change to 3 to match '
                      'scipy.ndimage ')
    elif order < 0 or 5 < order:
        raise ValueError('spline order is not supported')

    if mode not in ('constant', 'grid-constant', 'nearest', 'mirror',
                    'reflect', 'grid-mirror', 'wrap', 'grid-wrap', 'opencv',
                    '_opencv_edge'):
        raise ValueError('boundary mode ({}) is not supported'.format(mode))
    

def affine_transform_prep(input, matrix, offset=0.0, output_shape=None, output=None,
                     order=3, mode='constant', cval=0.0, prefilter=True, GPUBackend='OpenCL', *,
                     texture_memory=False):
    """ Modified from cupyx.scipy.nd_image._interpolation's affine_transform function 
    to use numpy arrays and return values to be used for call to modified affine_transform
    kernel
    """
    
    ''' Added '''
    input = np.asarray(input)

    ''' Removed '''
    # if texture_memory:
    #     if runtime.is_hip:
    #         raise RuntimeError(
    #             'HIP currently does not support texture acceleration')
    #     tm_interp = 'linear' if order > 0 else 'nearest'
    #     return _texture.affine_transformation(data=input,
    #                                           transformation_matrix=matrix,
    #                                           output_shape=output_shape,
    #                                           output=output,
    #                                           interpolation=tm_interp,
    #                                           mode=mode,
    #                                           border_value=cval)
    
    _check_parameter('affine_transform', order, mode)

    offset = _fix_sequence_arg(offset, input.ndim, 'offset', float)

    if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
        raise RuntimeError('no proper affine matrix provided')
    if matrix.ndim == 2:
        if matrix.shape[0] == matrix.shape[1] - 1:
            offset = matrix[:, -1]
            matrix = matrix[:, :-1]
        elif matrix.shape[0] == input.ndim + 1:
            offset = matrix[:-1, -1]
            matrix = matrix[:-1, :-1]
        if matrix.shape != (input.ndim, input.ndim):
            raise RuntimeError('improper affine shape')

    ''' Removed '''
    # if mode == 'opencv':
    #     m = cupy.zeros((input.ndim + 1, input.ndim + 1))
    #     m[:-1, :-1] = matrix
    #     m[:-1, -1] = offset
    #     m[-1, -1] = 1
    #     m = cupy.linalg.inv(m)
    #     m[:2] = cupy.roll(m[:2], 1, axis=0)
    #     m[:2, :2] = cupy.roll(m[:2, :2], 1, axis=1)
    #     matrix = m[:-1, :-1]
    #     offset = m[:-1, -1]

    if output_shape is None:
        output_shape = input.shape

    ''' Removed '''
    # if mode == 'opencv' or mode == '_opencv_edge':
    #     if matrix.ndim == 1:
    #         matrix = cupy.diag(matrix)
    #     coordinates = cupy.indices(output_shape, dtype=cupy.float64)
    #     coordinates = cupy.dot(matrix, coordinates.reshape((input.ndim, -1)))
    #     coordinates += cupy.expand_dims(cupy.asarray(offset), -1)
    #     ret = _util._get_output(output, input, shape=output_shape)
    #     ret[:] = map_coordinates(input, coordinates, ret.dtype, order, mode,
    #                              cval, prefilter).reshape(output_shape)
    #     return ret
    
    ''' Changed to numpy '''
    # matrix = matrix.astype(cupy.float64, copy=False)
    matrix = matrix.astype(np.float64, copy=False)
    ndim = input.ndim

    ''' Replaced cupy code with scipy code'''
    # output = _util._get_output(output, input, shape=output_shape)
    complex_output = np.iscomplexobj(input)
    output = _get_output(output, input, shape=output_shape,complex_output=complex_output)
    
    if input.dtype.kind in 'iu':
        ''' Changed to numpy '''
        # input = input.astype(cupy.float32)
        input = input.astype(np.float32)
    ''' Modified call '''
    # filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)
    filtered, nprepad = _filter_input_modified(input, prefilter, mode, cval, order)

    integer_output = output.dtype.kind in 'iu'

    ''' Modified call '''
    # _util._check_cval(mode, cval, integer_output)
    _check_cval_modified(mode, cval, integer_output)

    ''' Changed to numpy '''
    # _prod = cupy._core.internal.prod
    # large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
    large_int = max(np.prod(input.shape), np.prod(output_shape)) > 1 << 31

    ''' Changed since we don't need to generate kernel, matrix.ndim == 3, and need numpy'''
    # if matrix.ndim == 1:
    #     offset = cupy.asarray(offset, dtype=cupy.float64)
    #     offset = -offset / matrix
    #     kern = _interp_kernels._get_zoom_shift_kernel(
    #         ndim, large_int, output_shape, mode, cval=cval, order=order,
    #         integer_output=integer_output, nprepad=nprepad)
    #     kern(filtered, offset, matrix, output)
    # else:
    #     kern = _interp_kernels._get_affine_kernel(
    #         ndim, large_int, output_shape, mode, cval=cval, order=order,
    #         integer_output=integer_output, nprepad=nprepad)
    #     m = cupy.zeros((ndim, ndim + 1), dtype=cupy.float64)
    #     m[:, :-1] = matrix
    #     m[:, -1] = cupy.asarray(offset, dtype=cupy.float64)
    #     kern(filtered, m, output)
    m = np.zeros((ndim, ndim + 1), dtype=np.float64)
    m[:, :-1] = matrix
    m[:, -1] = np.asarray(offset, dtype=np.float64)

    return filtered, m, output, mode, cval, order, integer_output, large_int

def ResampleFromTo(from_img, to_vox_map,order=3,mode="constant",cval=0.0,out_class=Nifti1Image,GPUBackend='OpenCL'):

    logger.info(f"\nStarting Resample")

    # This check requires `shape` attribute of image
    if not spatial_axes_first(from_img):
        raise ValueError(
            f'Cannot predict position of spatial axes for Image type {type(from_img)}'
        )
    
    try:
        to_shape, to_affine = to_vox_map.shape, to_vox_map.affine
    except AttributeError:
        to_shape, to_affine = to_vox_map
    
    a_to_affine = processing.adapt_affine(to_affine, len(to_shape))
    
    if out_class is None:
        out_class = from_img.__class__
    
    from_n_dim = len(from_img.shape)
    
    if from_n_dim < 3:
        raise AffineError('from_img must be at least 3D')
    
    a_from_affine = processing.adapt_affine(from_img.affine, from_n_dim)
    to_vox2from_vox = npl.inv(a_from_affine).dot(a_to_affine)
    rzs, trans = to_matvec(to_vox2from_vox)
    
    # If using cuda, we try to complete using cupy's cndimage.affine_transform function
    # since it is more accurate than our custom kernel. Switch to looping method if GPU
    # memory is not sufficient for array size.
    if GPUBackend=='CUDA':
        try:
            with ctx:
                image_gpu = clp.asarray(from_img.dataobj)
                rzs_gpu=clp.asarray(rzs)

                data_gpu = cndimage.affine_transform(
                image_gpu, rzs_gpu, trans, to_shape, order=order, mode=mode, cval=cval
                )
                data = clp.asnumpy(data_gpu)
            return out_class(data, to_affine, from_img.header)
        except clp.cuda.memory.OutOfMemoryError as e:
            print(f"{e}\nNot enough memory to complete resample in one go. Switching to looping method")
    
    # Looping method
    filtered, m, output, mode, cval, order, integer_output, large_int = affine_transform_prep(from_img.dataobj, rzs, trans, to_shape, order=order, mode=mode, cval=cval, GPUBackend=GPUBackend)
    
    filtered = filtered.astype(np.float32, copy=False)
    m = m.astype(np.float32, copy=False)
    output = output.astype(np.float32, copy=False)
    float_params = np.zeros(2,dtype=np.float32)
    int_params = np.zeros(10, dtype=np.uint32)
    float_params[0] = cval
    int_params[0] = order
    int_params[1] = filtered.shape[0]
    int_params[2] = filtered.shape[1]
    int_params[3] = filtered.shape[2]
    int_params[4] = output.shape[0]
    int_params[5] = output.shape[1]
    int_params[6] = output.shape[2]

    assert(np.isfortran(output)==False)
    assert(np.isfortran(m)==False)
    assert(np.isfortran(filtered)==False)

    totalPoints = output.size
    step = get_step_size(sel_device,num_large_buffers=1,data_type=output.dtype,GPUBackend=GPUBackend)
    logger.info(f"Total points: {totalPoints}")
    for point in range(0,totalPoints,step):

        # Grab indices for current output section
        slice_start = (point // (output.shape[1] * output.shape[2]))
        slice_end = min(((point + step) // (output.shape[1] * output.shape[2])),output.shape[0])
        logger.info(f"Working on slices {slice_start} to {slice_end} out of {output.shape[0]}")
        
        # Grab section of output
        output_section = np.copy(output[slice_start:slice_end,:,:])
        
        # Keep track of overall position in output array
        output_start = slice_start * output.shape[1] * output.shape[2]
        
        # Since we run into issues sending numbers larger than 32 bits due to buffer size restrictions, 
        # we check the size here, send info to kernel, and create number there as workaround
        base_32 = output_start // (2**32)
        output_start = output_start - (base_32 * (2**32))

        int_params[7] = output_start
        int_params[8] = base_32
        int_params[9] = output_section.size

        if GPUBackend == 'CUDA':
            with ctx:
                # Move input data from host to device memory
                filtered_gpu = clp.asarray(filtered)
                m_gpu = clp.asarray(m)
                output_section_gpu = clp.asarray(output_section)
                float_params_gpu = clp.asarray(float_params)
                int_params_gpu = clp.asarray(int_params)

                # Define block and grid sizes
                block_size = (8, 8, 8)
                grid_size = ((output_section.shape[0] + block_size[0] - 1) // block_size[0],
                                (output_section.shape[1] + block_size[1] - 1) // block_size[1],
                                (output_section.shape[2] + block_size[2] - 1) // block_size[2])
                
                # Deploy kernel
                knl_at(grid_size,block_size,(filtered_gpu,m_gpu,output_section_gpu,float_params_gpu,int_params_gpu))

                # Move kernel output data back to host memory
                output_section = output_section_gpu.get()

        elif GPUBackend == 'OpenCL':

            # Move input data from host to device memory
            filtered_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filtered)
            m_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
            output_section_gpu = clp.Buffer(ctx, mf.WRITE_ONLY, output_section.nbytes)
            float_params_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=float_params)
            int_params_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int_params)

            # Deploy affine transform kernel
            knl_at(queue, output_section.shape,
                None,
                filtered_gpu,
                m_gpu,
                output_section_gpu,
                float_params_gpu,
                int_params_gpu
            ).wait()
            queue.finish()

            # Move kernel output data back to host memory
            clp.enqueue_copy(queue, output_section, output_section_gpu)
            queue.finish()

            # Release GPU memory
            filtered_gpu.release()
            m_gpu.release()
            output_section_gpu.release()
            queue.finish()

        elif GPUBackend == 'Metal':

            # Move input data from host to device memory
            filtered_gpu = ctx.buffer(filtered)
            m_gpu = ctx.buffer(m) 
            output_section_gpu = ctx.buffer(output_section)
            float_params_gpu = ctx.buffer(float_params)
            int_params_gpu = ctx.buffer(int_params)

            # Deploy affine transform kernel
            ctx.init_command_buffer()
            handle=knl_at(output_section.size,filtered_gpu,m_gpu,output_section_gpu,float_params_gpu,int_params_gpu)
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if 'arm64' not in platform.platform():
                ctx.sync_buffers((output_section_gpu,float_params_gpu))

            # Move kernel output data back to host memory
            output_section = np.frombuffer(output_section_gpu,dtype=np.float32).reshape(output_section.shape)
            
        # Record results in output array
        output[slice_start:slice_end,:,:] = output_section[:,:,:]

    if integer_output:
        output = output.astype("int16")

    return out_class(output, to_affine, from_img.header)