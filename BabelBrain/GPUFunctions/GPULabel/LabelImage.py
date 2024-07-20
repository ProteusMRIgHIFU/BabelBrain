import logging
logger = logging.getLogger()
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.ndimage._morphology import generate_binary_structure
from skimage.measure import label

try:
    from GPUUtils import InitCUDA,InitOpenCL,InitMetal
except:
    from ..GPUUtils import InitCUDA,InitOpenCL,InitMetal

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

def InitLabel(DeviceName='A6000',GPUBackend='OpenCL'):
    global queue
    global prgcl
    global sel_device
    global ctx
    global knl_label_init
    global knl_label_connect
    global knl_label_count
    global knl_label_labels
    global knl_label_finalize
    global mf
    global clp
    global cndimage

    kernel_files = [
        os.path.join(resource_path(), 'label.cpp'),
    ]

    if GPUBackend == 'CUDA':
        import cupy as cp
        from cupyx.scipy import ndimage
        clp = cp
        cndimage = ndimage

        ctx,_,sel_device = InitCUDA(DeviceName=DeviceName)

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        clp = pocl

        preamble = '#define _OPENCL'
        queue,prgcl,sel_device,ctx,mf = InitOpenCL(preamble,kernel_files=kernel_files,DeviceName=DeviceName)
        
        # Create kernels from program function
        knl_label_init = prgcl.label_init
        knl_label_connect = prgcl.label_connect
        knl_label_count = prgcl.label_count
        knl_label_labels = prgcl.label_labels
        knl_label_finalize = prgcl.label_finalize

    elif GPUBackend == 'Metal':
        import metalcomputebabel as mc

        clp = mc
        preamble = '#define _METAL' 
        prgcl, sel_device, ctx = InitMetal(preamble,DeviceName=DeviceName,kernel_files=kernel_files)
       
        # Create kernel from program function
        knl_label_init = prgcl.function('label_init')
        knl_label_connect = prgcl.function('label_connect')
        knl_label_count = prgcl.function('label_count')
        knl_label_labels = prgcl.function('label_labels')
        knl_label_finalize = prgcl.function('label_finalize')


def _label_modified(x, structure, y, GPUBackend='OpenCL'):

    elems = np.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(x.ndim)]
    offset = vecs[0]
    for dm in range(1, x.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = np.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(x.ndim)] for dr in indxs]

    dirs = np.array(dirs, dtype=np.int32)
    ndirs = indxs.shape[0]
    y_shape = np.array(y.shape, dtype=np.int32)
    count = np.zeros(2, dtype=np.int32)

    x = x.astype(np.bool_, copy=False)
    y = y.astype(np.int32, copy=False)
    y_shape = y_shape.astype(np.int32, copy=False)
    dirs = dirs.astype(np.int32, copy=False)

    assert(np.isfortran(x)==False)
    assert(np.isfortran(y)==False)
    assert(np.isfortran(y_shape)==False)
    assert(np.isfortran(dirs)==False)

    if GPUBackend=='OpenCL':
        # Step 1
        x_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        y_gpu = clp.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)

        knl_label_init(queue, y.shape,
                       None,
                       x_gpu,
                       y_gpu)
        
        # Step 2
        y_shape_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_shape)
        dirs_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dirs)

        knl_label_connect(queue, y.shape,
                          None,
                          y_shape_gpu,
                          dirs_gpu,
                          np.int32(ndirs),
                          np.int32(x.ndim),
                          y_gpu)

        # Step 3
        count_gpu = clp.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=count)

        knl_label_count(queue, y.shape,
                        None,
                        y_gpu,
                        count_gpu)

        # Step 4
        clp.enqueue_copy(queue, count, count_gpu)
        maxlabel = int(count[0])
        labels = np.empty(maxlabel, dtype=np.int32)
        labels_gpu = clp.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=labels)
        
        knl_label_labels(queue, y.shape,
                         None,
                         y_gpu,
                         count_gpu,
                         labels_gpu)

        # Step 5
        clp.enqueue_copy(queue, labels, labels_gpu)
        labels = np.sort(labels)
        labels_gpu = clp.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=labels)
        
        knl_label_finalize(queue, y.shape,
                           None,
                           np.int32(maxlabel),
                           labels_gpu,
                           y_gpu)

        clp.enqueue_copy(queue, y, y_gpu)
        clp.enqueue_copy(queue, count, count_gpu)
        clp.enqueue_copy(queue, labels, labels_gpu)
    else: # Metal
        # Template created however kernel atomic functions are not as easily 
        # implemented in metalcompute therefore shelved for now
        int_params=np.zeros(3,np.int32)
        int_params[0] = ndirs
        int_params[1] = x.ndim
        # assign last element later

        # Step 1
        x_gpu = ctx.buffer(x)
        y_gpu = ctx.buffer(y)

        ctx.init_command_buffer()

        handle = knl_label_init(int(np.prod(y.shape)),
                                x_gpu,
                                y_gpu)

        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((x_gpu,y_gpu))

        # Step 2
        y_shape_gpu = ctx.buffer(y_shape)
        dirs_gpu = ctx.buffer(dirs)
        int_params_gpu = ctx.buffer(int_params)

        handle = knl_label_connect(int(np.prod(y.shape)),
                                   y_shape_gpu,
                                   dirs_gpu,
                                   int_params_gpu,
                                   y_gpu)
        
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((y_shape_gpu,y_gpu))
        
        # Step 3
        count_gpu = ctx.buffer(count)

        handle = knl_label_count(int(np.prod(y.shape)),
                                 y_gpu,
                                 count_gpu)
        
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((y_gpu,count_gpu))

        # Step 4
        count = np.frombuffer(count_gpu,dtype=np.int32).reshape(count.shape)
        maxlabel = int(count[0])
        labels = np.empty(maxlabel, dtype=np.int32)
        labels_gpu = ctx.buffer(labels)
        
        handle = knl_label_labels(int(np.prod(y.shape)),
                                  y_gpu,
                                  count_gpu,
                                  labels_gpu)

        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((y_gpu,labels_gpu))

        # Step 5
        labels = np.frombuffer(labels_gpu,dtype=np.int32).reshape(labels.shape)
        labels = np.sort(labels)
        labels_gpu = ctx.buffer(labels)
        
        int_params[5] = maxlabel
        int_params_gpu = ctx.buffer(int_params)
        
        handle = knl_label_finalize(int(np.prod(y.shape)),
                                    int_params_gpu,
                                    labels_gpu,
                                    y_gpu)
                            
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((int_params_gpu,y_gpu))

        y = np.frombuffer(y_gpu,dtype=np.int32).reshape(y.shape)
        count = np.frombuffer(count_gpu,dtype=np.int32).reshape(count.shape)
        labels = np.frombuffer(labels_gpu,dtype=np.int32).reshape(labels.shape)

    return y, maxlabel

def label_modified(input, structure=None, output=None, GPUBackend='OpenCL'):
    """Labels features in an array."""

    ''' Changed to numpy '''
    # if not isinstance(input, cupy.ndarray):
    #     raise TypeError('input must be cupy.ndarray')
    if not isinstance(input, np.ndarray):
        raise TypeError('input must be np.ndarray')
    
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')
    if structure is None:
        ''' Call scipy's version of generate_binary_structure instead of cupy's'''
        # structure = _generate_binary_structure(input.ndim, 1)
        structure = generate_binary_structure(input.ndim, 1)
    ''' Removed since it won't be a cupy array '''
    # elif isinstance(structure, cupy.ndarray):
    #     structure = cupy.asnumpy(structure)

    structure = np.array(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for i in structure.shape:
        if i != 3:
            raise ValueError('structure dimensions must be equal to 3')

    ''' Changed to numpy '''
    # if isinstance(output, cupy.ndarray):
    if isinstance(output, np.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            ''' Changed to numpy '''
            # output = cupy.empty(input.shape, numpy.int32)
            output = np.empty(input.shape, np.int32)
        else:
            ''' Changed to numpy '''
            # output = cupy.empty(input.shape, output)
            output = np.empty(input.shape, output)

    if input.size == 0:
        # empty
        maxlabel = 0
    elif input.ndim == 0:
        # 0-dim array
        maxlabel = 0 if input.item() == 0 else 1
        output.fill(maxlabel)
    else:
        if output.dtype != np.int32:
            ''' Changed to numpy '''
            # y = cupy.empty(input.shape, numpy.int32)
            y = np.empty(input.shape, np.int32)
        else:
            y = output

        ''' Replace with custom kernel call'''
        # maxlabel = _label(input, structure, y)
        output, maxlabel = _label_modified(input, structure, y,GPUBackend=GPUBackend)

        ''' Removed since it ensure it dtype in _label_modified'''
        # if output.dtype != np.int32:
        #     _core.elementwise_copy(y, output)

    if caller_provided_output:
        return maxlabel
    else:
        return output, maxlabel


def LabelImage(image, background=None, return_num=False, connectivity=None, GPUBackend='OpenCL'): # return_num=False, 
    """
    Modified from Skimage's label function to work using GPU (Cupy, OpenCL, and Metal)

    see Skimage.measure._label.label for reference
    """
    logger.info("\nStarting Label")

    ''' Changed since we're only doing boolean case '''
    # if label_image.dtype == bool:
    #     return _label_bool(label_image, background=background,
    #                        return_num=return_num, connectivity=connectivity)
    # else:
    #     return clabel(label_image, background, return_num, connectivity)
    if image.dtype != bool:
        msg = f"Image datatype must be boolean. For other datatypes, use Skimage.measure.label function"
        raise RuntimeError(msg)

    ''' Changed import call '''
    # from ..morphology._util import _resolve_neighborhood
    from skimage.morphology._util import _resolve_neighborhood

    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood(None, connectivity, image.ndim)

    ''' We do custom kernel call instead '''
    # result = ndimage.label(image, structure=footprint)

    # We try to run label step through GPU but switch to CPU method if
    # GPU memory is not sufficient for array size.
    if GPUBackend=='CUDA':
        # If CUDA selected, use existing cupy function
        try:
            with ctx:
                image_gpu = clp.asarray(image)
                # footprint_gpu=cupy.asarray(footprint)

                result_gpu = cndimage.label(image_gpu,structure=footprint)
                result = result_gpu[0].get()
            return result
        except clp.cuda.memory.OutOfMemoryError as e:
            print(f"{e}\nNot enough memory to complete label step in one go. Switching to CPU method")
            result = label(image)
            return result
    elif GPUBackend=='OpenCL':
        try:
            '''Modified from cupyz.scipy.ndimage._measurements.label function '''
            result =  label_modified(image,structure=footprint, GPUBackend=GPUBackend)

            if return_num:
                return result
            else:
                return result[0]
        except clp.MemoryError as e:
            print(f"{e}\nNot enough memory to complete label step in one go. Switching to CPU method")
            result = label(image)
            return result
    else: # Metal
        raise ValueError('Metal version has not been implemented')