import numpy as np
import numpy.linalg as npl
import sys
import os
import warnings
import functools
import operator
import platform

import nibabel
from nibabel import processing
from nibabel.affines import AffineError, append_diag, to_matvec, from_matvec
from nibabel.imageclasses import spatial_axes_first
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage

from numpy.core.multiarray import normalize_axis_index
from collections.abc import Iterable

if sys.platform in ['linux','win32']:
    import cupy 
    import cupyx 
    from cupyx.scipy import ndimage as cndimage
    from cupyx.scipy.ndimage import _interp_kernels

_code = '''
#ifdef _OPENCL
__kernel void label_init(__global const bool * x, 
                         __global int * y
                         ) 
{
    const int ysize_0 = get_global_size(0);
    const int ysize_1 = get_global_size(1);
    const int ysize_2 = get_global_size(2);
    const int xind =  get_global_id(0);
    const int yind =  get_global_id(1);
    const int zind =  get_global_id(2);

    int _i = xind*ysize_1*ysize_2 + yind*ysize_2 + zind;
#endif
#ifdef _METAL
#include <metal_stdlib>
using namespace metal;

kernel void label_init(const device bool * x [[ buffer(0) ]],
                       device int * y [[ buffer(2) ]],
                       uint gid[[thread_position_in_grid]]) 
{
    #define _i gid
#endif

    if (x[_i] == 0)
    { 
        y[_i] = -1; 
    } 
    else 
    { 
        y[_i] = _i; 
    }
    
}


#ifdef _OPENCL
__kernel void label_connect(__global const int * shape,
                            __global const int * dirs,
                            const int ndirs,
                            const int ndim, 
                            __global int * y
                            )
{

    const int ysize_0 = get_global_size(0);
    const int ysize_1 = get_global_size(1);
    const int ysize_2 = get_global_size(2);
    const int xind =  get_global_id(0);
    const int yind =  get_global_id(1);
    const int zind =  get_global_id(2);

    int _i = xind*ysize_1*ysize_2 + yind*ysize_2 + zind;

#endif
#ifdef _METAL
kernel void label_connect(const device int * shape [[ buffer(0) ]],
                          const device int * dirs [[ buffer(1) ]],
                          const device int * int_params [[ buffer(2) ]],
                          device int * y [[ buffer(3) ]],
                          uint gid[[thread_position_in_grid]])
{
    #define ndirs int_params[0]
    #define ndim int_params[1]
    #define _i gid
#endif
    
    if (y[_i] < 0) return;
    for (int dr = 0; dr < ndirs; dr++) 
    {
        int j = _i;
        int rest = j;
        int stride = 1;
        int k = 0;
        for (int dm = ndim-1; dm >= 0; dm--) 
        {
            int pos = rest % shape[dm] + dirs[dm + dr * ndim];
            if (pos < 0 || pos >= shape[dm]) 
            {
                k = -1;
                break;
            }
            k += pos * stride;
            rest /= shape[dm];
            stride *= shape[dm];
        }
        if (k < 0) continue;
        if (y[k] < 0) continue;
        while (1) 
        {
            while (j != y[j]) 
            {  
                j = y[j]; 
            }
            while (k != y[k]) 
            { 
                k = y[k]; 
            }
            if (j == k) break;
            if (j < k) 
            {
                #ifdef _OPENCL
                int old = atomic_cmpxchg(&y[k], k, j);
                #endif
                #ifdef _METAL
                int old = atomic_cmpxchg(&y[k], k, j);
                #endif
                if (old == k) break;
                k = old;
            }
            else 
            {
                #ifdef _OPENCL
                int old = atomic_cmpxchg( &y[j], j, k );
                #endif
                #ifdef _METALL
                int old = atomic_cmpxchg( &y[j], j, k );
                #endif
                if (old == j) break;
                j = old;
            }
        }
    }
      
}

#ifdef _OPENCL
__kernel void label_count(__global int * y, 
                          __global int * count) 
{
    const int ysize_0 = get_global_size(0);
    const int ysize_1 = get_global_size(1);
    const int ysize_2 = get_global_size(2);
    const int xind =  get_global_id(0);
    const int yind =  get_global_id(1);
    const int zind =  get_global_id(2);

    int _i = xind*ysize_1*ysize_2 + yind*ysize_2 + zind;
#endif
#ifdef _METAL
kernel void label_count(device int * y [[ buffer(0) ]], 
                        device int * count [[ buffer(1) ]],
                        uint gid[[thread_position_in_grid]]) 
{
    #define _i gid
#endif

    if (y[_i] < 0)
    {
        return;
    }
    int j = _i;
    while (j != y[j]) 
    { 
        j = y[j]; 
    }
    if (j != _i)
    {
        y[_i] = j;
    }
    else
    {
        #ifdef _OPENCL
        atomic_add(&count[0], 1);
        #endif
        #ifdef _METAL
        atomic_add(&count[0], 1);
        #endif
    }
}

#ifdef _OPENCL
__kernel void label_labels(__global int * y,
                           __global int * count, 
                           __global int * labels) 
{
    const int ysize_0 = get_global_size(0);
    const int ysize_1 = get_global_size(1);
    const int ysize_2 = get_global_size(2);
    const int xind =  get_global_id(0);
    const int yind =  get_global_id(1);
    const int zind =  get_global_id(2);

    int _i = xind*ysize_1*ysize_2 + yind*ysize_2 + zind;
#endif
#ifdef _METAL
kernel void label_labels(device int * y [[ buffer(0) ]],
                         device int * count [[ buffer(1) ]], 
                         device int * labels [[ buffer (2) ]],
                         uint gid[[thread_position_in_grid]]) 
{
    #define _i gid
#endif

    if (y[_i] != _i)
    {
        return;
    }
    
    #ifdef _OPENCL
    int j = atomic_add(&count[1], 1);
    #endif
    #ifdef _METAL
    int j = atomic_add(&count[1], 1);
    #endif

    labels[j] = _i;
}

#ifdef _OPENCL

__kernel void label_finalize(const int maxlabel,
                            __global int * labels, 
                            __global int * y) 
{

    const int ysize_0 = get_global_size(0);
    const int ysize_1 = get_global_size(1);
    const int ysize_2 = get_global_size(2);
    const int xind =  get_global_id(0);
    const int yind =  get_global_id(1);
    const int zind =  get_global_id(2);

    int _i = xind*ysize_1*ysize_2 + yind*ysize_2 + zind;

#endif
#ifdef _METAL
kernel void label_finalize(const device int * int_params [[ buffer(0)]] ,
                           device int * labels [[ buffer(1) ]], 
                           device int * y [[ buffer(2) ]],
                           uint gid[[thread_position_in_grid]]) 
{
    #define maxlabel int_params[2]
    #define _i gid
#endif

    if (y[_i] < 0) 
    {
        y[_i] = 0;
        return;
    }
    int yi = y[_i];
    int j_min = 0;
    int j_max = maxlabel - 1;
    int j = (j_min + j_max) / 2;
    while (j_min < j_max) 
    {
        if (yi == labels[j]) break;
        if (yi < labels[j]) 
            j_max = j - 1;
        else 
            j_min = j + 1;
        j = (j_min + j_max) / 2;
    }
    y[_i] = j + 1;
}
'''

Platforms=None
queue = None
prgcl = None
ctx = None
knl_label_init = None
knl_label_connect = None
knl_label_count = None
knl_label_labels = None
knl_label_finalize = None
mf = None
clp = None

def InitCUDA(DeviceName='A6000'):
    import cupy as cp
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global clp

    devCount = cp.cuda.runtime.getDeviceCount()
    print("Number of CUDA devices found:", devCount)
    if devCount == 0:
        raise SystemError("There are no CUDA devices.")
        
    selDevice = None

    for deviceID in range(0, devCount):
        d=cp.cuda.runtime.getDeviceProperties(deviceID)
        if DeviceName in d['name'].decode('UTF-8'):
            selDevice=cp.cuda.Device(deviceID)
            break

    if selDevice is None:
        raise SystemError("There are no devices supporting CUDA or that matches selected device.")
      
    ctx=selDevice
    clp=cp

def InitOpenCL(DeviceName='AMD'):
    import pyopencl as cl
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl_label_init
    global knl_label_connect
    global knl_label_count
    global knl_label_labels
    global knl_label_finalize
    global mf
    global clp
    clp=cl
    
    # Obtain list of openCL platforms
    Platforms=cl.get_platforms()
    if len(Platforms)==0:
        raise SystemError("No OpenCL platforms")
    
    # btain list of available devices and select one 
    SelDevice=None
    for device in Platforms[0].get_devices():
        print(device.name)
        if DeviceName in device.name:
            SelDevice=device
    if SelDevice is None:
        raise SystemError("No OpenCL device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', SelDevice.name)

    # Create context for selected device
    ctx = cl.Context([SelDevice])
    
    # Build program from source code
    prgcl = cl.Program(ctx,"#define _OPENCL\n"+_code).build()

    # Create kernels from program functions
    knl_label_init = prgcl.label_init
    knl_label_connect = prgcl.label_connect
    knl_label_count = prgcl.label_count
    knl_label_labels = prgcl.label_labels
    knl_label_finalize = prgcl.label_finalize

    # Create command queue for selected device
    queue = cl.CommandQueue(ctx)

    # Allocate device memory
    mf = cl.mem_flags
    
def InitMetal(DeviceName='AMD'):
    global ctx
    global knl_label_init
    global knl_label_connect
    global knl_label_count
    global knl_label_labels
    global knl_label_finalize
    
    import metalcomputebabel as mc

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

    prgcl = ctx.kernel('#define _METAL\n'+_code)

    knl_label_init = prgcl.function('label_init')
    knl_label_connect = prgcl.function('label_connect')
    knl_label_count = prgcl.function('label_count')
    knl_label_labels = prgcl.function('label_labels')
    knl_label_finalize = prgcl.function('label_finalize')

def _label_modified(x, structure, y, GPUBackend='OpenCL'):

    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl_label_init
    global knl_label_connect
    global knl_label_count
    global knl_label_labels
    global knl_label_finalize
    global mf
    global clp

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

def _generate_binary_structure(rank, connectivity):
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return np.array(True, dtype=bool)
    output = np.fabs(np.indices([3] * rank) - 1)
    output = np.add.reduce(output, 0)
    return output <= connectivity

def label_modified(input, structure=None, output=None, GPUBackend='OpenCL'):
    """Labels features in an array."""
    if not isinstance(input, np.ndarray):
        raise TypeError('input must be np.ndarray')
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = _generate_binary_structure(input.ndim, 1)

    structure = np.array(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for i in structure.shape:
        if i != 3:
            raise ValueError('structure dimensions must be equal to 3')

    if isinstance(output, np.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = np.empty(input.shape, np.int32)
        else:
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
            y = np.empty(input.shape, np.int32)
        else:
            y = output
        output, maxlabel = _label_modified(input, structure, y,GPUBackend=GPUBackend)

    if caller_provided_output:
        return maxlabel
    else:
        return output, maxlabel

def _resolve_neighborhood_modified(footprint, connectivity, ndim,
                          enforce_adjacency=True):
    """Validate or create a footprint (structuring element)."""
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        footprint = ndimage.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        footprint = np.asarray(footprint, dtype=bool)
        # Must specify neighbors for all dimensions
        if footprint.ndim != ndim:
            raise ValueError(
                "number of dimensions in image and footprint do not"
                "match"
            )
        # Must only specify direct neighbors
        if enforce_adjacency and any(s != 3 for s in footprint.shape):
            raise ValueError("dimension size in footprint is not 3")
        elif any((s % 2 != 1) for s in footprint.shape):
            raise ValueError("footprint size must be odd along all dimensions")

    return footprint

def LabelImage(image, background=None, return_num=False, connectivity=None, GPUBackend='OpenCL'): # return_num=False, 
    """
    Modified from Skimage's label function to work using GPU (Cupy, OpenCL, and Metal)

    see Skimage.measure.label for reference
    """

    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    global mf
    global clp
    
    if image.dtype != bool:
        msg = f"Image datatype must be boolean. For other datatypes, use Skimage.measure.label function"
        raise RuntimeError(msg)

    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood_modified(None, connectivity, image.ndim)

    # If CUDA selected, use existing cupy function
    if GPUBackend=='CUDA':
        with ctx:
            image_gpu = cupy.asarray(image)
            # footprint_gpu=cupy.asarray(footprint)

            result_gpu = cndimage.label(image_gpu,structure=footprint)
            result = result_gpu[0].get()
        return result
    else: # Use modified cupy functions
        result =  label_modified(image,structure=footprint, GPUBackend=GPUBackend)

        if return_num:
            return result
        else:
            return result[0]