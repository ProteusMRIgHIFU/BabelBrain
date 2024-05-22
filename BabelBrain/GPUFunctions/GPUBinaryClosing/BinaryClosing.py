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

from scipy.ndimage import _nd_image
from numpy.core.multiarray import normalize_axis_index
from collections.abc import Iterable

if sys.platform in ['linux','win32']:
    import cupy 
    import cupyx 
    from cupyx.scipy import ndimage as cndimage
    from cupyx.scipy.ndimage import _interp_kernels

_code = '''
#ifdef _OPENCL
__kernel void binary_erosion(__global const bool * x, 
                             __global const bool * w, 
                             __global const int * int_params,
                             __global bool * y
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
kernel void binary_erosion(const device bool * x [[ buffer(0) ]], 
                           const device bool * w [[ buffer(1) ]], 
                           const device int * int_params[[ buffer(2) ]],
                           device bool * y [[ buffer(3) ]],
                           uint gid[[thread_position_in_grid]])
{
    #define _i gid
    const int ysize_0 = int_params[0];
    const int ysize_1 = int_params[1];
    const int ysize_2 = int_params[2];
    const int xind =  gid/(ysize_1*ysize_2);
    const int yind =  (gid-xind*ysize_1*ysize_2)/ysize_2;
    const int zind =  gid -xind*ysize_1*ysize_2 - yind * ysize_2;
    
#endif

    #define xsize_0 int_params[0]
    #define xsize_1 int_params[1]
    #define xsize_2 int_params[2]
    #define xstride_0 int_params[3]
    #define xstride_1 int_params[4]
    #define xstride_2 int_params[5]
    #define true_val int_params[6]
    #define false_val int_params[7]
    #define border_value int_params[8]
    #define center_is_true int_params[9]

    const int w_shape[3] = {int_params[10],int_params[11],int_params[12]};
    const int offsets[3] = {int_params[13],int_params[14],int_params[15]};

    int i = _i;
    int ind_2 = _i % ysize_2 - offsets[2]; _i /= ysize_2;
    int ind_1 = _i % ysize_1 - offsets[1]; _i /= ysize_1;
    int ind_0 = _i - offsets[0];

    #ifdef _OPENCL
    __global const unsigned char* data = (__global const unsigned char*)&x[0];
    #endif
    #ifdef _METAL
    device const unsigned char* data = (const device unsigned char*)&x[0];
    #endif
    int iws = 0;

    bool _in = (bool)x[i];
    if (center_is_true && _in == false_val) 
    {
        y[i] = (bool)_in;
        return;
    }
    y[i] = (bool)true_val;

    for (int iw_0 = 0; iw_0 < w_shape[0]; iw_0++)
    {
        int ix_0 = ind_0 + iw_0;
    
        if ((ix_0 < 0) || ix_0 >= xsize_0) 
        {
            ix_0 = -1;
        }
        ix_0 *= xstride_0;
    

        for (int iw_1 = 0; iw_1 < w_shape[1]; iw_1++)
        {
            int ix_1 = ind_1 + iw_1;
            
            if ((ix_1 < 0) || ix_1 >= xsize_1) 
            {
                ix_1 = -1;
            }
            ix_1 *= xstride_1;
    

            for (int iw_2 = 0; iw_2 < w_shape[2]; iw_2++)
            {
                int ix_2 = ind_2 + iw_2;
                
                if ((ix_2 < 0) || ix_2 >= xsize_2) {
                    ix_2 = -1;
                }
                ix_2 *= xstride_2;
                
                // inner-most loop
                bool wval = w[iws];
                
                {
                    if ((ix_0 < 0) || (ix_1 < 0) || (ix_2 < 0)) 
                    {
                        if (!border_value) 
                        {
                            y[i] = (bool)false_val;
                            return;
                        }
                    } 
                    else 
                    {
                        #ifdef _OPENCL
                        bool nn = (*(__global bool*)&data[ix_0 + ix_1 + ix_2]) ? true_val : false_val;
                        #endif
                        #ifdef _METAL
                        bool nn = (*(device bool*)&data[ix_0 + ix_1 + ix_2]) ? true_val : false_val;
                        #endif
                        if (!nn) 
                        {
                            y[i] = (bool)false_val;
                            return;
                        }
                    }
                }
                iws++;
            }
        }
    };
}
'''

Platforms=None
queue = None
prgcl = None
ctx = None
knl = None
mf = None
clp = None

def InitCUDA(DeviceName='A6000'):
    import cupy as cp
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global clp
    global knl

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
    global knl
    global knl
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

    # SCreate kernels from program function
    knl=prgcl.binary_erosion

    # Create command queue for selected device
    queue = cl.CommandQueue(ctx)

    # Allocate device memory
    mf = cl.mem_flags
    
def InitMetal(DeviceName='AMD'):
    global ctx
    global knl
    
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

    knl=prgcl.function('binary_erosion')

def erode_kernel(input, structure, output, offsets, border_value, center_is_true, invert, GPUBackend='OpenCL'):

    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    global mf
    global clp

    if invert:
        border_value = int(not border_value)
        true_val = 0
        false_val = 1
    else:
        true_val = 1
        false_val = 0

    input = input.astype(np.bool_, copy=False)
    structure = structure.astype(np.bool_, copy=False)
    output = output.astype(np.bool_, copy=False)

    int_params=np.zeros(16,np.int32)
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

    assert(np.isfortran(output)==False)
    assert(np.isfortran(input)==False)
    assert(np.isfortran(structure)==False)
    assert(np.isfortran(int_params)==False)

    if GPUBackend=='OpenCL':

        # Move input data from host to device memory
        input_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input)
        structure_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=structure)
        int_params_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int_params)
        output_gpu = clp.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

        # Deploy kernel
        knl(queue, output.shape,
            None,
            input_gpu,
            structure_gpu,
            int_params_gpu,
            output_gpu,
        )

        # Move kernel output data back to host memory
        clp.enqueue_copy(queue, output, output_gpu)

        return output
    else: # Metal

        input_gpu = ctx.buffer(input)
        structure_gpu = ctx.buffer(structure)
        int_params_gpu = ctx.buffer(int_params)
        output_gpu = ctx.buffer(output)
        
        ctx.init_command_buffer()

        handle=knl(int(np.prod(output.shape)),input_gpu,structure_gpu,int_params_gpu,output_gpu)
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((int_params_gpu,output_gpu))
        output = np.frombuffer(output_gpu,dtype=np.bool_).reshape(output.shape)

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

def _get_output_modified(output, input, shape=None, complex_output=False):
    shape = input.shape if shape is None else shape
    if output is None:
        if complex_output:
            _dtype = np.promote_types(input.dtype, np.complex64)
        else:
            _dtype = input.dtype
        output = np.empty(shape, dtype=_dtype)
    elif isinstance(output, (type, np.dtype)):
        if complex_output and np.dtype(output).kind != 'c':
            warnings.warn("promoting specified output dtype to complex")
            output = np.promote_types(output, np.complex64)
        output = np.empty(shape, dtype=output)
    elif isinstance(output, str):
        output = np.sctypeDict[output]
        if complex_output and np.dtype(output).kind != 'c':
            raise RuntimeError("output must have complex dtype")
        output = np.empty(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError("output must have complex dtype")
    return output

def _center_is_true(structure, origin):
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape, origin)])
    return bool(structure[coor])  # device synchronization

def _binary_erosion_modified(input, structure, iterations, mask, output, border_value,
                             origin, invert, brute_force=True, GPUBackend='OpenCL'):
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise TypeError('iterations parameter should be an integer')

    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')

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
            mask = np.ascontiguousarray(mask)
        masked = True
    else:
        masked = False
    origin = _fix_sequence_arg(origin, input.ndim, 'origin', int)

    if isinstance(output, np.ndarray):
        if output.dtype.kind == 'c':
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _get_output_modified(output, input)
    # temp_needed = np.shares_memory(output, input, max_work='MAY_SHARE_BOUNDS')
    temp_needed = np.may_share_memory(output, input)
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _get_output_modified(output.dtype, input)
    if structure.ndim == 0:
        # kernel doesn't handle ndim=0, so special case it here
        if float(structure):
            output[...] = np.asarray(input, dtype=bool)
        else:
            output[...] = ~np.asarray(input, dtype=bool)
        return output
    origin = tuple(origin)
    int_type = _get_inttype(input)
    offsets = _origins_to_offsets(origin, structure.shape)
    if not default_structure:
        # synchronize required to determine if all weights are non-zero
        nnz = int(np.count_nonzero(structure))
        all_weights_nonzero = nnz == structure.size
        if all_weights_nonzero:
            center_is_true = True
        else:
            center_is_true = _center_is_true(structure, origin)

    if iterations == 1:
        output = erode_kernel(input, structure, output, offsets, border_value, center_is_true, invert, GPUBackend)
    elif center_is_true and not brute_force:
        raise NotImplementedError(
            'only brute_force iteration has been implemented'
        )
    else:
        if np.may_share_memory(output, input):
            raise ValueError('output and input may not overlap in memory')
        tmp_in = np.empty_like(input, dtype=output.dtype)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        tmp_out = erode_kernel(input, structure, tmp_out, border_value, offsets, center_is_true, invert, GPUBackend)
        # TODO: kernel doesn't return the changed status, so determine it here
        changed = not (input == tmp_out).all()  # synchronize!
        ii = 1
        while ii < iterations or ((iterations < 1) and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            tmp_out = erode_kernel(tmp_in, structure, tmp_out, border_value, offsets, center_is_true, invert, GPUBackend)
            changed = not (tmp_in == tmp_out).all()
            ii += 1
            if not changed and (not ii & 1):  # synchronize!
                # can exit early if nothing changed
                # (only do this after even number of tmp_in/out swaps)
                break
        output = tmp_out
    if temp_needed:
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

def binary_dilation_modified(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0, brute_force=False, GPUBackend='OpenCL'):
    """Multidimensional binary dilation with the given structuring element."""

    origin = _fix_sequence_arg(origin, input.ndim, 'origin', int)
    structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1
    return _binary_erosion_modified(input, structure, iterations, mask, output,
                                    border_value, origin, 1, brute_force, GPUBackend)

def BinaryClose(input, structure, iterations=1, output=None, origin=0,
                   mask=None, border_value=0, brute_force=False, GPUBackend='OpenCL'):
    """
    Modified from cupy's binary_closing function to work for OpenCL and Metal 
    backends in addition to CUDA.

    see cupyx.scipy.nd_image._morphology.binary_closing for reference
    """

    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    global mf
    global clp
    
    # If CUDA selected, use existing cupy function
    if GPUBackend=='CUDA':
        with ctx:
            input_gpu = cupy.asarray(input)
            structure_gpu=cupy.asarray(structure)

            output_gpu = cndimage.binary_closing(input_gpu,structure=structure_gpu)

            output = output_gpu.get()
        return output
    else: # Use modified cupy functions
        tmp = binary_dilation_modified(input, structure, iterations, mask, None,
                                       border_value, origin, brute_force, GPUBackend)
        return binary_erosion_modified(tmp, structure, iterations, mask, output,
                                       border_value, origin, brute_force, GPUBackend)