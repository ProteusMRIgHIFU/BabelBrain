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

# _prod = cupy._core.internal.prod
_prod_numpy =np.prod

_transform_code = '''
#ifdef _OPENCL
__kernel void affine_transform(__global const float * x, 
                      __global const float * mat,
                      __global float * y,
                      const float cval,
                      const int order,
                      const int in_dims_0,
                      const int in_dims_1,
                      const int in_dims_2) 

{

    unsigned int out_dims_1 = get_global_size(1);
    unsigned int out_dims_2 = get_global_size(2);

    const int xind =  get_global_id(0);
    const int yind =  get_global_id(1);
    const int zind =  get_global_id(2);

    const int _i = xind*out_dims_1*out_dims_2 + yind*out_dims_2 + zind;
#endif

#ifdef _METAL
#include <metal_stdlib>
using namespace metal;

// To be defined when compiling
// in_dims_0, in_dims_1, in_dims_2
// out_dims_0, out_dims_1, out_dims_2  
// cval 
// order  

kernel void affine_transform(const device float * x [[ buffer(0) ]], 
                             const device float * mat [[ buffer(1) ]],
                             device float * y [[ buffer(2) ]],
                             constant int * int_params [[ buffer(3) ]],
                             constant float * float_params [[ buffer(4) ]],
                             uint gid[[thread_position_in_grid]]) 
{
    #define in_dims_0 int_params[0]
    #define in_dims_1 int_params[1]
    #define in_dims_2 int_params[2]
    #define out_dims_0 int_params[3]
    #define out_dims_1 int_params[4]
    #define out_dims_2 int_params[5]
    #define order int_params[6]
    #define cval float_params[0]

    const int xind =  gid/(out_dims_1*out_dims_2);
    const int yind =  (gid-xind*out_dims_1*out_dims_2)/out_dims_2;
    const int zind =  gid -xind*out_dims_1*out_dims_2 - yind * out_dims_2;
    #define _i gid
#endif

    float out = 0.0;
    const int xsize_0 = in_dims_0;
    const int xsize_1 = in_dims_1;
    const int xsize_2 = in_dims_2;
    const unsigned int sx_2 = 1;
    const unsigned int sx_1 = sx_2 * xsize_2;
    const unsigned int sx_0 = sx_1 * xsize_1;
    
    int in_coord[3] = {xind,yind,zind};

    float c_0 = (float)0.0;
    c_0 += mat[0] * (float)in_coord[0];
    c_0 += mat[1] * (float)in_coord[1];
    c_0 += mat[2] * (float)in_coord[2];
    c_0 += mat[3];

    float c_1 = (float)0.0;
    c_1 += mat[4] * (float)in_coord[0];
    c_1 += mat[5] * (float)in_coord[1];
    c_1 += mat[6] * (float)in_coord[2];
    c_1 += mat[7];

    float c_2 = (float)0.0;
    c_2 += mat[8] * (float)in_coord[0];
    c_2 += mat[9] * (float)in_coord[1];
    c_2 += mat[10] * (float)in_coord[2];
    c_2 += mat[11];

    if ((c_0 < 0) || (c_0 > xsize_0 - 1) || (c_1 < 0) || (c_1 > xsize_1 - 1) || (c_2 < 0) || (c_2 > xsize_2 - 1))
    {
       out = (float)cval;
    }
    else
    {
        if (order == 0)
        {
            int cf_0 = (int)floor((float)c_0 + 0.5);
            int ic_0 = cf_0 * sx_0;
            int cf_1 = (int)floor((float)c_1 + 0.5);
            int ic_1 = cf_1 * sx_1;
            int cf_2 = (int)floor((float)c_2 + 0.5);
            int ic_2 = cf_2 * sx_2;
            out = (float)x[ic_0 + ic_1 + ic_2];
        }
        else
        {
            float wx, wy;
            int start;
            float weights_0[4];
            wx = c_0 - floor(3 & 1 ? c_0 : c_0 + 0.5);
            wy = 1.0 - wx;
            weights_0[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
            weights_0[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
            weights_0[0] = wy * wy * wy / 6.0;
            weights_0[3] = 1.0 - weights_0[0] - weights_0[1] - weights_0[2];
            start = (int)floor((float)c_0) - 1;
            int ci_0[4];
            ci_0[0] = start + 0;
            if (xsize_0 == 1) 
            {
                ci_0[0] = 0;
            }
            else 
            {
                if (ci_0[0] < 0)
                {
                    ci_0[0] = -ci_0[0];
                }
                ci_0[0] = 1 + (ci_0[0] - 1) % ((xsize_0 - 1) * 2);
                ci_0[0] = min(ci_0[0], 2 * xsize_0 - 2 - ci_0[0]);
            }
            ci_0[1] = start + 1;
            if (xsize_0 == 1) 
            {
                ci_0[1] = 0;
            } 
            else 
            {
                if (ci_0[1] < 0) 
                {
                    ci_0[1] = -ci_0[1];
                }
                ci_0[1] = 1 + (ci_0[1] - 1) % ((xsize_0 - 1) * 2);
                ci_0[1] = min(ci_0[1], 2 * xsize_0 - 2 - ci_0[1]);
            }
            ci_0[2] = start + 2;
            if (xsize_0 == 1) 
            {
                ci_0[2] = 0;
            } 
            else 
            {
                if (ci_0[2] < 0) 
                {
                    ci_0[2] = -ci_0[2];
                }
                ci_0[2] = 1 + (ci_0[2] - 1) % ((xsize_0 - 1) * 2);
                ci_0[2] = min(ci_0[2], 2 * xsize_0 - 2 - ci_0[2]);
            }
            ci_0[3] = start + 3;
            if (xsize_0 == 1) 
            {
                ci_0[3] = 0;
            } 
            else 
            {
                if (ci_0[3] < 0) 
                {
                    ci_0[3] = -ci_0[3];
                }
                ci_0[3] = 1 + (ci_0[3] - 1) % ((xsize_0 - 1) * 2);
                ci_0[3] = min(ci_0[3], 2 * xsize_0 - 2 - ci_0[3]);
            }
            float w_0;
            int ic_0;
            for (int k_0 = 0; k_0 <= 3; k_0++)
            {
                w_0 = weights_0[k_0];
                ic_0 = ci_0[k_0] * sx_0;
                float weights_1[4];
                wx = c_1 - floor(3 & 1 ? c_1 : c_1 + 0.5);
                wy = 1.0 - wx;
                weights_1[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
                weights_1[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
                weights_1[0] = wy * wy * wy / 6.0;
                weights_1[3] = 1.0 - weights_1[0] - weights_1[1] - weights_1[2];
                start = (int)floor((float)c_1) - 1;
                int ci_1[4];
                ci_1[0] = start + 0;
                if (xsize_1 == 1) 
                {
                    ci_1[0] = 0;
                } 
                else 
                {
                    if (ci_1[0] < 0) 
                    {
                        ci_1[0] = -ci_1[0];
                    }
                    ci_1[0] = 1 + (ci_1[0] - 1) % ((xsize_1 - 1) * 2);
                    ci_1[0] = min(ci_1[0], 2 * xsize_1 - 2 - ci_1[0]);
                }
                ci_1[1] = start + 1;
                if (xsize_1 == 1) 
                {
                    ci_1[1] = 0;
                } 
                else 
                {
                    if (ci_1[1] < 0)
                    {
                        ci_1[1] = -ci_1[1];
                    }
                    ci_1[1] = 1 + (ci_1[1] - 1) % ((xsize_1 - 1) * 2);
                    ci_1[1] = min(ci_1[1], 2 * xsize_1 - 2 - ci_1[1]);
                }
                ci_1[2] = start + 2;
                if (xsize_1 == 1) 
                {
                    ci_1[2] = 0;
                } 
                else 
                {
                    if (ci_1[2] < 0) 
                    {
                        ci_1[2] = -ci_1[2];
                    }
                    ci_1[2] = 1 + (ci_1[2] - 1) % ((xsize_1 - 1) * 2);
                    ci_1[2] = min(ci_1[2], 2 * xsize_1 - 2 - ci_1[2]);
                }
                ci_1[3] = start + 3;
                if (xsize_1 == 1) 
                {
                    ci_1[3] = 0;
                } 
                else 
                {
                    if (ci_1[3] < 0) 
                    {
                        ci_1[3] = -ci_1[3];
                    }
                    ci_1[3] = 1 + (ci_1[3] - 1) % ((xsize_1 - 1) * 2);
                    ci_1[3] = min(ci_1[3], 2 * xsize_1 - 2 - ci_1[3]);
                }
                float w_1;
                int ic_1;
                for (int k_1 = 0; k_1 <= 3; k_1++)
                {
                    w_1 = weights_1[k_1];
                    ic_1 = ci_1[k_1] * sx_1;
                    float weights_2[4];
                    wx = c_2 - floor(3 & 1 ? c_2 : c_2 + 0.5);
                    wy = 1.0 - wx;
                    weights_2[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
                    weights_2[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
                    weights_2[0] = wy * wy * wy / 6.0;
                    weights_2[3] = 1.0 - weights_2[0] - weights_2[1] - weights_2[2];
                    start = (int)floor((float)c_2) - 1;
                    int ci_2[4];
                    ci_2[0] = start + 0;
                    if (xsize_2 == 1) 
                    {
                        ci_2[0] = 0;
                    } 
                    else 
                    {
                        if (ci_2[0] < 0) 
                        {
                            ci_2[0] = -ci_2[0];
                        }
                        ci_2[0] = 1 + (ci_2[0] - 1) % ((xsize_2 - 1) * 2);
                        ci_2[0] = min(ci_2[0], 2 * xsize_2 - 2 - ci_2[0]);
                    }
                    ci_2[1] = start + 1;
                    if (xsize_2 == 1) 
                    {
                        ci_2[1] = 0;
                    } 
                    else 
                    {
                        if (ci_2[1] < 0) 
                        {
                            ci_2[1] = -ci_2[1];
                        }
                        ci_2[1] = 1 + (ci_2[1] - 1) % ((xsize_2 - 1) * 2);
                        ci_2[1] = min(ci_2[1], 2 * xsize_2 - 2 - ci_2[1]);
                    }
                    ci_2[2] = start + 2;
                    if (xsize_2 == 1) 
                    {
                        ci_2[2] = 0;
                    } 
                    else 
                    {
                        if (ci_2[2] < 0) 
                        {
                            ci_2[2] = -ci_2[2];
                        }
                        ci_2[2] = 1 + (ci_2[2] - 1) % ((xsize_2 - 1) * 2);
                        ci_2[2] = min(ci_2[2], 2 * xsize_2 - 2 - ci_2[2]);
                    }
                    ci_2[3] = start + 3;
                    if (xsize_2 == 1) 
                    {
                        ci_2[3] = 0;
                    } 
                    else 
                    {
                        if (ci_2[3] < 0) 
                        {
                            ci_2[3] = -ci_2[3];
                        }
                        ci_2[3] = 1 + (ci_2[3] - 1) % ((xsize_2 - 1) * 2);
                        ci_2[3] = min(ci_2[3], 2 * xsize_2 - 2 - ci_2[3]);
                    }
                    float w_2;
                    int ic_2;
                    for (int k_2 = 0; k_2 <= 3; k_2++)
                    {
                        w_2 = weights_2[k_2];
                        ic_2 = ci_2[k_2] * sx_2;
                        if ((ic_0 < 0) || (ic_1 < 0) || (ic_2 < 0)) 
                        {
                            out += (float)cval * (float)(w_0 * w_1 * w_2);
                        } 
                        else 
                        {
                            float val = (float)x[ic_0 + ic_1 + ic_2];
                            out += val * (float)(w_0 * w_1 * w_2);
                        }
                    }
                }
            }
        }
    }
    if(order == 0)
    {
        y[_i] = (short)rint((float)out);
    }
    else
    {
        y[_i] = (float)out;
    }
}
'''

_spline_code='''
#ifdef _OPENCL
__kernel void spline_filter_3d(__global float* y, 
                               __global const int* info,
                               const int axis,
                               const int n_bound) 
{

    const int _i = get_global_id(0);

    const int n_signals = info[0];
    const int n_samples = info[1];
    const __global int * shape = info+2;
#endif

#ifdef _METAL
#include <metal_stdlib>
using namespace metal;

kernel void spline_filter_3d(device float * y [[ buffer(0) ]], 
                             const device int * info [[ buffer(1) ]],
                             constant int * int_params [[ buffer(2) ]],
                             uint gid[[thread_position_in_grid]])
{
    #define axis int_params[0]
    #define n_bound int_params[1]

    #define _i gid

    const uint n_signals = info[0];
    const uint n_samples = info[1];
    const device int * shape = info+2;
#endif
    
    int elem_stride = 1;
    for (int a = 3 - 1; a > axis; --a) // 3 is ndim
    { 
        elem_stride *= shape[a]; 
    }
    
    if (_i < n_signals)
    {
        int i_tmp = _i;
        int index = 0, stride = 1;
        for (int a = 3 - 1; a > 0; --a) 
        {
            if (a != axis) 
            {
                index += (i_tmp % shape[a]) * stride;
                i_tmp /= shape[a];
            }
            stride *= shape[a];
        }
        
        int row = index + stride * i_tmp;

        int i, n = n_samples;

        float z, z_i;
        float z_n_1;

        //select the current pole
        z = -0.2679491924311227;

        //causal init for mode=mirror
        z_i = z;
        if (n < 50)
            z_n_1 = pow(z, (float)(n - 1));
        else
            z_n_1 = 0.0;

        y[row] = y[row] + z_n_1 * y[row + (n - 1) * elem_stride];
        for (i = 1; i < min(n - 1, n_bound); ++i)
        {
           y[row] += z_i * (y[row + i * elem_stride] +
                       z_n_1 * y[row + (n - 1 - i) * elem_stride]);
           z_i *= z;
        }
        y[row] /= 1 - z_n_1 * z_n_1;

        // apply the causal filter for the current pole
        for (i = 1; i < n; ++i) {
           y[row + i * elem_stride] += z * y[row + (i - 1) * elem_stride];
        }

        // anti-causal init for mode=mirror
        y[row + (n - 1) * elem_stride] = (
           z * y[row + (n - 2) * elem_stride] +
           y[row + (n - 1) * elem_stride]) * z / (z * z - 1);

        // apply the anti-causal filter for the current pole
        for (i = n - 2; i >= 0; --i) {
           y[row + i * elem_stride] = z * (y[row + (i + 1) * elem_stride] -
                                       y[row + i * elem_stride]);
        }
    }
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
    global knl_at
    global knl_sf
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
    prgcl = cl.Program(ctx,"#define _OPENCL\n"+_transform_code+_spline_code).build()


    # SCreate kernels from program function
    knl_at=prgcl.affine_transform
    knl_sf=prgcl.spline_filter_3d

    # Create command queue for selected device
    queue = cl.CommandQueue(ctx)

    # Allocate device memory
    mf = cl.mem_flags
    
def InitMetal(DeviceName='AMD'):
    global ctx
    global knl_at
    global knl_sf
    
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

    prgcl = ctx.kernel('#define _METAL\n'+_transform_code+_spline_code)

    knl_at=prgcl.function('affine_transform')
    knl_sf=prgcl.function('spline_filter_3d')


def _check_parameter_modified(func_name, order, mode):
    if order < 0 or 5 < order:
        raise ValueError('spline order is not supported')

    if mode not in ('constant'):
        raise ValueError('boundary mode ({}) is not supported'.format(mode))

def _fix_sequence_arg_modified(arg, ndim, name, conv=lambda x: x):
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

def _prepad_for_spline_filter_modified(input, mode, cval):
    npad = 0
    padded = input
    return padded, npad

def _get_spline_output_modified(input, output):
    complex_data = input.dtype.kind == 'c'
    if complex_data:
        min_float_dtype = np.complex64
    else:
        min_float_dtype = np.float32
    if isinstance(output, np.ndarray):
        if complex_data and output.dtype.kind != 'c':
            raise ValueError(
                'output must have complex dtype for complex inputs'
            )
        float_dtype = np.promote_types(output.dtype, min_float_dtype)
        output_dtype = output.dtype
    else:
        if output is None:
            output = output_dtype = input.dtype
        else:
            output_dtype = np.dtype(output)
        float_dtype = np.promote_types(output, min_float_dtype)

    if (isinstance(output, np.ndarray)
            and output.dtype == float_dtype == output_dtype
            and output.flags.c_contiguous):
        if output is not input:
            np.copyto(output, input)
        temp = output
    else:
        temp = input.astype(float_dtype, copy=False)
        temp = np.ascontiguousarray(temp)
        if np.shares_memory(temp, input):
            temp = temp.copy()
    return temp, float_dtype, output_dtype

def _get_inttype_modified(input):
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    return 'int' if nbytes < (1 << 31) else 'ptrdiff_t'

def get_poles_modified(order):
    if order == 2:
        # sqrt(8.0) - 3.0
        return (-0.171572875253809902396622551580603843,)
    elif order == 3:
        # sqrt(3.0) - 2.0
        return (-0.267949192431122706472553658494127633,)
    elif order == 4:
        # sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0
        # sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0
        return (-0.361341225900220177092212841325675255,
                -0.013725429297339121360331226939128204)
    elif order == 5:
        # sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5
        # sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5
        return (-0.430575347099973791851434783493520110,
                -0.043096288203264653822712376822550182)
    else:
        raise ValueError('only order 2-5 supported')
    
def get_gain_modified(poles):
    return functools.reduce(operator.mul,
                            [(1.0 - z) * (1.0 - 1.0 / z) for z in poles])

def spline_filter1d_modified(input, order=3, axis=-1, output=np.float64,
                    mode='mirror', GPUBackend='OpenCL'):
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    x = np.asarray(input)
    ndim = x.ndim
    axis = normalize_axis_index(axis, ndim)

    # order 0, 1 don't require reshaping as no kernel will be called
    # scalar or size 1 arrays also don't need to be filtered
    run_kernel = not (order < 2 or x.ndim == 0 or x.shape[axis] == 1)
    if not run_kernel:
        output = _get_output_modified(output, input)
        np.copyto(output,x)
        return output

    temp, data_dtype, output_dtype = _get_spline_output_modified(x, output)
    data_type = np.core._dtype._kind_name(temp.dtype)
    pole_type = np.core._dtype._kind_name(temp.real.dtype)

    index_type = _get_inttype_modified(input)
    index_dtype = np.int32 if index_type == 'int' else np.int64

    n_samples = x.shape[axis]
    n_signals = x.size // n_samples
    info = np.array((n_signals, n_samples) + x.shape, dtype=index_dtype)

    # empirical choice of block size that seemed to work well
    block_size = max(2 ** np.math.ceil(np.log2(n_samples / 32)), 8)

    # Due to recursive nature, a given line of data must be processed by a
    # single thread. n_signals lines will be processed in total.
    block = (block_size,)
    grid = ((n_signals + block[0] - 1) // block[0],)

    # apply prefilter gain
    poles = get_poles_modified(order=order)
    temp *= get_gain_modified(poles)

    poles = get_poles_modified(order)

    # determine number of samples for the boundary approximation
    # (SciPy uses n_boundary = n_samples but this is excessive)
    largest_pole = max([abs(p) for p in poles])
    # tol < 1e-7 fails test cases comparing to SciPy at atol = rtol = 1e-5
    tol = 1e-10 if pole_type == 'float' else 1e-18
    n_boundary = np.math.ceil(np.math.log(tol, largest_pole))

    temp = temp.astype('float32')
    info = info.astype('int32')

    assert(np.isfortran(info)==False)
    assert(np.isfortran(temp)==False)

    if GPUBackend == 'OpenCL':
        info_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=info)
        temp_gpu = clp.Buffer(ctx, mf.READ_WRITE | clp.mem_flags.USE_HOST_PTR, hostbuf=temp)

        knl_sf(queue,grid,
            block,
            temp_gpu,
            info_gpu,
            np.int32(axis),
            np.int32(n_boundary),
            g_times_l=True)
        
        clp.enqueue_copy(queue, temp, temp_gpu)
    else: # Metal
        int_params=np.zeros(2,np.int32)
        int_params[0] = axis
        int_params[1] = n_boundary

        info_gpu = ctx.buffer(info)
        temp_gpu = ctx.buffer(temp)
        int_params_gpu = ctx.buffer(int_params)

        ctx.init_command_buffer()
        handle=knl_sf(int(n_signals),temp_gpu,info_gpu, int_params_gpu)
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((temp_gpu,info_gpu))
        temp = np.frombuffer(temp_gpu,dtype=np.float32).reshape(temp.shape)

    if isinstance(output, np.ndarray) and temp is not output:
        # copy kernel output into the user-provided output array
        np.copyto(output, temp)
        return output
    return temp.astype(output_dtype, copy=False)

def spline_filter_modified(input, order=3, output=np.float64, mode='mirror',GPUBackend='OpenCL'):
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')

    x = np.asarray(input)
    temp, data_dtype, output_dtype = _get_spline_output_modified(x, output)
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(x.ndim):
            spline_filter1d_modified(x, order, axis, output=temp, mode=mode, GPUBackend=GPUBackend)
            x = temp
    if isinstance(output, np.ndarray):
        np.copyto(output, temp)
    else:
        output = temp
    if output.dtype != output_dtype:
        output = output.astype(output_dtype)
    return output

def _filter_input_modified(image, prefilter, mode, cval, order,GPUBackend='OpenCL'):
    if not prefilter or order < 2:
        return (np.ascontiguousarray(image), 0)
    padded, npad = _prepad_for_spline_filter_modified(image, mode, cval)
    float_dtype = np.promote_types(image.dtype, np.float32)
    filtered = spline_filter_modified(padded, order, output=float_dtype, mode=mode, GPUBackend=GPUBackend)
    return np.ascontiguousarray(filtered), npad

def _check_cval_modified(mode, cval, integer_output):
    if mode == 'constant' and integer_output and not np.isfinite(cval):
        raise NotImplementedError("Non-finite cval is not supported for "
                                  "outputs with integer dtype.")

def affine_transform_prep(input, matrix, offset=0.0, output_shape=None, output=None,
                     order=3, mode='constant', cval=0.0, prefilter=True, GPUBackend='OpenCL', *,
                     texture_memory=False):
    """ Modified from cupyx.scipy.nd_image._interpolation's affine_transform function 
    to use numpy arrays and return values to be used for call to modified affine_transform
    kernel
    """

    _check_parameter_modified('affine_transform', order, mode)

    input = np.asarray(input)
    offset = _fix_sequence_arg_modified(offset, input.ndim, 'offset', float)

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

    if output_shape is None:
        output_shape = input.shape

    matrix = matrix.astype(np.float64, copy=False)
    ndim = input.ndim
    output = _get_output_modified(output, input, shape=output_shape)
    
    if input.dtype.kind in 'iu':
        input = input.astype(np.float32)
    filtered, nprepad = _filter_input_modified(input, prefilter, mode, cval, order, GPUBackend=GPUBackend)

    integer_output = output.dtype.kind in 'iu'
    _check_cval_modified(mode, cval, integer_output)
    large_int = max(_prod_numpy(input.shape), _prod_numpy(output_shape)) > 1 << 31

    m = np.zeros((ndim, ndim + 1), dtype=np.float64)
    m[:, :-1] = matrix
    m[:, -1] = np.asarray(offset, dtype=np.float64)

    return filtered, m, output, mode, cval, order, integer_output

def ResampleFromTo(from_img, to_vox_map,order=3,mode="constant",cval=0.0,out_class=Nifti1Image,GPUBackend='OpenCL'):
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    global mf
    global clp

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
    
    if GPUBackend=='CUDA':
        with ctx:
            image_gpu = cupy.asarray(from_img.dataobj)
            rzs_gpu=cupy.asarray(rzs)

            data_gpu = cndimage.affine_transform(
            image_gpu, rzs_gpu, trans, to_shape, order=order, mode=mode, cval=cval
            )

            data = cupy.asnumpy(data_gpu)
        return out_class(data, to_affine, from_img.header)
    
    elif GPUBackend=='OpenCL':

        filtered, m, output, mode, cval, order, integer_output= affine_transform_prep(from_img.dataobj, rzs, trans, to_shape, order=order, mode=mode, cval=cval, GPUBackend=GPUBackend)
        
        filtered = filtered.astype(np.float32, copy=False)
        m = m.astype(np.float32, copy=False)
        output = output.astype(np.float32, copy=False)
        
        # Move input data from host to device memory
        filtered_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filtered)
        m_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
        output_gpu = clp.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

        assert(np.isfortran(output)==False)
        assert(np.isfortran(m)==False)
        assert(np.isfortran(filtered)==False)

        # Deploy affiner transform kernel
        knl_at(queue, output.shape,
            None,
            filtered_gpu,
            m_gpu,
            output_gpu,
            np.float32(cval),
            np.int32(order),
            np.int32(filtered.shape[0]),
            np.int32(filtered.shape[1]),
            np.int32(filtered.shape[2]),
        )

        # Move kernel output data back to host memory
        clp.enqueue_copy(queue, output, output_gpu)

        if integer_output:
            output = output.astype("int16")

        return out_class(output, to_affine, from_img.header)
    else: # Metal

        filtered, m, output, mode, cval, order, integer_output= affine_transform_prep(from_img.dataobj, rzs, trans, to_shape, order=order, mode=mode, cval=cval,GPUBackend=GPUBackend)
        
        filtered = filtered.astype("float32", copy=False)
        m = m.astype("float32", copy=False)
        output = output.astype("float32", copy=False)
        
        int_params=np.zeros(7,np.int32)
        float_params= np.zeros(2,np.float32)
        int_params[0] = filtered.shape[0]
        int_params[1] = filtered.shape[1]
        int_params[2] = filtered.shape[2]
        int_params[3] = output.shape[0]
        int_params[4] = output.shape[1]
        int_params[5] = output.shape[2]
        int_params[6] = order
        float_params[0] = cval

        assert(np.isfortran(output)==False)
        assert(np.isfortran(m)==False)
        assert(np.isfortran(filtered)==False)

        filtered_gpu = ctx.buffer(filtered)
        m_gpu = ctx.buffer(m) 
        output_gpu = ctx.buffer(output)
        int_params_gpu = ctx.buffer(int_params)
        float_params_gpu = ctx.buffer(float_params)

        ctx.init_command_buffer()

        handle=knl_at(int(np.prod(output.shape)),filtered_gpu,m_gpu,output_gpu,int_params_gpu, float_params_gpu)
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((output_gpu,float_params_gpu))
        output = np.frombuffer(output_gpu,dtype=np.float32).reshape(output.shape)

        if integer_output:
            output = output.astype("int16")

        return out_class(output, to_affine, from_img.header)