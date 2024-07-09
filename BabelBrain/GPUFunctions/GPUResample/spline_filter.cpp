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
