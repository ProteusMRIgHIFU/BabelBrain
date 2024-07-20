#define SIGNED_INT32_LIM 2147483648
#define UNSIGNED_INT32_LIM 4294967296

typedef float W;
typedef float X;
typedef short Y;

#ifdef _METAL
ptrdiff_t ptrdiff_t_min(ptrdiff_t a, ptrdiff_t b)
{
    if (a < b)
    {
        return a;
    }
    else
    {
        return b;
    }
}
#endif

#ifdef _CUDA
extern "C" __global__ void affine_transform(const float * x, 
                                            const float * mat,
                                            float * y,
                                            const float * float_params,
                                            unsigned int * int_params) 

{
    const float cval = float_params[0];
    const unsigned int order = int_params[0];
    const unsigned int in_dims_0 = int_params[1];
    const unsigned int in_dims_1 = int_params[2];
    const unsigned int in_dims_2 = int_params[3];
    // const unsigned int out_dims_0 = int_params[4];
    const unsigned int out_dims_1 = int_params[5];
    const unsigned int out_dims_2 = int_params[6];
    unsigned int output_idx_tmp = int_params[7];
    unsigned int base_32 = int_params[8];
    unsigned int section_size = int_params[9];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    size_t gid = (size_t)(idx + idy * gridDim.x * blockDim.x + idz * gridDim.x * blockDim.x * gridDim.y * blockDim.y);
    size_t output_idx = output_idx_tmp + (base_32 * UNSIGNED_INT32_LIM);

    size_t true_gid = gid + output_idx; // overall position in output array

    const ptrdiff_t xind =  true_gid/(out_dims_1*out_dims_2);
    const ptrdiff_t yind =  (true_gid-xind*out_dims_1*out_dims_2)/out_dims_2;
    const ptrdiff_t zind =  true_gid -xind*out_dims_1*out_dims_2 - yind * out_dims_2;
    size_t _i  = gid; // current index in output section array

    if (_i >= section_size) return;

#endif
#ifdef _OPENCL
__kernel void affine_transform(__global const float * x, 
                      __global const float * mat,
                      __global float * y,
                      __global const float * float_params,
                      __global unsigned int * int_params) 

{
    const float cval = float_params[0];
    const unsigned int order = int_params[0];
    const unsigned int in_dims_0 = int_params[1];
    const unsigned int in_dims_1 = int_params[2];
    const unsigned int in_dims_2 = int_params[3];
    const unsigned int out_dims_0 = int_params[4];
    const unsigned int out_dims_1 = int_params[5];
    const unsigned int out_dims_2 = int_params[6];
    unsigned int output_idx_tmp = int_params[7];
    unsigned int base_32 = int_params[8];

    size_t out_dims_0_tmp = get_global_size(0);
    size_t out_dims_1_tmp = get_global_size(1);
    size_t out_dims_2_tmp = get_global_size(2);

    const size_t xind_tmp =  get_global_id(0);
    const size_t yind_tmp =  get_global_id(1);
    const size_t zind_tmp =  get_global_id(2);

    size_t gid = xind_tmp*out_dims_1_tmp*out_dims_2_tmp + yind_tmp*out_dims_2_tmp + zind_tmp;
    size_t output_idx = output_idx_tmp + (base_32 * UNSIGNED_INT32_LIM);

    size_t true_gid = gid + output_idx; // overall position in output array

    const ptrdiff_t xind =  true_gid/(out_dims_1*out_dims_2);
    const ptrdiff_t yind =  (true_gid-xind*out_dims_1*out_dims_2)/out_dims_2;
    const ptrdiff_t zind =  true_gid -xind*out_dims_1*out_dims_2 - yind * out_dims_2;
    size_t _i  = gid; // current index in output section array

    // if (_i > (out))

#endif

#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
kernel void affine_transform(const device float * x [[ buffer(0) ]], 
                             const device float * mat [[ buffer(1) ]],
                             device float * y [[ buffer(2) ]],
                             const device float * float_params [[ buffer(3) ]],
                             device unsigned int * int_params [[ buffer(4) ]],
                             uint gid[[thread_position_in_grid]]) 
{
    const float cval = float_params[0];
    const unsigned int order = int_params[0];
    const unsigned int in_dims_0 = int_params[1];
    const unsigned int in_dims_1 = int_params[2];
    const unsigned int in_dims_2 = int_params[3];
    const unsigned int out_dims_0 = int_params[4];
    const unsigned int out_dims_1 = int_params[5];
    const unsigned int out_dims_2 = int_params[6];
    unsigned int output_idx_tmp = int_params[7];
    unsigned int base_32 = int_params[8];

    size_t output_idx = output_idx_tmp + (base_32 * UNSIGNED_INT32_LIM);
    size_t true_gid = gid + output_idx; // overall position in output array
    
    const ptrdiff_t xind =  true_gid/(out_dims_1*out_dims_2);
    const ptrdiff_t yind =  (true_gid-xind*out_dims_1*out_dims_2)/out_dims_2;
    const ptrdiff_t zind =  true_gid -xind*out_dims_1*out_dims_2 - yind * out_dims_2;
    size_t _i  = gid; // current index in output section array
#endif

    W out = 0.0;

    // const int xsize_0 = in_dims_0;
    // const int xsize_1 = in_dims_1;
    // const int xsize_2 = in_dims_2;
    // const unsigned int sx_2 = 1;
    // const unsigned int sx_1 = sx_2 * xsize_2;
    // const unsigned int sx_0 = sx_1 * xsize_1;

    // int in_coord[3] = {xind,yind,zind};

    const ptrdiff_t xsize_0 = in_dims_0;
    const ptrdiff_t xsize_1 = in_dims_1;
    const ptrdiff_t xsize_2 = in_dims_2;
    const size_t sx_2 = 1;
    const size_t sx_1 = sx_2 * xsize_2;
    const size_t sx_0 = sx_1 * xsize_1;

    ptrdiff_t in_coord[3] = {xind,yind,zind};
    
    W c_0 = (W)0.0;
    c_0 += mat[0] * (W)in_coord[0];
    c_0 += mat[1] * (W)in_coord[1];
    c_0 += mat[2] * (W)in_coord[2];
    c_0 += mat[3];

    W c_1 = (W)0.0;
    c_1 += mat[4] * (W)in_coord[0];
    c_1 += mat[5] * (W)in_coord[1];
    c_1 += mat[6] * (W)in_coord[2];
    c_1 += mat[7];

    W c_2 = (W)0.0;
    c_2 += mat[8] * (W)in_coord[0];
    c_2 += mat[9] * (W)in_coord[1];
    c_2 += mat[10] * (W)in_coord[2];
    c_2 += mat[11];

    if ((c_0 < 0) || (c_0 > xsize_0 - 1) || (c_1 < 0) || (c_1 > xsize_1 - 1) || (c_2 < 0) || (c_2 > xsize_2 - 1))
    {
       out = (W)cval;
    }
    else
    {
        if (order == 0)
        {
            ptrdiff_t cf_0 = (ptrdiff_t)floor((W)c_0 + 0.5);
            ptrdiff_t ic_0 = cf_0 * sx_0;
            ptrdiff_t cf_1 = (ptrdiff_t)floor((W)c_1 + 0.5);
            ptrdiff_t ic_1 = cf_1 * sx_1;
            ptrdiff_t cf_2 = (ptrdiff_t)floor((W)c_2 + 0.5);
            ptrdiff_t ic_2 = cf_2 * sx_2;
            // int cf_0 = (int)floor((float)c_0 + 0.5);
            // int ic_0 = cf_0 * sx_0;
            // int cf_1 = (int)floor((float)c_1 + 0.5);
            // int ic_1 = cf_1 * sx_1;
            // int cf_2 = (int)floor((float)c_2 + 0.5);
            // int ic_2 = cf_2 * sx_2;
            out = (W)x[ic_0 + ic_1 + ic_2];
        }
        else
        {
            W wx, wy;
            ptrdiff_t start;
            // int start;

            W weights_0[4];
            
            wx = c_0 - floor(3 & 1 ? c_0 : c_0 + 0.5);
            wy = 1.0 - wx;
            weights_0[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
            weights_0[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
            weights_0[0] = wy * wy * wy / 6.0;
            weights_0[3] = 1.0 - weights_0[0] - weights_0[1] - weights_0[2];
            
            // start = (int)floor((float)c_0) - 1;
            // int ci_0[4];
            start = (ptrdiff_t)floor((W)c_0) - 1;
            ptrdiff_t ci_0[4];
            
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
                #ifdef _METAL
                ci_0[0] = ptrdiff_t_min(ci_0[0], 2 * xsize_0 - 2 - ci_0[0]);
                #else
                ci_0[0] = min(ci_0[0], 2 * xsize_0 - 2 - ci_0[0]);
                #endif
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
                #ifdef _METAL
                ci_0[1] = ptrdiff_t_min(ci_0[1], 2 * xsize_0 - 2 - ci_0[1]);
                #else
                ci_0[1] = min(ci_0[1], 2 * xsize_0 - 2 - ci_0[1]);
                #endif
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
                #ifdef _METAL
                ci_0[2] = ptrdiff_t_min(ci_0[2], 2 * xsize_0 - 2 - ci_0[2]);
                #else
                ci_0[2] = min(ci_0[2], 2 * xsize_0 - 2 - ci_0[2]);
                #endif
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
                #ifdef _METAL
                ci_0[3] = ptrdiff_t_min(ci_0[3], 2 * xsize_0 - 2 - ci_0[3]);
                #else
                ci_0[3] = min(ci_0[3], 2 * xsize_0 - 2 - ci_0[3]);
                #endif
            }
            W w_0;
            
            ptrdiff_t ic_0;
            // int ic_0;

            for (int k_0 = 0; k_0 <= 3; k_0++)
            {
                w_0 = weights_0[k_0];
                ic_0 = ci_0[k_0] * sx_0;
                W weights_1[4];
                wx = c_1 - floor(3 & 1 ? c_1 : c_1 + 0.5);
                wy = 1.0 - wx;
                weights_1[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
                weights_1[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
                weights_1[0] = wy * wy * wy / 6.0;
                weights_1[3] = 1.0 - weights_1[0] - weights_1[1] - weights_1[2];

                start = (ptrdiff_t)floor((W)c_1) - 1;
                ptrdiff_t ci_1[4];
                // start = (int)floor((float)c_1) - 1;
                // int ci_1[4];

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
                    #ifdef _METAL
                    ci_1[0] = ptrdiff_t_min(ci_1[0], 2 * xsize_1 - 2 - ci_1[0]);
                    #else
                    ci_1[0] = min(ci_1[0], 2 * xsize_1 - 2 - ci_1[0]);
                    #endif
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
                    #ifdef _METAL
                    ci_1[1] = ptrdiff_t_min(ci_1[1], 2 * xsize_1 - 2 - ci_1[1]);
                    #else
                    ci_1[1] = min(ci_1[1], 2 * xsize_1 - 2 - ci_1[1]);
                    #endif
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
                    #ifdef _METAL
                    ci_1[2] = ptrdiff_t_min(ci_1[2], 2 * xsize_1 - 2 - ci_1[2]);
                    #else
                    ci_1[2] = min(ci_1[2], 2 * xsize_1 - 2 - ci_1[2]);
                    #endif
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
                    #ifdef _METAL
                    ci_1[3] = ptrdiff_t_min(ci_1[3], 2 * xsize_1 - 2 - ci_1[3]);
                    #else
                    ci_1[3] = min(ci_1[3], 2 * xsize_1 - 2 - ci_1[3]);
                    #endif
                }
                
                W w_1;
                ptrdiff_t ic_1;
                // int ic_1;

                for (int k_1 = 0; k_1 <= 3; k_1++)
                {
                    w_1 = weights_1[k_1];
                    ic_1 = ci_1[k_1] * sx_1;
                    W weights_2[4];
                    wx = c_2 - floor(3 & 1 ? c_2 : c_2 + 0.5);
                    wy = 1.0 - wx;
                    weights_2[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
                    weights_2[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
                    weights_2[0] = wy * wy * wy / 6.0;
                    weights_2[3] = 1.0 - weights_2[0] - weights_2[1] - weights_2[2];
                    
                    // start = (int)floor((float)c_2) - 1;
                    // int ci_2[4];
                    start = (ptrdiff_t)floor((W)c_2) - 1;
                    ptrdiff_t ci_2[4];

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
                        #ifdef _METAL
                        ci_2[0] = ptrdiff_t_min(ci_2[0], 2 * xsize_2 - 2 - ci_2[0]);
                        #else
                        ci_2[0] = min(ci_2[0], 2 * xsize_2 - 2 - ci_2[0]);
                        #endif
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
                        #ifdef _METAL
                        ci_2[1] = ptrdiff_t_min(ci_2[1], 2 * xsize_2 - 2 - ci_2[1]);
                        #else
                        ci_2[1] = min(ci_2[1], 2 * xsize_2 - 2 - ci_2[1]);
                        #endif
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
                        #ifdef _METAL
                        ci_2[2] = ptrdiff_t_min(ci_2[2], 2 * xsize_2 - 2 - ci_2[2]);
                        #else
                        ci_2[2] = min(ci_2[2], 2 * xsize_2 - 2 - ci_2[2]);
                        #endif
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
                        #ifdef _METAL
                        ci_2[3] = ptrdiff_t_min(ci_2[3], 2 * xsize_2 - 2 - ci_2[3]);
                        #else
                        ci_2[3] = min(ci_2[3], 2 * xsize_2 - 2 - ci_2[3]);
                        #endif
                    }
                    W w_2;

                    ptrdiff_t ic_2;
                    // int ic_2;

                    for (int k_2 = 0; k_2 <= 3; k_2++)
                    {
                        w_2 = weights_2[k_2];
                        ic_2 = ci_2[k_2] * sx_2;
                        if ((ic_0 < 0) || (ic_1 < 0) || (ic_2 < 0)) 
                        {
                            out += (W)cval * (W)(w_0 * w_1 * w_2);
                        } 
                        else 
                        {
                            W val = (W)x[ic_0 + ic_1 + ic_2];
                            out += val * (W)(w_0 * w_1 * w_2);
                        }
                    }
                }
            }
        }
    }
    // if(order == 0)
    // {
    //     y[_i] = (float)rint((float)out);
    // }
    // else
    // {
    //     y[_i] = (float)out;
    // }

    y[_i] = (float)rint((W)out);

}
