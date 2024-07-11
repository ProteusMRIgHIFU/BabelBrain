#define SIGNED_INT32_LIM 2147483648
#define UNSIGNED_INT32_LIM 4294967296

#ifdef _CUDA
extern "C" __global__ void binary_erosion(const unsigned char * x,
                                          const unsigned char * w,
                                          const int * int_params,
                                          unsigned char * y
                                          )
{

    ptrdiff_t section_size = int_params[19];
    
    ptrdiff_t output_size_0 = int_params[0];
    ptrdiff_t output_size_1 = int_params[1];
    ptrdiff_t output_size_2 = int_params[2];
    ptrdiff_t output_size = output_size_2 * output_size_1 * output_size_0;

    int section_xind = blockIdx.x * blockDim.x + threadIdx.x;
    int section_yind = blockIdx.y * blockDim.y + threadIdx.y;
    int section_zind = blockIdx.z * blockDim.z + threadIdx.z;

    size_t section_gid = (size_t)(section_xind + section_yind * gridDim.x * blockDim.x + section_zind * gridDim.x * blockDim.x * gridDim.y * blockDim.y);

    if (section_gid >= section_size) return;

#endif
#ifdef _OPENCL
__kernel void binary_erosion(__global const unsigned char * x,
                             __global const unsigned char * w,
                             __global const int * int_params,
                             __global unsigned char * y
                             )
{

    ptrdiff_t section_size_0 = get_global_size(0);
    ptrdiff_t section_size_1 = get_global_size(1);
    ptrdiff_t section_size_2 = get_global_size(2);
    ptrdiff_t section_size = section_size_2*section_size_1*section_size_0;
    
    ptrdiff_t output_size_0 = int_params[0];
    ptrdiff_t output_size_1 = int_params[1];
    ptrdiff_t output_size_2 = int_params[2];
    ptrdiff_t output_size = output_size_2 * output_size_1 * output_size_0;
    
    ptrdiff_t section_xind = get_global_id(0);
    ptrdiff_t section_yind = get_global_id(1);
    ptrdiff_t section_zind = get_global_id(2);

    ptrdiff_t section_gid = section_xind*section_size_1*section_size_2 + section_yind*section_size_2 + section_zind;

#endif
#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
kernel void binary_erosion(const device unsigned char * x [[ buffer(0) ]], 
                           const device unsigned char * w [[ buffer(1) ]], 
                           const device int * int_params[[ buffer(2) ]],
                           device unsigned char * y [[ buffer(3) ]],
                           uint gid[[thread_position_in_grid]])
{
    
    ptrdiff_t section_size = int_params[19];

    ptrdiff_t output_size_0 = int_params[0];
    ptrdiff_t output_size_1 = int_params[1];
    ptrdiff_t output_size_2 = int_params[2];
    ptrdiff_t output_size = output_size_2 * output_size_1 * output_size_0;

    ptrdiff_t section_gid = (ptrdiff_t) gid;
    
#endif

    ptrdiff_t input_size_0 = int_params[0];
    ptrdiff_t input_size_1 = int_params[1];
    ptrdiff_t input_size_2 = int_params[2];
    ptrdiff_t input_stride_0 = int_params[3];
    ptrdiff_t input_stride_1 = int_params[4];
    ptrdiff_t input_stride_2 = int_params[5];
    int true_val = int_params[6];
    int false_val = int_params[7];
    int border_value = int_params[8];
    int center_is_true = int_params[9];
    const int w_shape[3] = {int_params[10],int_params[11],int_params[12]};
    const int offsets[3] = {int_params[13],int_params[14],int_params[15]};
    ptrdiff_t start_idx = int_params[16];
    int base_32 = int_params[17];
    int padding = int_params[18];

    // Adjust start index
    start_idx += (base_32 * SIGNED_INT32_LIM);

    // Get overall position in output array
    ptrdiff_t output_gid = section_gid + start_idx; 

    // For middle sections, skip padding at start of array
    if (section_gid < (padding * output_size_1 * output_size_2) && start_idx != 0)
    {
        return;
    }

    // For middle sections, skip padding at end of array
    if (section_gid >= (section_size - (padding * output_size_1 * output_size_2)) && output_gid < (output_size - (padding * output_size_1 * output_size_2)))
    {
        return;
    }

    // Get z index, y index, and x index for output_gid
    ptrdiff_t tmp_gid = output_gid;
    ptrdiff_t ind_2 = tmp_gid % output_size_2 - offsets[2]; tmp_gid /= output_size_2;
    ptrdiff_t ind_1 = tmp_gid % output_size_1 - offsets[1]; tmp_gid /= output_size_1;
    ptrdiff_t ind_0 = tmp_gid - offsets[0];

    #ifdef _CUDA
    const unsigned char* data = (const unsigned char*)&x[0];
    #endif
    #ifdef _OPENCL
    __global const unsigned char* data = (__global const unsigned char*)&x[0];
    #endif
    #ifdef _METAL
    device const unsigned char* data = (const device unsigned char*)&x[0];
    #endif

    unsigned char _in = (unsigned char)x[section_gid];
    if (center_is_true && _in == (unsigned char)false_val)
    {
        y[section_gid] = (unsigned char)_in;
        return;
    }
    y[section_gid] = (unsigned char)true_val;

    // loop through neighbouring voxels (ix_0, ix_1, ix_2 are indexes along each dimension)
    // Extent of neighbouring voxels is determined by w_shape
    for (int iw_0 = 0; iw_0 < w_shape[0]; iw_0++)
    {
        ptrdiff_t ix_0 = ind_0 + iw_0;

        // assign -1 for out of bound indexes
        if ((ix_0 < 0) || ix_0 >= input_size_0) 
        {
            ix_0 = -1;
        }
        ix_0 *= input_stride_0;

        for (int iw_1 = 0; iw_1 < w_shape[1]; iw_1++)
        {
            ptrdiff_t ix_1 = ind_1 + iw_1;

            // assign -1 for out of bound indexes
            if ((ix_1 < 0) || ix_1 >= input_size_1) 
            {
                ix_1 = -1;
            }
            ix_1 *= input_stride_1;

            for (int iw_2 = 0; iw_2 < w_shape[2]; iw_2++)
            {
                ptrdiff_t ix_2 = ind_2 + iw_2;

                // assign -1 for out of bound indexes
                if ((ix_2 < 0) || ix_2 >= input_size_2) {
                    ix_2 = -1;
                }
                ix_2 *= input_stride_2;
                
                // If out of bounds and border_value is set to 0, set current voxel to false_val
                if ((ix_0 < 0) || (ix_1 < 0) || (ix_2 < 0)) 
                {
                    if (!border_value) 
                    {
                        y[section_gid] = (unsigned char)false_val;
                        return;
                    }
                } 
                else 
                {
                    // get overall index for selected neighbouring voxel
                    ptrdiff_t input_idx = ix_0 + ix_1 + ix_2 - start_idx;

                    #ifdef _CUDA
                    unsigned char nn = (*(unsigned char*)&data[input_idx]) ? true_val : false_val;
                    #endif
                    #ifdef _OPENCL
                    unsigned char nn = (*(__global unsigned char*)&data[input_idx]) ? true_val : false_val;
                    #endif
                    #ifdef _METAL
                    unsigned char nn = (*(device unsigned char*)&data[input_idx]) ? true_val : false_val;
                    #endif
                    if (!nn) 
                    {
                        y[section_gid] = (unsigned char)false_val;
                        return;
                    }
                    
                }
            }
        }
    };
}