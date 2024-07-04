#ifdef _OPENCL
__kernel void median_reflect(
                             __global const  PixelType * input,
                             __global PixelType * output,
                             const int dims_0,
                             const int dims_1,
                             const int dims_2,
                             const int filter_size_0,
                             const int filter_size_1,
                             const int filter_size_2) {
      
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    int _i = x*dims_1*dims_2 + y*dims_2 + z;
#endif
#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
kernel void median_reflect(
                           const device PixelType * input [[ buffer(0) ]],
                           device PixelType * output [[ buffer(1) ]],
                           const device int * int_params [[ buffer(2)]], 
                           uint gid[[thread_position_in_grid]]) {
    
    const int dims_0 = int_params[0];
    const int dims_1 = int_params[1];
    const int dims_2 = int_params[2];
    const int filter_size_0 = int_params[3];
    const int filter_size_1 = int_params[4];
    const int filter_size_2 = int_params[5];

    const int x = gid/(dims_1*dims_2);
    const int y = (gid-x*dims_1*dims_2)/dims_2;
    const int z = gid -x*dims_1*dims_2 - y * dims_2;
    #define _i gid
#endif
    
    
    int ind_2 = z - (filter_size_2/2);
    int ind_1 = y - (filter_size_1/2);
    int ind_0 = x - (filter_size_0/2);
    
    int iv = 0;
    int filter_size = filter_size_0*filter_size_1*filter_size_2;
    PixelType values[343]; // We allocate for max scenario (7x7x7) and adjust size from there
    
    // Grab values from neigbouring voxels for given median filter size
    for (int iw_0 = 0; iw_0 < filter_size_0; iw_0++)
    {
        int ix_0 = ind_0 + iw_0;

        if (ix_0 < 0) 
            ix_0 = - 1 -ix_0;
        else
            ix_0 = min(ix_0, 2 * dims_0 - 1 - ix_0);

        for (int iw_1 = 0; iw_1 < filter_size_1; iw_1++)
        {
            int ix_1 = ind_1 + iw_1;

            if (ix_1 < 0) 
                ix_1 = - 1 -ix_1;
            else
                ix_1 = min(ix_1, 2 * dims_1 - 1 - ix_1);
                
            for (int iw_2 = 0; iw_2 < filter_size_2; iw_2++)
                {
                    int ix_2 = ind_2 + iw_2;

                    if (ix_2 < 0) 
                        ix_2 = - 1 -ix_2;
                    else
                        ix_2 = min(ix_2, 2 * dims_2 - 1 - ix_2);

                // inner-most loop
              
                values[iv++] = input[ix_0*dims_1*dims_2 + ix_1*dims_2 + ix_2];
                
            }
        }
    }
     
    // Sort values and get median value
    const int size = filter_size;
    const int midpoint = size/2;
    int gap = 1;
    while (gap < filter_size){
        gap = 3*gap+1;
    }

    while (gap > 1) {
        gap /= 3;
        for (int i = gap; i < size; ++i) {
            PixelType value = values[i];
            int j = i - gap;
            while (j >= 0 && value < values[j]) {
                values[j + gap] = values[j];
                j -= gap;
            }
            values[j + gap] = value;
        }
    }
    
    // Return median value
    output[_i]=values[midpoint];

    }
