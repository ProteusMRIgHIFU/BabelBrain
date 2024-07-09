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