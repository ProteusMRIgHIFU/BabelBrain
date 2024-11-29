#ifdef _OPENCL
__kernel void ExtractPoints(__global const unsigned int* voxel_table,
                             __global uint * globalcount,
                             __global float * Points,
                             __global uint * int_params,
                             const unsigned int gx,
                             const unsigned int gy,
                             const unsigned int gz)
                            {
    size_t k = get_global_id(0);
#endif
#ifdef _CUDA
 extern "C" __global__ void ExtractPoints(const unsigned int* voxel_table,
                                          unsigned int * globalcount,
                                          float * Points,
                                          const unsigned int * int_params,
                                          const unsigned int gx,
                                          const unsigned int gy,
                                          const unsigned int gz)
                                          {
    size_t k = (size_t)(blockIdx.x*blockDim.x + threadIdx.x);
#endif
#ifdef _METAL                      
kernel void ExtractPoints( device const unsigned int* voxel_table [[ buffer(0) ]],
                           device atomic_uint * globalcount [[ buffer(1) ]],
                           device unsigned int *int_params [[buffer(2)]],
                           device float * Points [[ buffer(3) ]],
                           uint gid[[thread_position_in_grid]])
                           {
#endif
#if defined(_METAL) || defined(_MLX)
    #ifdef _MLX
    uint gid = thread_position_in_grid.x;
    #endif
    size_t k = (size_t)gid;
#endif
    #define  total  int_params[0]
    size_t base = (size_t)(int_params[1]);
    size_t base32 = (size_t)int_params[2];

    #define  basePoint int_params[3]
    
    if (base32 > 0){
        base += base32 * UNSIGNED_INT32_LIM;
    }

    if (k < (size_t)total) {
        size_t n= k + ((size_t)(base));
        if (checkVoxelInd(n,voxel_table))
        {
            size_t k=n/((size_t)(gx*gy));
            size_t j=(n-((size_t)(k*(gx*gy))))/((size_t)gx);
            size_t i=n-k*((size_t)(gx*gy))-j*((size_t)gx);
            #if defined(_OPENCL) 
            size_t nt = (size_t)(atomic_inc(globalcount));
            #endif
            #if defined(_CUDA)
            size_t nt = (size_t)(atomicAdd(&globalcount[0], 1));
            #endif
            #if defined(_METAL) || defined(_MLX)
            size_t nt = (size_t)(atomic_fetch_add_explicit(&globalcount[0],1,memory_order_relaxed));
            #endif
            nt-=(size_t)basePoint;
            #if defined(_MLX)
            atomic_store_explicit(&Points[nt*3],(float)i,memory_order_relaxed);
            atomic_store_explicit(&Points[nt*3+1],(float)j,memory_order_relaxed);
            atomic_store_explicit(&Points[nt*3+2],(float)k,memory_order_relaxed);
            #endif
            #if defined(_OPENCL) || defined(_CUDA) || defined(_METAL)
            Points[nt*3]=(float)i;
            Points[nt*3+1]=(float)j;
            Points[nt*3+2]=(float)k;
            #endif
        }
    }
}