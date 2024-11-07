#ifdef _OPENCL
__kernel void mapfilter(__global const  float * HUMap,
                        __global const unsigned char * IsBone,
                        __global const float * UniqueHU,
                        __global unsigned int * CtMap,
                        __global const unsigned int * int_params) {
      
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);

    const unsigned int dimUnique = int_params[0];
    const unsigned int dims_0 = int_params[1];
    const unsigned int dims_1 = int_params[2];
    const unsigned int dims_2 = int_params[3];

    if (x >= dims_0 || y >= dims_1 || z >= dims_2)
        return;
    const unsigned int section_gid = x*dims_1*dims_2 + y*dims_2 + z;
#endif
#ifdef _CUDA
extern "C" __global__ void mapfilter(const float * HUMap,
                                     const unsigned char * IsBone,
                                     const float * UniqueHU,
                                     unsigned int * CtMap,
                                     const unsigned int * int_params) {
      
    const unsigned int x = (blockIdx.x*blockDim.x + threadIdx.x);
    const unsigned int y = (blockIdx.y*blockDim.y + threadIdx.y);
    const unsigned int z = (blockIdx.z*blockDim.z + threadIdx.z);

    const unsigned int dimUnique = int_params[0];
    const unsigned int dims_0 = int_params[1];
    const unsigned int dims_1 = int_params[2];
    const unsigned int dims_2 = int_params[3];
    const unsigned int section_size = int_params[4];

    if (x >= dims_0 || y >= dims_1 || z >= dims_2)
        return;
    const unsigned int section_gid = x*dims_1*dims_2 + y*dims_2 + z;

    if (section_gid >= section_size) return;
#endif
#ifdef _METAL
    ptrdiff_t section_gid = thread_position_in_grid.x;
    const unsigned int dimUnique = int_params[0];
#endif

    // Skip if not bone
    if (IsBone[section_gid] == 0)
        return;

    const float selV = HUMap[section_gid];
    for (unsigned int iw_0 = 0; iw_0 < dimUnique; iw_0++)
    {
        if (selV == UniqueHU[iw_0])
        {
            CtMap[section_gid] =  iw_0;
            break;
        }
    }

}