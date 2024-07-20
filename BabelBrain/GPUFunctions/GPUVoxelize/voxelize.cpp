#ifdef _METAL
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;
#endif
#ifdef _CUDA
#include <helper_math.h>
#endif 

#define float_error 0.000001
#define SIGNED_INT32_LIM 2147483648
#define UNSIGNED_INT32_LIM 4294967296

// use Xor for voxels whose corresponding bits have to flipped
#ifdef _OPENCL
inline void setBitXor(__global unsigned int* voxel_table, size_t index) 
#endif
#ifdef _CUDA
__device__ inline void setBitXor(unsigned int* voxel_table, size_t index) 
#endif
#ifdef _METAL
inline void setBitXor( device atomic_uint * voxel_table, size_t index) 
#endif
{
    size_t int_location = index / 32;
    unsigned int bit_pos = 31 - ((unsigned int)(index) % 32); // we count bit positions RtL, but array indices LtR
    unsigned int mask = 1 << bit_pos;
#if defined(_OPENCL) 
    atom_xor(&(voxel_table[int_location]), mask);
#endif
#if defined(_CUDA) 
    atomicXor(&(voxel_table[int_location]), mask);
#endif
#ifdef _METAL
    atomic_fetch_xor_explicit(&(voxel_table[int_location]), mask,memory_order_relaxed);
#endif
}

//check the location with point and triangle
#if defined(_CUDA)
__device__
#endif
inline int check_point_triangle(float2 v0, float2 v1, float2 v2, float2 point)
{
    float2 PA = point - v0;
    float2 PB = point - v1;
    float2 PC = point - v2;

    float t1 = PA.x*PB.y - PA.y*PB.x;
    if (fabs(t1) < float_error&&PA.x*PB.x <= 0 && PA.y*PB.y <= 0)
        return 1;

    float t2 = PB.x*PC.y - PB.y*PC.x;
    if (fabs(t2) < float_error&&PB.x*PC.x <= 0 && PB.y*PC.y <= 0)
        return 2;

    float t3 = PC.x*PA.y - PC.y*PA.x;
    if (fabs(t3) < float_error&&PC.x*PA.x <= 0 && PC.y*PA.y <= 0)
        return 3;

    if (t1*t2 > 0 && t1*t3 > 0)
        return 0;
    else
        return -1;
}

//find the x coordinate of the voxel
#if defined(_CUDA)
__device__
#endif
inline float get_x_coordinate(float3 n, float3 v0, float2 point)
{
    return (-(n.y*(point.x - v0.y) + n.z*(point.y - v0.z)) / n.x + v0.x);
}

//check the triangle is counterclockwise or not
#if defined(_CUDA)
__device__
#endif
inline bool checkCCW(float2 v0, float2 v1, float2 v2)
{
    float2 e0 = v1 - v0;
    float2 e1 = v2 - v0;
    float result = e0.x*e1.y - e1.x*e0.y;
    if (result > 0)
        return true;
    else
        return false;
}

//top-left rule
#if defined(_CUDA)
__device__
#endif
inline bool TopLeftEdge(float2 v0, float2 v1)
{
    return ((v1.y<v0.y) || (v1.y == v0.y&&v0.x>v1.x));
}


//generate solid voxelization
#ifdef _OPENCL
__kernel void voxelize_triangle_solid(__global const float* triangle_data, 
                                      __global unsigned int* voxel_table)
{
    size_t thread_id = get_global_id(0);
#endif
#ifdef _CUDA
 extern "C" __global__ void voxelize_triangle_solid(const float* triangle_data, 
                                        unsigned int* voxel_table)
{
    size_t thread_id = (size_t)(blockIdx.x*blockDim.x + threadIdx.x);
#endif
#ifdef _METAL
kernel void voxelize_triangle_solid(const device float* triangle_data [[ buffer(0) ]], 
                                     device atomic_uint* voxel_table [[ buffer(1) ]],
                                    uint gid[[thread_position_in_grid]])
{
    size_t thread_id = gid;
#endif


    if (thread_id < info_n_triangles) { // every thread works on specific triangles in its stride
        size_t t = thread_id * 9; // triangle contains 9 vertices

                                  // COMPUTE COMMON TRIANGLE PROPERTIES
                                  // Move vertices to origin using bbox
        #if defined(_OPENCL) 
        float3 v0 = (float3)(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info_min;
        float3 v1 = (float3)(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info_min;
        float3 v2 = (float3)(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info_min;
        #endif
        #if defined(_METAL) || defined(_CUDA)
        float3 v0 = {triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]};
        v0-=info_min;
        float3 v1 = {triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]};
        v1-= info_min;
        float3 v2 = {triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]};
        v2-= info_min;    
        #endif
        // Edge vectors
        float3 e0 = v1 - v0;
        float3 e1 = v2 - v1;
        float3 e2 = v0 - v2;
        // Normal vector pointing up from the triangle
        float3 n = normalize(cross(e0, e1));
        if (fabs(n.x) < float_error)
            return;

        // Calculate the projection of three point into yoz plane
        #if defined(_OPENCL) 
        float2 v0_yz = (float2)(v0.y, v0.z);
        float2 v1_yz = (float2)(v1.y, v1.z);
        float2 v2_yz = (float2)(v2.y, v2.z);
        #endif
        #if defined(_METAL) || defined(_CUDA)
        float2 v0_yz = {v0.y, v0.z};
        float2 v1_yz = {v1.y, v1.z};
        float2 v2_yz = {v2.y, v2.z};
        #endif
        // Set the triangle counterclockwise
        if (!checkCCW(v0_yz, v1_yz, v2_yz))
        {
            float2 v3 = v1_yz;
            v1_yz = v2_yz;
            v2_yz = v3;
        }

        // COMPUTE TRIANGLE BBOX IN GRID
        // Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
        #if  defined(_CUDA)
        float2 bbox_max = fmaxf(v0_yz, fmaxf(v1_yz, v2_yz));
        float2 bbox_min = fminf(v0_yz, fminf(v1_yz, v2_yz));
        #else
        float2 bbox_max = max(v0_yz, max(v1_yz, v2_yz));
        float2 bbox_min = min(v0_yz, min(v1_yz, v2_yz));
        #endif
        
        #if defined(_OPENCL)
        float2 bbox_max_grid = (float2)(floor(bbox_max.x / info_unit.y - 0.5), floor(bbox_max.y / info_unit.z - 0.5));
        float2 bbox_min_grid = (float2)(ceil(bbox_min.x / info_unit.y - 0.5), ceil(bbox_min.y / info_unit.z - 0.5));
        #endif
        #if defined(_METAL) || defined(_CUDA)
        float2 bbox_max_grid = {floor(bbox_max.x / info_unit.y - (float)0.5), floor(bbox_max.y / info_unit.z - (float)0.5)};
        float2 bbox_min_grid = {ceil(bbox_min.x / info_unit.y - (float)0.5), ceil(bbox_min.y / info_unit.z -(float) 0.5)};

        #endif
        
        for (int y = bbox_min_grid.x; y <= bbox_max_grid.x; y++)
        {
            if ((y<0) || (y>=info_gridsize.y))
                continue; 
            for (int z = bbox_min_grid.y; z <= bbox_max_grid.y; z++)
            {
                if ((z<0) || (z>=info_gridsize.z))
                    continue;
                 #if defined(_OPENCL)
                float2 point = (float2)((y + 0.5)*info_unit.y, (z + 0.5)*info_unit.z);
                #endif
                #if defined(_METAL) || defined(_CUDA)
                float2 point = {(y + (float) 0.5)*info_unit.y, (z + (float) 0.5)*info_unit.z};
                #endif
                int checknum = check_point_triangle(v0_yz, v1_yz, v2_yz, point);
                if ((checknum == 1 && TopLeftEdge(v0_yz, v1_yz)) || (checknum == 2 && TopLeftEdge(v1_yz, v2_yz)) || (checknum == 3 && TopLeftEdge(v2_yz, v0_yz)) || (checknum == 0))
                {
                    int xmax = (int)(get_x_coordinate(n, v0, point) / info_unit.x - 0.5);
                    for (int x = 0; x <= xmax; x++)
                    {
                        if (x>=info_gridsize.x)
                            continue;
                        size_t location =
                            (size_t)(x) +
                            ((size_t)(y) * (size_t)(info_gridsize.x)) +
                            ((size_t)(z) * ((size_t)(info_gridsize.y) * (size_t)(info_gridsize.x))); 
                        setBitXor(voxel_table, location);
                        
                        continue;
                    }
                }
            }
        }
    }
}
#ifdef _OPENCL
inline bool checkVoxelInd(const size_t index, __global const unsigned int * vtable)
#endif
#ifdef _CUDA
__device__ inline bool checkVoxelInd(const size_t index,const unsigned int * vtable)
#endif
#ifdef _METAL
inline bool checkVoxelInd(const size_t index, device const unsigned int * vtable)
#endif
{
    size_t int_location = index / 32;
    unsigned int bit_pos = 31 - (((unsigned int)(index)) % 32); // we count bit positions RtL, but array indices LtR
    unsigned int mask = 1 << bit_pos;
    if (vtable[int_location] & mask)
        return true;
    return false;
}

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
                                          const unsigned int gz,
                                          const unsigned int section_size)
                                          {
    size_t k = (size_t)(blockIdx.x*blockDim.x + threadIdx.x);

    if (k >= section_size) return;
#endif

#ifdef _METAL                      
kernel void ExtractPoints( device const unsigned int* voxel_table [[ buffer(0) ]],
                           device atomic_uint * globalcount [[ buffer(1) ]],
                           device unsigned int *int_params [[buffer(2)]],
                           device float * Points [[ buffer(3) ]],
                           uint gid[[thread_position_in_grid]])
                           {
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
            #if  defined(_CUDA)
            size_t nt = (size_t)(atomicAdd(&globalcount[0], 1));
            #endif
            #ifdef _METAL
            size_t nt = (size_t)(atomic_fetch_add_explicit(globalcount,1,memory_order_relaxed));
            #endif
            nt-=(size_t)basePoint;
            Points[nt*3]=(float)i;
            Points[nt*3+1]=(float)j;
            Points[nt*3+2]=(float)k;
        }
    }
}
