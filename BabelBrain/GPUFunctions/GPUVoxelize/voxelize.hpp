#ifdef _METAL
#include <metal_stdlib>
#include <metal_atomic>
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
