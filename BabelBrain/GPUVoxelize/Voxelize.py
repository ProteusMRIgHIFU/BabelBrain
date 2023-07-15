import numpy as np
import pyopencl as cl
import os
from numba import jit,njit, prange
import sys
import platform
from pathlib import Path
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

@njit
def checkVoxelInd(location, vtable):
    int_location = location // 32
    bit_pos = 31 - (location % 32)
    if (vtable[int_location]) & (1 << bit_pos):
        return True
    return False

@njit(parallel=True)
def calctotalpoints(gridsize,vtable):
    y = np.zeros(1,np.int64)
    total=np.prod(np.array(gridsize))
    for i in prange(total):
        if checkVoxelInd(i,vtable):
            y += 1
    return y


@jit()
def ExtractPoints(gridsize,Points,vtable):
    total=np.prod(gridsize)
    nt=np.zeros(1,np.int64)
    for n in range(total):
        if checkVoxelInd(n,vtable):
            k=n//(gridsize[0]*gridsize[1])
            j=(n-k*(gridsize[0]*gridsize[1]))//gridsize[0]
            i=n-k*(gridsize[0]*gridsize[1])-j*gridsize[0]
            Points[nt,0]=i
            Points[nt,1]=j
            Points[nt,2]=k
            nt+=1
    print(nt)


##probably not the most optimal of filters.. .but still WAY faster than CPU based
_SCode_p1='''
#ifdef _METAL
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;
#endif
#ifdef _CUDA
#include <helper_math.h>
#endif 


typedef unsigned int uint32_t;

#define float_error 0.000001

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
'''


_SCode_p3='''
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
                if ((z<0) || (y>=info_gridsize.z))
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
                             __global unsigned int * globalcount,
                             __global float * Points,
                            const unsigned int total,
                            const unsigned int base,
                            const unsigned int basePoint,
                            const unsigned int gx,
                            const unsigned int gy,
                            const unsigned int gz)
                            {
    size_t k = get_global_id(0);
#endif

#ifdef _CUDA
 extern "C" __global__ void ExtractPoints( const unsigned int* voxel_table,
                              unsigned int * globalcount,
                              float * Points,
                            const unsigned int total,
                            const unsigned int base,
                            const unsigned int basePoint,
                            const unsigned int gx,
                            const unsigned int gy,
                            const unsigned int gz)
                            {
    size_t k = (size_t)(blockIdx.x*blockDim.x + threadIdx.x);
#endif

#ifdef _METAL                      
kernel void ExtractPoints( device const unsigned int* voxel_table [[ buffer(0) ]],
                           device atomic_uint * globalcount [[ buffer(1) ]],
                          constant unsigned int *inparas [[buffer(2)]],
                          device float * Points [[ buffer(3) ]],
                            uint gid[[thread_position_in_grid]])
                            {
    size_t k = (size_t)gid;
    #define  total  inparas[0]
    #define  base  inparas[1]
    #define  basePoint inparas[2]
    
#endif
    if (k < (size_t)total) {
        size_t n=k+ ((size_t)(base));
        if (checkVoxelInd(n,voxel_table))
        {
            size_t k=n/((size_t)(gx*gy));
            size_t j=(n-((size_t)(k*(gx*gy))))/((size_t)gx);
            size_t i=n-k*((size_t)(gx*gy))-j*((size_t)gx);
            #if defined(_OPENCL) 
            size_t nt = (size_t)(atomic_inc(globalcount));
            #endif
            #if  defined(_CUDA)
            size_t nt = (size_t)(atomicInc(globalcount,1));
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
'''

Platforms=None
queue = None
prgcl = None
ctx = None
clp = None

def InitOpenCL(DeviceName='AMD'):
    import pyopencl as cl
    global Platforms
    global queue 
    global ctx
    global clp
    clp=cl
    
    Platforms=cl.get_platforms()
    if len(Platforms)==0:
        raise SystemError("No OpenCL platforms")
    SelDevice=None
    for device in Platforms[0].get_devices():
        print(device.name)
        if DeviceName in device.name:
            SelDevice=device
    if SelDevice is None:
        raise SystemError("No OpenCL device containing name [%s]" %(DeviceName))
    else:
        print('Selecting device: ', SelDevice.name)
    ctx = cl.Context([SelDevice])
    queue = cl.CommandQueue(ctx)

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

    if selDevice is None:
        raise SystemError("There are no devices supporting CUDA or that matches selected device.")
      
    ctx=selDevice
    clp=cp
    
def InitMetal(DeviceName='AMD'):
    
    global ctx
    
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

def Voxelize(inputMesh,targetResolution=1333/500e3/6*0.75*1e3,GPUBackend='OpenCL'):
    global ctx
    global clp
    global queue
    
    # global knl_2px
    if np.isfortran(inputMesh.triangles):
        triangles=np.ascontiguousarray(inputMesh.triangles).astype(np.float32)
    else:
        triangles=inputMesh.triangles.astype(np.float32)
        

    n_triangles=inputMesh.triangles.shape[0]

    print('GPU Voxelizing # triangles', n_triangles)

    r=inputMesh.bounding_box.bounds
    dims=np.diff(r,axis=0).flatten()
    gx=int(np.ceil(dims[0]/targetResolution))
    gy=int(np.ceil(dims[1]/targetResolution))
    gz=int(np.ceil(dims[2]/targetResolution))
    dxzy=dims/np.array([gx,gy,gz])
    print('spatial step and  maximal grid dimensions',dxzy,gx,gy,gz)

    vtable_size = int(np.ceil((gx*gy*gz) / 32.0) * 4)
    
    vtable=np.zeros(vtable_size,np.uint8)
    if GPUBackend=='CUDA':
        with ctx:
            SCode_p2='''
                __constant__ float3 info_min={{ {xmin:10.9f},{ymin:10.9f},{zmin:10.9f} }};
                __constant__ float3 info_max={{ {xmax:10.9f},{ymax:10.9f},{zmax:10.9f} }};
                __constant__ uint3 info_gridsize={{ {gx},{gy},{gz} }};
                __constant__ size_t info_n_triangles={n_triangles};
                __constant__ float3 info_unit={{{dx:10.9f},{dy:10.9f},{dz:10.9f} }};
            '''.format(xmin=r[0,0],ymin=r[0,1],zmin=r[0,2],xmax=r[1,0],ymax=r[1,1],zmax=r[1,2],
                        n_triangles=n_triangles,gx=gx,gy=gy,gz=gz,dx=dxzy[0],dy=dxzy[1],dz=dxzy[2])
            if platform.system()=='Windows':
                sys.executable.split('\\')[:-1]
                options=('-I',os.path.join(os.getenv('CUDA_PATH'),'Library','Include'),
                         '-I',str(resource_path()))
            else:
                options=('-I',str(resource_path()))
            prgcl  = clp.RawModule(code= "#define _CUDA\n"+_SCode_p1+SCode_p2+_SCode_p3,
                                 options=options)
            knl=prgcl.get_function("voxelize_triangle_solid")

            vtable_dev=clp.zeros(vtable.shape,clp.uint8)
            triangles_dev=clp.asarray(triangles)
            Block=(64,1,1)
            Grid=(int(n_triangles//Block[0]+1),1,1)
            knl(Grid,Block,(triangles_dev,vtable_dev))
            vtable=vtable_dev.get()
            vtable=np.frombuffer(vtable,np.uint32)
    elif GPUBackend=='OpenCL':
        SCode_p2='''
                __constant float3 info_min={{ {xmin:10.9f},{ymin:10.9f},{zmin:10.9f} }};
                __constant float3 info_max={{ {xmax:10.9f},{ymax:10.9f},{zmax:10.9f} }};
                __constant uint3 info_gridsize={{ {gx},{gy},{gz} }};
                __constant size_t info_n_triangles={n_triangles};
                __constant float3 info_unit={{{dx:10.9f},{dy:10.9f},{dz:10.9f} }};
            '''.format(xmin=r[0,0],ymin=r[0,1],zmin=r[0,2],xmax=r[1,0],ymax=r[1,1],zmax=r[1,2],
                        n_triangles=n_triangles,gx=gx,gy=gy,gz=gz,dx=dxzy[0],dy=dxzy[1],dz=dxzy[2])

        mf=clp.mem_flags
        prg=clp.Program(ctx,"#define _OPENCL\n"+_SCode_p1+SCode_p2+_SCode_p3).build()
        vtable_dev=clp.Buffer(ctx, mf.READ_WRITE, vtable.nbytes)
        triangles_dev=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=triangles)
        prg.voxelize_triangle_solid(queue,[n_triangles],None,triangles_dev,vtable_dev)
        queue.finish()
        clp.enqueue_copy(queue, vtable,vtable_dev)
        queue.finish()
        vtable=np.frombuffer(vtable,np.uint32)
    else:
        assert(GPUBackend=='Metal')
        SCode_p2='''
                constant float3 info_min={{ {xmin:10.9f},{ymin:10.9f},{zmin:10.9f} }};
                constant float3 info_max={{ {xmax:10.9f},{ymax:10.9f},{zmax:10.9f} }};
                constant uint3 info_gridsize={{ {gx},{gy},{gz} }};
                constant size_t info_n_triangles={n_triangles};
                constant float3 info_unit={{{dx:10.9f},{dy:10.9f},{dz:10.9f} }};
            '''.format(xmin=r[0,0],ymin=r[0,1],zmin=r[0,2],xmax=r[1,0],ymax=r[1,1],zmax=r[1,2],
                        n_triangles=n_triangles,gx=gx,gy=gy,gz=gz,dx=dxzy[0],dy=dxzy[1],dz=dxzy[2])

        sdefine='''
        #define _METAL
        #define gx {gx}
        #define gy {gy}
        #define gz {gz}
        '''.format(gx=gx,gy=gy,gz=gz)
        prg = ctx.kernel(sdefine+_SCode_p1+SCode_p2+_SCode_p3)
        vtable_dev=ctx.buffer(vtable.nbytes)
        triangles_dev=ctx.buffer(triangles)
        ctx.init_command_buffer()
        handle = prg.function('voxelize_triangle_solid')(n_triangles,triangles_dev,vtable_dev)
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((vtable_dev,triangles_dev))
        vtable=np.frombuffer(vtable_dev,dtype=np.uint32)

    totalPoints=calctotalpoints((gx,gy,gz),vtable)[0]
    print('totalPoints',totalPoints)
    Points=np.zeros((totalPoints,3),np.float32)
    if GPUBackend=='CUDA': #for this step, we use numba, not sure why the version of the CUDA kernel (quite simple in principle) is not working
        ExtractPoints(np.array((gx,gy,gz),np.int64),Points,vtable)
    else:
        step=240000000
        sizePdev=min((step,totalPoints))
        Points_part=np.zeros((sizePdev,3),np.float32)
        globalcount=np.zeros(2,np.uint64)
        if GPUBackend=='OpenCL':
            Points_dev=clp.Buffer(ctx, mf.WRITE_ONLY, Points_part.nbytes)
            globalcount_dev=clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=globalcount )
        elif GPUBackend=='Metal' :
            Points_dev=ctx.buffer(Points_part.nbytes)
            globalcount_dev=ctx.buffer(globalcount)
            inparams=np.zeros(3,np.uint32)
            
        totalGrid=gx*gy*gz
        prevPInd=0
        if GPUBackend=='CUDA':
            with ctx:
                SCode_p2='''
                    __constant__ float3 info_min={{ {xmin:10.9f},{ymin:10.9f},{zmin:10.9f} }};
                    __constant__ float3 info_max={{ {xmax:10.9f},{ymax:10.9f},{zmax:10.9f} }};
                    __constant__ uint3 info_gridsize={{ {gx},{gy},{gz} }};
                    __constant__ size_t info_n_triangles={n_triangles};
                    __constant__ float3 info_unit={{{dx:10.9f},{dy:10.9f},{dz:10.9f} }};
                '''.format(xmin=r[0,0],ymin=r[0,1],zmin=r[0,2],xmax=r[1,0],ymax=r[1,1],zmax=r[1,2],
                            n_triangles=n_triangles,gx=gx,gy=gy,gz=gz,dx=dxzy[0],dy=dxzy[1],dz=dxzy[2])

                prgcl  = clp.RawModule(code= "#define _CUDA\n"+_SCode_p1+SCode_p2+_SCode_p3,
                                    options=('-I',str(resource_path())))
                knl=prgcl.get_function("ExtractPoints")
                Points_dev=clp.zeros(Points_part.shape,clp.float32)
                globalcount_dev=clp.asarray(globalcount)
                

                vtable_dev=clp.asarray(vtable)
                for nt in range(0,totalGrid,step):
                    ntotal=min((totalGrid-nt,step))
            
                    Block=(64,1,1)
                    Grid=(int(ntotal//Block[0]+1),1,1)
                    knl(Grid,Block,(vtable_dev,globalcount_dev,Points_dev,
                                                    ntotal,
                                                    nt,
                                                    prevPInd,
                                                    gx,
                                                    gy,
                                                    gz))
                    Points_part=Points_dev.get()
                    globalcount=globalcount_dev.get()
                    Points[prevPInd:int(globalcount[0]),:]=Points_part[:int(globalcount[0])-prevPInd,:]
                    prevPInd=int(globalcount[0])
        else:
            for nt in range(0,totalGrid,step):
                ntotal=min((totalGrid-nt,step))
                if GPUBackend=='OpenCL':
                    prg.ExtractPoints(queue,[ntotal],None,vtable_dev,
                                                    globalcount_dev,
                                                    Points_dev,
                                                    np.uint32(ntotal),
                                                    np.uint32(nt),
                                                    np.uint32(prevPInd),
                                                    np.uint32(gx),
                                                    np.uint32(gy),
                                                    np.uint32(gz))
                    queue.finish()
                    clp.enqueue_copy(queue, Points_part,Points_dev)
                    queue.finish()
                    clp.enqueue_copy(queue, globalcount,globalcount_dev)
                    queue.finish()
                else:
                    inparams[0]=ntotal
                    inparams[1]=nt
                    inparams[2]=prevPInd
                    inparams_dev=ctx.buffer(inparams)
                    ctx.init_command_buffer()
                    handle = prg.function('ExtractPoints')(ntotal,vtable_dev,globalcount_dev,inparams_dev,Points_dev)
                    ctx.commit_command_buffer()
                    ctx.wait_command_buffer()
                    del handle
                    if 'arm64' not in platform.platform():
                        ctx.sync_buffers((Points_dev,globalcount_dev))
                    Points_part=np.frombuffer(Points_dev,dtype=np.float32).reshape(sizePdev,3)
                    globalcount=np.frombuffer(globalcount_dev,dtype=np.uint64)
            
                Points[prevPInd:int(globalcount[0]),:]=Points_part[:int(globalcount[0])-prevPInd,:]
                prevPInd=int(globalcount[0])
            
            
        print('globalcount',globalcount)
 
    Points[:,0]+=0.5
    Points[:,1]+=0.5
    Points[:,2]+=0.5
    Points[:,0]*=dxzy[0]
    Points[:,1]*=dxzy[1]
    Points[:,2]*=dxzy[2]
    Points[:,0]+=r[0,0]
    Points[:,1]+=r[0,1]
    Points[:,2]+=r[0,2]
    return Points