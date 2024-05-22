import numpy as np
import os
import platform

##probably not the most optimal of filters.. .but still WAY faster than CPU based
_code='''
#ifdef _OPENCL
__kernel void mapfilter(
                                    __global const  float * HUMap,
                                    __global const unsigned char * IsBone,
                                    __global const float * UniqueHU,
                                    __global unsigned int * CtMap,
                                    const unsigned int dimUnique,
                                    const unsigned int dims_0,
                                    const unsigned int dims_1,
                                    const unsigned int dims_2) {
      
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);

    if (x > dims_0 || y > dims_1 || z > dims_2)
        return;
    const unsigned int ind = x*dims_1*dims_2 + y*dims_2 + z;
#endif
#ifdef _CUDA
extern "C" __global__ void mapfilter(  const float * HUMap,
                                      const unsigned char * IsBone,
                                     const float * UniqueHU,
                                     unsigned int * CtMap,
                                    const unsigned int dimUnique,
                                    const unsigned int dims_0,
                                    const unsigned int dims_1,
                                    const unsigned int dims_2) {
      
    const size_t x =  (size_t)(blockIdx.x*blockDim.x + threadIdx.x);
    const size_t y =  (size_t)(blockIdx.y*blockDim.y + threadIdx.y);
    const size_t z =  (size_t)(blockIdx.z*blockDim.z + threadIdx.z);

    if (x > ((size_t)dims_0) || y > ((size_t)dims_1) || z > ((size_t)dims_2))
        return;
    const  size_t ind = x*((size_t)dims_1)*((size_t)dims_2) + y*((size_t)dims_2) + z;
#endif

#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
kernel void mapfilter(
                                    const device  float * HUMap [[ buffer(0) ]],
                                    const device unsigned char * IsBone [[ buffer(1) ]],
                                    const device float * UniqueHU [[ buffer(2) ]],
                                    unsigned device int * CtMap [[ buffer(3) ]],
                                    uint ind[[thread_position_in_grid]]) {
      
 
#endif
    if (IsBone[ind] ==0)
        return; 
    const float selV = HUMap[ind];

    for (unsigned int iw_0 = 0; iw_0 < dimUnique; iw_0++)
    {
        if (selV == UniqueHU[iw_0])
        {
           
            CtMap[ind] =  iw_0;
            break;
            
        }
    }

}

'''
Platforms=None
queue = None
prgcl = None
ctx = None
knl = None
clp= None
ocl=None

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
            break

    if selDevice is None:
        raise SystemError("There are no devices supporting CUDA or that matches selected device.")
      
    ctx=selDevice
    clp=cp


def InitOpenCL(DeviceName='AMD'):
    import pyopencl as cl
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    global ocl
    
    ocl = cl

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
    prgcl = cl.Program(ctx, "#define _OPENCL\n"+_code).build()
    knl=prgcl.mapfilter

def InitMetal(DeviceName='AMD'):
    global ctx
    global prgcl
    global knl
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
    
def MapFilter(HUMap,SelBone,UniqueHU,GPUBackend='OpenCL'):
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    global clp
    global ocl
    # global knl_2px
    
    assert(HUMap.dtype==np.float32)
    assert(UniqueHU.dtype==np.float32)
    assert(SelBone.dtype==np.uint8)
    assert(np.all(np.array(HUMap.shape)==np.array(SelBone.shape)))
    assert(np.isfortran(HUMap)==False) 
    CtMap=np.zeros(HUMap.shape,np.uint32)
 
    if GPUBackend=='CUDA':
        with ctx:
            prgcl  = clp.RawModule(code= "#define _CUDA\n"+_code)
            knl=prgcl.get_function("mapfilter")

            HUMap_pr=clp.asarray(HUMap)
            UniqueHU_pr=clp.asarray(UniqueHU)
            SelBone_pr=clp.asarray(SelBone)
            CtMap_pr=clp.zeros(CtMap.shape,CtMap.dtype)
            Block=(4,4,4)
            Grid=(HUMap.shape[0]//Block[0]+1,HUMap.shape[1]//Block[1]+1,HUMap.shape[2]//Block[2]+1)
            knl(Grid,Block,(HUMap_pr,
                    SelBone_pr,
                    UniqueHU_pr,
                    CtMap_pr,
                    len(UniqueHU),
                    HUMap.shape[0],
                    HUMap.shape[1],
                    HUMap.shape[2]))
            
            CtMap=CtMap_pr.get()
    elif GPUBackend=='OpenCL':
        
        mf = ocl.mem_flags

        HUMap_pr = ocl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=HUMap)
        UniqueHU_pr = ocl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=UniqueHU)
        SelBone_pr = ocl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SelBone)
        CtMap_pr = ocl.Buffer(ctx, mf.WRITE_ONLY, CtMap.nbytes)

        knl(queue, HUMap.shape, 
                None,
                HUMap_pr,
                SelBone_pr,
                UniqueHU_pr,
                CtMap_pr,
                np.uint32(len(UniqueHU)),
                np.uint32(HUMap.shape[0]),
                np.uint32(HUMap.shape[1]),
                np.uint32(HUMap.shape[2]))

        ocl.enqueue_copy(queue, CtMap,CtMap_pr)
    else:
        prgcl = ctx.kernel("#define _METAL\n#define dimUnique %i\n" %(len(UniqueHU))+_code)
        knl=prgcl.function('mapfilter')
        HUMap_pr = ctx.buffer(HUMap) 
        UniqueHU_pr = ctx.buffer(UniqueHU)
        SelBone_pr =  ctx.buffer(SelBone)
        CtMap_pr=  ctx.buffer(CtMap.nbytes)
        ctx.init_command_buffer()
        handle=knl(int(np.prod(HUMap.shape)),HUMap_pr,SelBone_pr,UniqueHU_pr,CtMap_pr )
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((CtMap_pr,UniqueHU_pr))
        CtMap=np.frombuffer(CtMap_pr,dtype=CtMap.dtype).reshape(CtMap.shape)
  
    return CtMap

if __name__ == "__main__":
    from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
    t=ReadFromH5py('test.h5')
    InitCUDA('A6000')
    MapFilter(t['HUMap'],t['SelBone'],t['UniqueHU'],GPUBackend='CUDA')