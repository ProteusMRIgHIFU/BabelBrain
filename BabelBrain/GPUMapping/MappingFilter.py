import numpy as np
import os

##probably not the most optimal of filters.. .but still WAY faster than CPU based
_code='''
#ifdef _OPENCL
__kernel void mapfilter(
                                    __global const  float * HUMap,
                                    __global const unsigned char * IsBone,
                                    __global const float * UniqueHU,
                                    __global unsigned int * CtMap,
                                    const int dimUnique,
                                    const int dims_0,
                                    const int dims_1,
                                    const int dims_2) {
      
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x > dims_0 || y > dims_1 || z > dims_2)
        return;
    const int _i = x*dims_1*dims_2 + y*dims_2 + z;
#endif
#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
kernel void mapfilter(
                                    const device  float * HUMap [[ buffer(0) ]],
                                    const device unsigned char * IsBone [[ buffer(1) ]],
                                    const device float * UniqueHU [[ buffer(2) ]],
                                    unsigned device int * CtMap [[ buffer(3) ]],
                                    uint _i[[thread_position_in_grid]]) {
      
 
#endif
    if (IsBone[_i] ==0)
        return; 
    const float selV = HUMap[_i];

    for (int iw_0 = 0; iw_0 < dimUnique; iw_0++)
    {
        if (selV == UniqueHU[iw_0])
        {
            CtMap[_i] = (unsigned int) iw_0;
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

def InitOpenCL(DeviceName='AMD'):
    import pyopencl as cl
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    
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
    print(ctx)
    
def MapFilter(HUMap,SelBone,UniqueHU,GPUBackend='OpenCL'):
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl
    # global knl_2px
    assert(HUMap.dtype==np.float32)
    assert(UniqueHU.dtype==np.float32)
    assert(SelBone.dtype==np.uint8)
    assert(np.all(np.array(HUMap.shape)==np.array(SelBone.shape)))
    assert(np.isfortran(HUMap)==False) 
    CtMap=np.zeros(HUMap.shape,np.uint32)
 
    if GPUBackend=='OpenCL':
        
        mf = cl.mem_flags

        HUMap_pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=HUMap)
        UniqueHU_pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=UniqueHU)
        SelBone_pr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SelBone)
        CtMap_pr = cl.Buffer(ctx, mf.WRITE_ONLY, CtMap.nbytes)

        knl(queue, HUMap.shape, 
                None,
                HUMap_pr,
                SelBone_pr,
                UniqueHU_pr,
                CtMap_pr,
                np.int32(len(UniqueHU)),
                np.int32(HUMap.shape[0]),
                np.int32(HUMap.shape[1]),
                np.int32(HUMap.shape[2]))

        cl.enqueue_copy(queue, CtMap,CtMap_pr)
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
        CtMap=np.frombuffer(CtMap_pr,dtype=CtMap.dtype).reshape(CtMap.shape)
       
    return CtMap