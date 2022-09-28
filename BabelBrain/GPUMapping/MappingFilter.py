import numpy as np
import pyopencl as cl
import os

##probably not the most optimal of filters.. .but still WAY faster than CPU based
_code='''

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
    prgcl = cl.Program(ctx, _code).build()
    knl=prgcl.mapfilter

def MapFilter(HUMap,SelBone,UniqueHU):
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
       
    return CtMap