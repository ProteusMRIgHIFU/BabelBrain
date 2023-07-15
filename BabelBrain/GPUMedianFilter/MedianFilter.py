import numpy as np
import platform

import os


##probably not the most optimal of filters.. .but still WAY faster than CPU based
_code='''
#ifdef _OPENCL
__kernel void median_reflect_w7_7_7(
                                    __global const  PixelType * input,
                                    __global PixelType * output, 
                                    const int dims_0,
                                    const int dims_1,
                                    const int dims_2) {
      
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    int _i = x*dims_1*dims_2 + y*dims_2 + z;
#endif
#ifdef _METAL
#include <metal_stdlib>
using namespace metal;
//dims_0 //To be defined when compiling kernel
//dims_1  //To be defined when compiling kernel
//dims_2  //To be defined when compiling kernel
kernel void median_reflect_w7_7_7(
                                    const device PixelType * input [[ buffer(0) ]],
                                    device PixelType * output [[ buffer(1) ]],
                                    uint gid[[thread_position_in_grid]]) {
    const int x = gid/(dims_1*dims_2);
    const int y = (gid-x*dims_1*dims_2)/dims_2;
    const int z = gid -x*dims_1*dims_2 - y * dims_2;
    #define _i gid
#endif
    
    
    int ind_2 = z - 3;
    int ind_1 = y - 3;
    int ind_0 = x - 3;
    
    int iv = 0;
    PixelType values[343];
    
    for (int iw_0 = 0; iw_0 < 7; iw_0++)
    {
        int ix_0 = ind_0 + iw_0;

        if (ix_0 < 0) 
            ix_0 = - 1 -ix_0;
        else
            ix_0 = min(ix_0, 2 * dims_0 - 1 - ix_0);

        for (int iw_1 = 0; iw_1 < 7; iw_1++)
        {
            int ix_1 = ind_1 + iw_1;

            if (ix_1 < 0) 
                ix_1 = - 1 -ix_1;
            else
                ix_1 = min(ix_1, 2 * dims_1 - 1 - ix_1);
                
            for (int iw_2 = 0; iw_2 < 7; iw_2++)
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
     
    //sorting
    const int size = 343;
    int gap = 364;
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
    
    output[_i]=values[171];

    }

'''
Platforms=None
queue = None
prgcl = None
ctx = None
knl_1px = None
mf = None
clp = None

def InitOpenCL(DeviceName='AMD'):
    import pyopencl as cl
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl_1px
    global mf
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
    Preamble ='#define _OPENCL\ntypedef unsigned char PixelType;\n'
    prgcl = cl.Program(ctx, Preamble+_code).build()
    knl_1px=prgcl.median_reflect_w7_7_7
    mf = cl.mem_flags
    
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
    
    
def MedianFilterSize7(data,GPUBackend='OpenCL'):
    global Platforms
    global queue 
    global prgcl 
    global ctx
    global knl_1px
    global mf
    global clp
    assert(data.dtype==np.uint8)
    assert(np.isfortran(data)==False) 

    if GPUBackend=='OpenCL':
        
        data_pr = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        data_out = np.zeros_like(data)
        data_out_pr = clp.Buffer(ctx, mf.WRITE_ONLY, data_out.nbytes)

        knl_1px(queue, data.shape, 
                None,
                data_pr,
                data_out_pr,
                np.int32(data.shape[0]),
                np.int32(data.shape[1]),
                np.int32(data.shape[2]),
                g_times_l=False)

        clp.enqueue_copy(queue, data_out,data_out_pr)
    else:
        assert(GPUBackend=='Metal')
        Preamble ='#define _METAL\ntypedef unsigned char PixelType;\n'
        Preamble+='#define dims_0 %i \n' %(data.shape[0])
        Preamble+='#define dims_1 %i \n' %(data.shape[1])
        Preamble+='#define dims_2 %i \n' %(data.shape[2])
        prgcl = ctx.kernel(Preamble+_code)
        knl_1px=prgcl.function('median_reflect_w7_7_7')
        data_pr = ctx.buffer(data) 
        data_out_pr = ctx.buffer(data.nbytes)
        ctx.init_command_buffer()
        handle=knl_1px(int(np.prod(data.shape)),data_pr,data_out_pr)
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((data_out_pr,data_pr))
        data_out=np.frombuffer(data_out_pr,dtype=np.uint8).reshape(data.shape)
    return data_out