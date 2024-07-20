import logging
logger = logging.getLogger()
import os
from pathlib import Path
import platform
import sys

from numba import jit,njit, prange
import numpy as np

try:
    from GPUUtils import InitCUDA,InitOpenCL,InitMetal,get_step_size
except:
    from ..GPUUtils import InitCUDA,InitOpenCL,InitMetal,get_step_size

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

def InitVoxelize(DeviceName='A6000',GPUBackend='OpenCL'):
    global queue 
    global prgcl
    global kernel_code
    global sel_device
    global ctx
    global knl
    global mf
    global clp

    kernel_files = [os.path.join(resource_path(),'voxelize.cpp')]

    if GPUBackend == 'CUDA':
        import cupy as cp
        clp = cp

        ctx,kernel_code,sel_device = InitCUDA(kernel_files=kernel_files,DeviceName=DeviceName,build_later=True)

    elif GPUBackend == 'OpenCL':
        import pyopencl as pocl
        clp = pocl

        queue,kernel_code,sel_device,ctx,mf = InitOpenCL(kernel_files=kernel_files,DeviceName=DeviceName,build_later=True)

    elif GPUBackend == 'Metal':
        import metalcomputebabel as mc
        clp = mc

        kernel_code,sel_device,ctx = InitMetal(kernel_files=kernel_files,DeviceName=DeviceName,build_later=True)


def Voxelize(inputMesh,targetResolution=1333/500e3/6*0.75*1e3,GPUBackend='OpenCL'):
    global ctx
    global clp
    global queue
    
    logger.info(f"\nStarting Voxelization")

    # Determine number of mesh faces
    if np.isfortran(inputMesh.triangles):
        triangles=np.ascontiguousarray(inputMesh.triangles).astype(np.float32)
    else:
        triangles=inputMesh.triangles.astype(np.float32)
    n_triangles=inputMesh.triangles.shape[0]
    print('GPU Voxelizing # triangles', n_triangles)

    # Calculate mesh bounding box dimensions, number of grid points, and grid spacing
    r=inputMesh.bounding_box.bounds
    dims=np.diff(r,axis=0).flatten()
    gx=int(np.ceil(dims[0]/targetResolution))
    gy=int(np.ceil(dims[1]/targetResolution))
    gz=int(np.ceil(dims[2]/targetResolution))
    dxzy=dims/np.array([gx,gy,gz])
    print('spatial step and  maximal grid dimensions',dxzy,gx,gy,gz)

    # Create voxel table accounting for dtype
    vtable_size = int(np.ceil((gx*gy*gz) / 32.0) * 4)
    vtable=np.zeros(vtable_size,np.uint8)
    
    constant_defs='''
        constant float3 info_min = {{ {xmin:10.9f},{ymin:10.9f},{zmin:10.9f} }};
        constant float3 info_max = {{ {xmax:10.9f},{ymax:10.9f},{zmax:10.9f} }};
        constant uint3  info_gridsize = {{ {gx},{gy},{gz} }};
        constant size_t info_n_triangles = {n_triangles};
        constant float3 info_unit = {{{dx:10.9f},{dy:10.9f},{dz:10.9f} }};
    '''.format(xmin=r[0,0],ymin=r[0,1],zmin=r[0,2],
               xmax=r[1,0],ymax=r[1,1],zmax=r[1,2],
               gx=gx,gy=gy,gz=gz,
               n_triangles=n_triangles,
               dx=dxzy[0],dy=dxzy[1],dz=dxzy[2])
    
    if GPUBackend=='CUDA':
        with ctx:
            constant_defs = constant_defs.replace('constant','__constant__')
            
            # Windows sometimes has issues finding CUDA
            if platform.system()=='Windows':
                sys.executable.split('\\')[:-1]
                options=('-I',os.path.join(os.getenv('CUDA_PATH'),'Library','Include'),
                         '-I',str(resource_path()))
            else:
                options=('-I',str(resource_path()))
            
            # Build program from source code
            prgcl = clp.RawModule(code= "#define _CUDA\n" + constant_defs + kernel_code,
                                 options=options)
        
            # Create kernel from program function
            knl = prgcl.get_function("voxelize_triangle_solid")

            # Move input data from host to device memory
            vtable_gpu=clp.zeros(vtable.shape,clp.uint8)
            triangles_gpu=clp.asarray(triangles)

            # Deploy kernel
            Block=(64,1,1)
            Grid=(int(n_triangles//Block[0]+1),1,1)
            knl(Grid,Block,(triangles_gpu,vtable_gpu))

            # Move kernel output data back to host memory
            vtable=vtable_gpu.get()
            vtable=np.frombuffer(vtable,np.uint32)

    elif GPUBackend=='OpenCL':
        constant_defs = constant_defs.replace('constant','__constant')

        # Build program from source code
        mf=clp.mem_flags
        prg=clp.Program(ctx,"#define _OPENCL\n"+constant_defs+kernel_code).build()
        
        # Move input data from host to device memory
        vtable_gpu=clp.Buffer(ctx, mf.READ_WRITE, vtable.nbytes)
        triangles_gpu=clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=triangles)
        
        # Deploy kernel
        prg.voxelize_triangle_solid(queue,[n_triangles],None,triangles_gpu,vtable_gpu)
        queue.finish()
        
        # Move kernel output data back to host memory
        clp.enqueue_copy(queue, vtable,vtable_gpu)
        queue.finish()
        vtable=np.frombuffer(vtable,np.uint32)
    else:
        assert(GPUBackend=='Metal')

        metal_def='''
        #define _METAL
        #define gx {gx}
        #define gy {gy}
        #define gz {gz}
        '''.format(gx=gx,gy=gy,gz=gz)

        # Build program from source code
        prg = ctx.kernel(metal_def+constant_defs+kernel_code)

        # Move input data from host to device memory
        vtable_gpu=ctx.buffer(vtable.nbytes)
        triangles_gpu=ctx.buffer(triangles)
        ctx.init_command_buffer()

        # Deploy kernel
        handle = prg.function('voxelize_triangle_solid')(n_triangles,triangles_gpu,vtable_gpu)
        ctx.commit_command_buffer()
        ctx.wait_command_buffer()
        del handle

        # Move kernel output data back to host memory
        if 'arm64' not in platform.platform():
            ctx.sync_buffers((vtable_gpu,triangles_gpu))
        vtable=np.frombuffer(vtable_gpu,dtype=np.uint32)

    # Create points array 
    totalPoints=calctotalpoints((gx,gy,gz),vtable)[0]
    print('totalPoints',totalPoints)
    Points=np.zeros((totalPoints,3),np.float32)

    # Extract points
    totalGrid=gx*gy*gz
    logger.info(f"TotalGrid: {totalGrid}")
    step = get_step_size(sel_device,num_large_buffers=2,data_type=Points.dtype,GPUBackend=GPUBackend)
    points_section_size = min(step,totalPoints)
    globalcount=np.zeros(2,np.uint32)
    int_params = np.zeros(4,np.uint32)
    prev_start_ind = 0
    for point in range(0,totalGrid,step):
        ntotal = min((totalGrid-point),step)
        logger.info(f"\nWorking on points {point} to {point+ntotal} out of {totalGrid}")

        # Grab sections of data
        points_section = np.zeros((points_section_size,3),np.float32)

        # Since we run into issues sending numbers larger than 32 bits due to buffer size restrictions, 
        # we check the size here, send info to kernel, and create number there as workaround
        current_position = point
        base_32 = current_position // (2**32)
        current_position = current_position - (base_32 * (2**32))

        int_params[0]=ntotal
        int_params[1]=current_position
        int_params[2]=base_32
        int_params[3]=prev_start_ind

        if GPUBackend=='CUDA':
            with ctx:
                # Move input data from host to device memory
                points_section_gpu = clp.asarray(points_section)
                globalcount_gpu = clp.asarray(globalcount)
                int_params_gpu = clp.asarray(int_params)

                # Define block and grid sizes
                block_size = (64,1,1)
                grid_size=(int(ntotal//Block[0]+1),1,1)
                
                # Deploy kernel
                prgcl.get_function("ExtractPoints")(grid_size,block_size,
                                                    (vtable_gpu,
                                                    globalcount_gpu,
                                                    points_section_gpu,
                                                    int_params_gpu,
                                                    np.uint32(gx),
                                                    np.uint32(gy),
                                                    np.uint32(gz),
                                                    np.uint32(points_section.size))
                                                    )

                # Move kernel output data back to host memory
                points_section=points_section_gpu.get()
                globalcount=globalcount_gpu.get()

        elif GPUBackend=='OpenCL':
            # Move input data from host to device memory
            points_section_gpu=clp.Buffer(ctx, mf.WRITE_ONLY, points_section.nbytes)
            globalcount_gpu=clp.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=globalcount )
            int_params_gpu = clp.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int_params)
        
            # Deploy kernel
            prg.ExtractPoints(queue,[ntotal],None,vtable_gpu,
                                                globalcount_gpu,
                                                points_section_gpu,
                                                int_params_gpu,
                                                np.uint32(gx),
                                                np.uint32(gy),
                                                np.uint32(gz))
            queue.finish()

            # Move kernel output data back to host memory
            clp.enqueue_copy(queue, points_section,points_section_gpu)
            queue.finish()
            clp.enqueue_copy(queue, globalcount,globalcount_gpu)
            queue.finish()

        elif GPUBackend=='Metal' :
            # Move input data from host to device memory
            points_section_gpu=ctx.buffer(points_section.nbytes)
            globalcount_gpu=ctx.buffer(globalcount)
            int_params_gpu = ctx.buffer(int_params)

            # Deploy kernel
            ctx.init_command_buffer()
            handle = prg.function('ExtractPoints')(ntotal,vtable_gpu,globalcount_gpu,int_params_gpu,points_section_gpu)
            ctx.commit_command_buffer()
            ctx.wait_command_buffer()
            del handle
            if 'arm64' not in platform.platform():
                ctx.sync_buffers((points_section_gpu,globalcount_gpu))

            # Move kernel output data back to host memory
            points_section=np.frombuffer(points_section_gpu,dtype=np.float32).reshape(points_section.shape)
            globalcount=np.frombuffer(globalcount_gpu,dtype=np.uint32)
            logger.info(f"globalcount: {globalcount}")

        try:
            Points[prev_start_ind:int(globalcount[0]),:]=points_section[:int(globalcount[0])-prev_start_ind,:]
        except Exception as e:
            print(e)
            raise ValueError("Error running voxelization for specified parameters, suggest lowering PPW")
        prev_start_ind=int(globalcount[0])
        
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