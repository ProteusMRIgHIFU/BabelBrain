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
    uint gid = thread_position_in_grid.x;
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