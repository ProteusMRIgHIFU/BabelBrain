import logging
import os
import re

from nibabel import processing, nifti1, affines
import numpy as np
import pytest

from BabelBrain.GPUFunctions.GPUVoxelize import Voxelize
from BabelBrain.BabelDatasetPreps import FixMesh

@pytest.mark.step1
def test_Voxelize_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_pyvista_plot,compare_data,request):
        
     # Parameters
     input_folder = dataset['folder_path']
     input_files = {
          'T1W': dataset['T1_path'],
          'skin': dataset['m2m_folder_path'] + os.sep + 'skin.stl',
     }
     spatial_step_text = re.sub("\.","_",str(spatial_step))
     output_fnames = {
          'Output_Truth': input_folder + f"T1W_cpu_voxelized_spatial_step_{spatial_step_text}.npy",
     }

     # Initialize GPU Backend
     gpu_device = get_gpu_device()
     Voxelize.InitVoxelize(gpu_device,computing_backend['type'])

     # Load inputs
     input_data = load_files(input_files)

     # Fix mesh if needed
     if input_data['skin'].body_count != 1:
          logging.info("Fixing mesh file")
          input_data['skin'] = FixMesh(input_data['skin'])    
          input_data['skin'].export(input_files['skin'])

     # Resample Data
     zooms = np.asarray(input_data['T1W'].header.get_zooms())
     new_zooms = np.full(3,spatial_step)
     logging.info(f"Original zooms: {zooms}")
     logging.info(f"New zooms: {new_zooms}")
     new_x_dim = int(input_data['T1W'].shape[0]/(new_zooms[0]/zooms[0]))
     new_y_dim = int(input_data['T1W'].shape[1]/(new_zooms[1]/zooms[1]))
     new_z_dim = int(input_data['T1W'].shape[2]/(new_zooms[2]/zooms[2]))
     new_affine = affines.rescale_affine(input_data['T1W'].affine.copy(),
                                             input_data['T1W'].shape,
                                             new_zooms,
                                             (new_x_dim,new_y_dim,new_z_dim))

     # Run voxelization step
     logging.info('Running Voxelization via GPU')
     points_gpu = Voxelize.Voxelize(input_data['skin'],targetResolution=spatial_step,GPUBackend=computing_backend['type'])
     try:
          logging.info('Reloading voxelization truth')
          points_cpu = load_files([output_fnames['Output_Truth']],skip_test=False)[0]
     except:
          logging.info("File doesn't exist")
          logging.info('Generating voxelization truth')
          points_cpu = input_data['skin'].voxelized(spatial_step,max_iter=30).fill().points
          logging.info('Saving file for future use')
          np.save(output_fnames['Output_Truth'],points_cpu)

     # Save plot screenshot to be added to html report later
     save_voxel_plot = get_pyvista_plot['voxel_plot']
     save_mesh_plot = get_pyvista_plot['mesh_plot']
     request.node.screenshots = []
     screenshot = save_mesh_plot(input_data['skin'], title="Skin Mesh")
     request.node.screenshots.append(screenshot)
     screenshot = save_voxel_plot(input_data['skin'],points_cpu, title="Skin Mesh Voxelized - Truth")
     request.node.screenshots.append(screenshot)
     screenshot = save_voxel_plot(input_data['skin'],points_gpu, title="Skin Mesh Voxelized - Test")
     request.node.screenshots.append(screenshot)
     
     # Convert voxels back to 3D indices
     points = np.hstack((points_gpu,np.ones((points_gpu.shape[0],1),dtype=points_gpu.dtype))).T
     points_truth = np.hstack((points_cpu,np.ones((points_cpu.shape[0],1),dtype=points_cpu.dtype))).T
     ijk = np.round(np.linalg.inv(new_affine).dot(points)).T
     ijk_truth = np.round(np.linalg.inv(new_affine).dot(points_truth)).T
     ijk = np.ascontiguousarray(ijk[:,:3])
     ijk_truth = np.ascontiguousarray(ijk_truth[:,:3])

     # Remove duplicates
     ijk_unique = np.unique(ijk, axis=0,)
     ijk_truth_unique = np.unique(ijk_truth, axis=0)
     logging.info(f"Number of GPU indexes: {ijk_unique.shape[0]}")
     logging.info(f"Number of CPU indexes: {ijk_truth_unique.shape[0]}")

     # Count number of matches
     set1 = set(map(tuple, ijk_unique))
     set2 = set(map(tuple, ijk_truth_unique))
     common_coordinates = set1.intersection(set2)
     match_count = len(common_coordinates)

     # Calculate dice coefficient
     dice_coeff = 2 * match_count / (ijk_unique.shape[0] + ijk_truth_unique.shape[0])
     logging.info(f"Dice coefficient: {dice_coeff}")
     
     assert dice_coeff > 0.95, f"Dice coefficient is less than 0.95 ({dice_coeff})"