import os
import sys
sys.path.append('BabelBrain/GPUVoxelize')
import logging

import pytest
import numpy as np
from nibabel import affines

import Voxelize
from BabelDatasetPreps import FixMesh

def test_voxelize(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_pyvista_plot,compare_data,request):
        
     # Parameters
     input_folder = dataset['folder_path']

     # Initialize GPU Backend
     gpu_device = get_gpu_device()
     if computing_backend['type'] == 'OpenCL':
          Voxelize.InitOpenCL(gpu_device)
     elif computing_backend['type'] == 'CUDA':
          Voxelize.InitCUDA(gpu_device)
     elif computing_backend['type'] == 'Metal':
          Voxelize.InitMetal(gpu_device)

     # Load inputs
     input_files = {
          'T1W': input_folder + 'T1W.nii.gz',
          'skin': input_folder + f"m2m_{dataset['id']}" + os.sep + 'skin.stl',
     }
     input_data = load_files(input_files)

     # Fix mesh if needed
     if input_data['skin'].body_count != 1:
          input_data['skin'] = FixMesh(input_data['skin'])

     # Run voxelization step
     points = Voxelize.Voxelize(input_data['skin'],targetResolution=spatial_step,GPUBackend=computing_backend['type'])
     points_truth = input_data['skin'].voxelized(spatial_step,max_iter=30).fill().points

     # Save plot screenshot to be added to html report later
     save_voxel_plot = get_pyvista_plot['voxel_plot']
     save_mesh_plot = get_pyvista_plot['mesh_plot']
     request.node.screenshots = []
     screenshot = save_mesh_plot(input_data['skin'], title="Skin Mesh")
     request.node.screenshots.append(screenshot)
     screenshot = save_voxel_plot(input_data['skin'],points_truth, title="Skin Mesh Voxelized - Truth")
     request.node.screenshots.append(screenshot)
     screenshot = save_voxel_plot(input_data['skin'],points, title="Skin Mesh Voxelized - Test")
     request.node.screenshots.append(screenshot)
     
     # Calculate new affine
     T1W_affine_upscaled = affines.rescale_affine(input_data['T1W'].affine.copy(),input_data['T1W'].shape,spatial_step,(int(input_data['T1W'].shape[0]//spatial_step)+1,int(input_data['T1W'].shape[1]//spatial_step)+1,int(input_data['T1W'].shape[2]//spatial_step)+1))

     # Convert voxels back to 3D indices
     points = np.hstack((points,np.ones((points.shape[0],1),dtype=points.dtype))).T
     points_truth = np.hstack((points_truth,np.ones((points_truth.shape[0],1),dtype=points_truth.dtype))).T
     ijk = np.round(np.linalg.inv(T1W_affine_upscaled).dot(points)).T
     ijk_truth = np.round(np.linalg.inv(T1W_affine_upscaled).dot(points_truth)).T

     # Count number of matches
     matches = np.isin(ijk,ijk_truth)
     matches = np.prod(matches,axis=1)
     match_count = len(matches[matches==True])

     # Calculate dice coefficient
     dice_coeff = 2 * match_count / (ijk.shape[0] + ijk_truth.shape[0])
     logging.info(f"Dice coefficient: {dice_coeff}")
     
     assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"