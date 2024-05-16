import os
import sys
sys.path.append('BabelBrain/GPUResample')
import logging

import pytest
from nibabel import processing, nifti1, affines
import numpy as np

import Resample

def test_resample(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_mpl_plot,compare_data,request):
        
        # Parameters
        input_folder = dataset['folder_path']

        # Initialize GPU Backend
        gpu_device = get_gpu_device()
        if computing_backend['type'] == 'OpenCL':
            Resample.InitOpenCL(gpu_device)
        elif computing_backend['type'] == 'CUDA':
            Resample.InitCUDA(gpu_device)
        elif computing_backend['type'] == 'Metal':
             Resample.InitMetal(gpu_device)

        # Load inputs
        input_files = {
             'T1W': input_folder + 'T1W.nii.gz',
        }
        input_data = load_files(input_files)

        # Calculate new affine
        T1W_affine_upscaled = affines.rescale_affine(input_data['T1W'].affine.copy(),input_data['T1W'].shape,spatial_step,(int(input_data['T1W'].shape[0]//spatial_step)+1,int(input_data['T1W'].shape[1]//spatial_step)+1,int(input_data['T1W'].shape[2]//spatial_step)+1))
        
        # output dimensions
        T1W_data = input_data['T1W'].get_fdata().astype(np.uint8)
        T1W_nifti = nifti1.Nifti1Image(T1W_data,input_data['T1W'].affine)
        output_data = np.zeros((int(T1W_data.shape[0]//spatial_step)+1,int(T1W_data.shape[1]//spatial_step)+1,int(T1W_data.shape[2]//spatial_step)+1),dtype=np.uint8)
        output_nifti = nifti1.Nifti1Image(output_data,T1W_affine_upscaled)

        # Run resample step
        T1W_resampled = Resample.ResampleFromTo(T1W_nifti,output_nifti,mode='constant',order=0,cval=T1W_data.min(),GPUBackend=computing_backend['type'])
        T1W_resampled_truth = processing.resample_from_to(T1W_nifti,output_nifti,mode='constant',order=0,cval=T1W_data.min()) # Truth method

        # Save plot screenshot to be added to html report later
        request.node.screenshots = []
        plots = [T1W_data,T1W_resampled_truth.get_fdata(),T1W_resampled.get_fdata()]
        plot_names = ['Original Data', 'Truth Data', 'Resampled Data']
        screenshot = get_mpl_plot(plots, axes_num=1,titles=plot_names,color_map='gray')
        request.node.screenshots.append(screenshot)
        
        # Calculate dice coefficient
        calc_dice_func = compare_data['dice_coefficient']
        dice_coeff = calc_dice_func(T1W_resampled.get_fdata(),T1W_resampled_truth.get_fdata())
        logging.info(f"Dice coefficient: {dice_coeff}")
     
        assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"