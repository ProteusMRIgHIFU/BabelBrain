import os
import sys
sys.path.append('BabelBrain/GPUMedianFilter')
import logging

import pytest
import numpy as np
from scipy import ndimage

import MedianFilter

def test_median_filter(computing_backend,dataset,check_os,get_gpu_device,load_files,get_mpl_plot,compare_data,request):
        
     # Parameters
     input_folder = dataset['folder_path']

     # Initialize GPU Backend
     gpu_device = get_gpu_device()
     if computing_backend['type'] == 'OpenCL':
          MedianFilter.InitOpenCL(gpu_device)
     elif computing_backend['type'] == 'CUDA':
          MedianFilter.InitCUDA(gpu_device)
     elif computing_backend['type'] == 'Metal':
          MedianFilter.InitMetal(gpu_device)

     # Load inputs
     input_files = {
          'CT': input_folder + 'CT.nii.gz',
     }
     input_data = load_files(input_files)

     # Run median filter step
     CT_data = np.ascontiguousarray(input_data['CT'].get_fdata().astype(np.uint8))
     filtered_data = MedianFilter.MedianFilterSize7(CT_data,GPUBackend=computing_backend['type'])
     filtered_data_truth = ndimage.median_filter(CT_data,7)

     # Save plot screenshot to be added to html report later
     request.node.screenshots = []
     plots = [CT_data,filtered_data_truth,filtered_data]
     plot_names = ['Original Data', 'Truth Data', 'Filtered Data']
     screenshot = get_mpl_plot(plots, axes_num=1,titles=plot_names,color_map='gray')
     request.node.screenshots.append(screenshot)
     
     # Calculate dice coefficient
     calc_dice_func = compare_data['dice_coefficient']
     dice_coeff = calc_dice_func(filtered_data,filtered_data_truth)
     logging.info(f"Dice coefficient: {dice_coeff}")

     assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"