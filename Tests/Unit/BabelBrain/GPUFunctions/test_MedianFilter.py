import logging
import re

import nibabel
import numpy as np
import pytest
from scipy import ndimage

from BabelBrain.GPUFunctions.GPUMedianFilter import MedianFilter

@pytest.mark.step1
def test_MedianFilter_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_resampled_input,get_mpl_plot,compare_data,request):
        
     # Parameters
     input_folder = dataset['folder_path']
     input_files = {'CT': input_folder + 'CT.nii.gz',}
     spatial_step_text = re.sub("\.","_",str(spatial_step))
     output_fnames = {
          'Resampled_Input': input_folder + f"CT_cpu_resampled_spatial_step_{spatial_step_text}.nii.gz",
          'Output_Truth': input_folder + f"CT_cpu_median_filter_spatial_step_{spatial_step_text}.nii.gz"
     }

     # Initialize GPU Backend
     gpu_device = get_gpu_device()
     MedianFilter.InitMedianFilter(gpu_device,computing_backend['type'])

     # Load inputs
     input_data = load_files(input_files)

     # Get resampled input for specified spatial step
     resampled_nifti, resampled_data = get_resampled_input(input_data['CT'],spatial_step,output_fnames['Resampled_Input'])

     # Determine median filter size
     filter_size = np.round((np.ones(3)*2)/resampled_nifti.header.get_zooms()).astype(int)
     logging.info(f"Median Filter Size: {filter_size}")
     
     if np.any(filter_size[filter_size > 7]):
          logging.warning(f"Median filter was capped at 7")
          filter_size = 7 

     # Run median filter step
     CT_data = np.ascontiguousarray(input_data['CT'].get_fdata().astype(np.uint8))
     logging.info('Running Median Filter via GPU')
     data_median_filter_gpu = MedianFilter.MedianFilter(CT_data,filter_size,GPUBackend=computing_backend['type'])
     try:
          logging.info('Reloading median filter truth')
          nifti_median_filter_cpu = load_files([output_fnames['Output_Truth']],skip_test=False)[0]
          data_median_filter_cpu = nifti_median_filter_cpu.get_fdata()
     except:
          logging.info("File doesn't exist")
          logging.info('Generating median filter truth')
          data_median_filter_cpu = ndimage.median_filter(CT_data,filter_size)
          nifti_median_filter_cpu = nibabel.Nifti1Image(data_median_filter_cpu,resampled_nifti.affine)
          logging.info('Saving file for future use')
          nibabel.save(nifti_median_filter_cpu,output_fnames['Output_Truth'])

     # Save plot screenshot to be added to html report later
     request.node.screenshots = []
     plots = [resampled_data,data_median_filter_cpu,data_median_filter_gpu]
     plot_names = ['Resampled Data','CPU Median Filter\nOutput (Truth)', 'GPU Median\n Filter Output']
     screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map='gray')
     request.node.screenshots.append(screenshot)
     
     # Calculate dice coefficient
     calc_dice_func = compare_data['dice_coefficient']
     dice_coeff = calc_dice_func(data_median_filter_gpu,data_median_filter_cpu)
     logging.info(f"Dice coefficient: {dice_coeff}")

     assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"