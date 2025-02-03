import logging
import os
import re
import sys
sys.path.append('BabelBrain/GPUFunctions')
sys.path.append('BabelBrain/GPUFunctions/GPUResample')

import nibabel
from nibabel import processing, nifti1, affines
import numpy as np
import pytest

import Resample

def test_Resample(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_mpl_plot,compare_data,request):
        
    # Parameters
    input_folder = dataset['folder_path']
    order = 3
    input_files = {'CT': input_folder + 'CT.nii.gz',}
    spatial_step_text = re.sub("\.","_",str(spatial_step))
    output_fnames = {
        'Output_Truth': input_folder + f"CT_cpu_resampled_order_{order}_spatial_step_{spatial_step_text}.nii.gz",
    }

    # Initialize GPU Backend
    gpu_device = get_gpu_device()
    Resample.InitResample(gpu_device,computing_backend['type'])

    # Load inputs
    input_data = load_files(input_files)

    # Resample Data
    zooms = np.asarray(input_data['CT'].header.get_zooms())
    new_zooms = np.full(3,spatial_step)
    logging.info(f"Original zooms: {zooms}")
    logging.info(f"New zooms: {new_zooms}")
    new_x_dim = int(input_data['CT'].shape[0]/(new_zooms[0]/zooms[0]))
    new_y_dim = int(input_data['CT'].shape[1]/(new_zooms[1]/zooms[1]))
    new_z_dim = int(input_data['CT'].shape[2]/(new_zooms[2]/zooms[2]))
    new_affine = affines.rescale_affine(input_data['CT'].affine.copy(),
                                        input_data['CT'].shape,
                                        new_zooms,
                                        (new_x_dim,new_y_dim,new_z_dim))
    
    # Output dimensions
    output_data = np.zeros((new_x_dim,new_y_dim,new_z_dim),dtype=np.uint8)
    output_nifti = nifti1.Nifti1Image(output_data,new_affine)
    logging.info(f"Output Dimensions: {output_data.shape}")
    logging.info(f"Output Size: {output_data.size}")

    # Run resample step
    logging.info('Running resample step via GPU')
    nifti_resampled_gpu = Resample.ResampleFromTo(input_data['CT'],output_nifti,mode='constant',order=order,cval=input_data['CT'].get_fdata().min(),GPUBackend=computing_backend['type'])
    data_resampled_gpu = nifti_resampled_gpu.get_fdata()
    try:
        logging.info('Reloading resample truth')
        nifti_resampled_cpu = load_files([output_fnames['Output_Truth']],skip_test=False)[0]
        data_resampled_cpu = nifti_resampled_cpu.get_fdata()
    except:
        logging.info("File doesn't exist")
        logging.info('Generating resample truth')
        nifti_resampled_cpu = processing.resample_from_to(input_data['CT'],output_nifti,mode='constant',order=order,cval=input_data['CT'].get_fdata().min())
        data_resampled_cpu = nifti_resampled_cpu.get_fdata()
        logging.info('Saving file for future use')
        nibabel.save(nifti_resampled_cpu,output_fnames['Output_Truth'])

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [input_data['CT'].get_fdata(),data_resampled_cpu,data_resampled_gpu]
    plot_names = ['Original Data', 'CPU Resample\nOutput (Truth)', 'GPU Resample\nOutput']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map='gray')
    request.node.screenshots.append(screenshot)
    
    # Calculate Bhatt Distance
    calc_bd_func = compare_data['bhatt_distance']
    bhatt_dist = calc_bd_func(data_resampled_gpu,data_resampled_cpu,256)
    logging.info(f"Bhatt Distance: {bhatt_dist}")

    assert bhatt_dist < 0.01, f"Bhatt Distance is grater than 0.01 ({bhatt_dist})"