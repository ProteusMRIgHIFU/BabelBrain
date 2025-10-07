import logging
import re

import nibabel
from nibabel import processing, nifti1, affines
import numpy as np
import pytest

from BabelBrain.GPUFunctions.GPUResample import Resample

@pytest.mark.step1
def test_Resample_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_mpl_plot,compare_data,request):
        
    # Parameters
    input_folder = dataset['folder_path']
    order = 3
    file_type = 'T1W'
    input_files = {file_type: input_folder + 'T1W.nii.gz',}
    spatial_step_text = re.sub("\.","_",str(spatial_step))
    output_fnames = {
        'Output_Truth': input_folder + f"{file_type}_cpu_resampled_order_{order}_spatial_step_{spatial_step_text}.nii.gz",
    }

    # Initialize GPU Backend
    gpu_device = get_gpu_device()
    Resample.InitResample(gpu_device,computing_backend['type'])

    # Load inputs
    input_data = load_files(input_files)

    # Resample Data
    zooms = np.asarray(input_data[file_type].header.get_zooms())
    new_zooms = np.full(3,spatial_step)
    logging.info(f"Original zooms: {zooms}")
    logging.info(f"New zooms: {new_zooms}")
    new_x_dim = int(input_data[file_type].shape[0]/(new_zooms[0]/zooms[0]))
    new_y_dim = int(input_data[file_type].shape[1]/(new_zooms[1]/zooms[1]))
    new_z_dim = int(input_data[file_type].shape[2]/(new_zooms[2]/zooms[2]))
    new_affine = affines.rescale_affine(input_data[file_type].affine.copy(),
                                        input_data[file_type].shape,
                                        new_zooms,
                                        (new_x_dim,new_y_dim,new_z_dim))
    
    # Output dimensions
    output_data = np.zeros((new_x_dim,new_y_dim,new_z_dim),dtype=np.uint8)
    output_nifti = nifti1.Nifti1Image(output_data,new_affine)
    logging.info(f"Output Dimensions: {output_data.shape}")
    logging.info(f"Output Size: {output_data.size}")

    # Run resample step
    logging.info('Running resample step via GPU')
    nifti_resampled_gpu = Resample.ResampleFromTo(input_data[file_type],output_nifti,mode='constant',order=order,cval=input_data[file_type].get_fdata().min(),GPUBackend=computing_backend['type'])
    data_resampled_gpu = nifti_resampled_gpu.get_fdata()
    try:
        logging.info('Reloading resample truth')
        nifti_resampled_cpu = load_files([output_fnames['Output_Truth']],skip_test=False)[0]
        data_resampled_cpu = nifti_resampled_cpu.get_fdata()
    except:
        logging.info("File doesn't exist")
        logging.info('Generating resample truth')
        nifti_resampled_cpu = processing.resample_from_to(input_data[file_type],output_nifti,mode='constant',order=order,cval=input_data[file_type].get_fdata().min())
        data_resampled_cpu = nifti_resampled_cpu.get_fdata()
        logging.info('Saving file for future use')
        nibabel.save(nifti_resampled_cpu,output_fnames['Output_Truth'])

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [input_data[file_type].get_fdata(),data_resampled_cpu,data_resampled_gpu]
    plot_names = ['Original Data', 'CPU Resample\nOutput (Truth)', 'GPU Resample\nOutput']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map='gray')
    request.node.screenshots.append(screenshot)
    
    # Calculate Bhatt Coefficient
    calc_bc_func = compare_data['bhatt_coeff']
    bhatt_coeff = calc_bc_func(data_resampled_gpu,data_resampled_cpu)
    logging.info(f"Bhatt Coefficient: {bhatt_coeff}")

    assert bhatt_coeff == pytest.approx(1.0, rel=1e-9), f"Bhattacharyya Coefficient is not 1.0 ({bhatt_coeff})"