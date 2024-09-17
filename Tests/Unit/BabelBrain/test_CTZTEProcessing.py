import os
import sys
import logging

import pytest
import nibabel
import numpy as np
import SimpleITK as sitk

from BabelBrain import CTZTEProcessing

def test_N4BiasCorrec(dataset,load_files,get_mpl_plot,compare_data,tmp_path,request):

    # # Load inputs
    input_files = {'T1W': dataset['folder_path'] + 'T1W-isotropic.nii.gz'}
    input_data = load_files(input_files)

    # Load truth data
    truth_files = {'T1W_bias_correc': dataset['folder_path'] + 'Truth' + os.sep + 'Unit' + os.sep + 'T1W-isotropic_BiasCorrec.nii.gz'}
    truth_data = load_files(truth_files)

    # Run save_T1W_iso function
    output_fname= str(tmp_path) + os.sep + 'T1W_BiasCorrec.nii.gz'
    CTZTEProcessing.N4BiasCorrec(input_files['T1W'], hashFiles=[],output=output_fname)

    # Load output
    T1W_bias_correc = nibabel.load(output_fname)

    # Load data
    T1W_data = input_data['T1W'].get_fdata()
    T1W_bias_correc_data = T1W_bias_correc.get_fdata()
    T1W_bias_correc_data_truth = truth_data['T1W_bias_correc'].get_fdata()
    
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [T1W_data, T1W_bias_correc_data, T1W_bias_correc_data_truth]
    plot_names = ['T1W', 'T1W Bias Field Corrected', 'T1W Bias Field\nCorrected Truth']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map='gray')
    request.node.screenshots.append(screenshot)

    dice_coeff = compare_data['dice_coefficient'](T1W_bias_correc_data, T1W_bias_correc_data_truth)

    assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"