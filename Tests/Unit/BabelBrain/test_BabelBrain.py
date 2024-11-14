import os
import sys
import logging

import pytest
import nibabel
import numpy as np

from BabelBrain import BabelBrain

def test_save_T1W_iso(dataset,load_files,check_data,get_mpl_plot,compare_data,tmp_path,request):

    # Load inputs
    input_files = {'T1W': dataset['T1_path']}
    input_data = load_files(input_files)

    # Run save_T1W_iso function
    output_fname= dataset['T1_iso_path']
    BabelBrain.save_T1W_iso(input_files['T1W'], output_fname)

    # Load output
    T1W_iso = nibabel.load(output_fname)

    # Check that data is isometric
    isometric = check_data['isometric'](T1W_iso)
    if not isometric:
        pytest.fail("T1W data is not isometric")

    # Check origin has not changed
    same_origin = input_data['T1W'].affine[:3,3] == T1W_iso.affine[:3,3]
    same_origin = np.prod(same_origin)
    if not same_origin:
        pytest.fail("T1W data has changed origin")

    # Load data
    T1W_data = input_data['T1W'].get_fdata()
    T1W_iso_data = T1W_iso.get_fdata()
    
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [T1W_data, T1W_iso_data]
    plot_names = ['T1W data', 'T1W Iso Data']
    screenshot = get_mpl_plot(plots, axes_num=2,titles=plot_names,color_map='gray')
    request.node.screenshots.append(screenshot)

    # Calculate Bhattacharyya coefficient, output is between 0 and 1
    bhatt_coeff = compare_data['bhatt_coeff'](T1W_data, T1W_iso_data)

    assert bhatt_coeff > 0.99, f"Bhattacharyya Coefficient is less than 0.99 ({bhatt_coeff})"