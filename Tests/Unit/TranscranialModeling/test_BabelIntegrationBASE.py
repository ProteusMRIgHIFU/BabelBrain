import os
import sys
import logging

import pytest
import nibabel

from TranscranialModeling import BabelIntegrationBASE as BIBase

def test_SaveNiftiEnforcedISO(dataset,load_files,check_data,tmp_path):

    # Load inputs
    input_files = {'T1W': dataset['T1_path']}
    input_data = load_files(input_files)

    # Run SaveNiftiEnforcedISO
    output_fname_initial = str(tmp_path) + os.sep + 'T1W-isotropic__.nii.gz'
    output_fname_final = str(tmp_path) + os.sep + 'T1W-isotropic.nii.gz'
    BIBase.SaveNiftiEnforcedISO(input_data['T1W'], output_fname_initial)

    # Check intermediate files are deleted
    intermediate_files = [output_fname_initial.split('.gz')[0], output_fname_final.split('.gz')[0]]
    for fname in intermediate_files:
        if os.path.exists(fname):
            logging.warning(f"temporary intermediate file ({fname}) was not deleted")

    # Load output
    enforced_iso_nifti = nibabel.load(output_fname_final)

    # Check that data is isometric
    isometric = check_data['isometric'](enforced_iso_nifti)

    assert isometric, "Data is not isometric"