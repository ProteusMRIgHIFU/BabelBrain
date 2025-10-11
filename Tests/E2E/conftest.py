import glob
import logging
import os
from pathlib import Path
import re

import pytest

@pytest.fixture
def compare_BabelBrain_Outputs(compare_data):
    def _compare_BabelBrain_Outputs(ref_folder,test_folder,tolerance,node_screenshots):
        
        # Find h5 files in folders
        pattern = "**/*ThermalField_AllCombinations.h5"

        h5_refs = glob.glob(os.path.join(ref_folder, pattern),recursive=True)
        h5_tests = glob.glob(os.path.join(test_folder, pattern),recursive=True)
        
        # Check for presence of files
        if len(h5_refs) == 0:
            pytest.skip(f"Files not found in {ref_folder}")
        if len(h5_tests) == 0:
            pytest.skip(f"Files not found in {test_folder}")
        
        def grab_folder_file_name(file, no_gpu=False):
            base_file_path = os.path.join(*file.split(os.sep)[-2:])
            if no_gpu:
                base_file_path = re.sub("(CUDA|OpenCL|Metal|MLX)","",base_file_path)
            
            return base_file_path
        
        # Build lookup for reference files
        ref_lookup = {grab_folder_file_name(h5): h5 for h5 in h5_refs}
        ref_lookup_no_gpu = {grab_folder_file_name(h5,no_gpu=True): h5 for h5 in h5_refs} # Used as backup in case exact file doesn't exist
        
        # Compare output against reference outputs
        compare_h5 = compare_data["h5_data"]
        matches = []
        missing_ref_files = []
        for h5_test in h5_tests:
            test_base = grab_folder_file_name(h5_test)
            h5_ref = ref_lookup.get(test_base)
            if h5_ref:
                logging.info(f"\n\nComparing\n{h5_test}\n{h5_ref}\n")
                matches.append(compare_h5(h5_ref, h5_test, node_screenshots,tolerance=tolerance))
            else:
                # See if there is another reference file that uses different gpu
                test_base_no_gpu = grab_folder_file_name(h5_test,no_gpu=True)
                h5_ref = ref_lookup_no_gpu.get(test_base_no_gpu)
                if h5_ref:
                    logging.info(f"\nComparing\n{h5_test}\n{h5_ref}\n")
                    matches.append(compare_h5(h5_ref, h5_test, node_screenshots,tolerance=tolerance))
                else:
                    missing_ref_files.append(h5_test)
        
        if len(missing_ref_files) > 0:
            files = '\n'.join(str(f) for f in missing_ref_files)
            pytest.skip(f"The following files are missing equivalent files in {ref_folder}:\n{files}")
        
        # Check that all matches are True
        outputs_match = all(matches) 
            
        return outputs_match
    
    return _compare_BabelBrain_Outputs