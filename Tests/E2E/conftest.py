import glob
import logging
import os

import pytest

@pytest.fixture
def compare_BabelBrain_Outputs(compare_data):
    def _compare_BabelBrain_Outputs(ref_folder,test_folder,tolerance):
        
        # Find h5 files in folders
        pattern = "**/*ThermalField_AllCombinations.h5"

        h5_refs = glob.glob(os.path.join(ref_folder, pattern),recursive=True)
        h5_tests = glob.glob(os.path.join(test_folder, pattern),recursive=True)
        
        # Build lookup for reference files
        ref_lookup = {os.path.basename(h5): h5 for h5 in h5_refs}
        
        # Compare output against reference outputs
        compare_h5 = compare_data["h5_data"]
        matches = []
        missing_ref_files = []
        for h5_test in h5_tests:
            test_base = os.path.basename(h5_test)
            h5_ref = ref_lookup.get(test_base)
            if h5_ref:
                logging.info(f"\nComparing\n{h5_test}\n{h5_ref}\n")
                matches.append(compare_h5(h5_ref, h5_test, tolerance=tolerance))
            else:
                missing_ref_files.append(h5_test)
        
        if len(missing_ref_files) > 0:
            files = '\n'.join(str(f) for f in missing_ref_files)
            pytest.skip(f"The following files are missing equivalent files in {ref_folder}:\n{files}")
        
        # Check that all matches are True
        outputs_match = all(matches) 
            
        return outputs_match
    
    return _compare_BabelBrain_Outputs