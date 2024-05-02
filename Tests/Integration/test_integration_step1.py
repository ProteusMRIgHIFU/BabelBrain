import os
import sys
sys.path.append('./BabelBrain/')
import glob
import logging

import pytest

class TestStep1:

    def test_step1_valid_case(self,qtbot,trajectory,transducer,scan_type,dataset,babelbrain_widget,load_files,compare_data,mock_NotifyError,mock_UpdateMask,tmp_path):

        # Truth folder path
        if scan_type == "NONE":
            truth_folder = dataset['folder_path'] + 'Truth' + os.sep + 'Integration' + os.sep + 'T1W_Only' + os.sep + transducer['name'] + os.sep + trajectory + os.sep
        elif scan_type == "CT":
            truth_folder = dataset['folder_path'] + 'Truth' + os.sep + 'Integration' + os.sep + 'T1W_with_CT' + os.sep + transducer['name'] + os.sep + trajectory + os.sep
        elif scan_type == "ZTE":
            truth_folder = dataset['folder_path'] + 'Truth' + os.sep + 'Integration' + os.sep + 'T1W_with_ZTE' + os.sep + transducer['name'] + os.sep + trajectory + os.sep
        elif scan_type == "PETRA":
            truth_folder = dataset['folder_path'] + 'Truth' + os.sep + 'Integration' + os.sep + 'T1W_with_PETRA' + os.sep + transducer['name'] + os.sep + trajectory + os.sep
        else:
            truth_folder = ""

        # Quick check for truth data existence
        if len(glob.glob(truth_folder + '*')) < 3: # If there are less than 3 files present, we have missing truth files
            pytest.skip('Skipping test because truth files are missing')

        # Run Step 1
        babelbrain_widget.testing_error = False
        babelbrain_widget.Widget.CalculatePlanningMask.click()

        # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
        qtbot.waitUntil(babelbrain_widget.Widget.tabWidget.isEnabled,timeout=900000)

        # Check if step 1 failed
        if babelbrain_widget.testing_error == True:
            pytest.fail(f"Test failed due to error in execution")

        # Load output data
        output_files = {}
        for output_file in glob.glob(os.path.join(tmp_path,'*')):
            output_file_name = os.path.basename(output_file)
            output_files[output_file_name] = output_file
        output_data = load_files(output_files)

        # Load truth data
        truth_files = {}
        for fname in output_data.keys():
            truth_files[fname] = truth_folder + fname
        truth_data = load_files(truth_files)
        
        # Compare data
        compare_arrays_func = compare_data['array_data']
        compare_stl_area_func = compare_data['stl_area']
        lengths_equal = True
        cumulative_error = 0
        for fname in output_data.keys():
            logging.info(fname)
            if '.nii.gz' in fname:
                output_data[fname] = output_data[fname].get_fdata()

            if '.stl' in fname:
                percent_area_error = compare_stl_area_func(output_data[fname],truth_data[fname])
                if percent_area_error > 0.01:
                    pytest.fail(f"{fname} has a percent error of {percent_area_error*100}% for mesh area")
            else:
                array_length_same, array_norm_rmse = compare_arrays_func(output_data[fname],truth_data[fname])

            lengths_equal *= array_length_same
            cumulative_error += array_norm_rmse
        
        assert cumulative_error == 0 and lengths_equal, f"Array lengths were equal? {lengths_equal}\nCumulative error was {cumulative_error}"

    def test_step1_invalid_case(self,qtbot,trajectory,transducer,scan_type,dataset,babelbrain_widget,mock_NotifyError,mock_UpdateMask,tmp_path):

        # Run Step 1
        babelbrain_widget.Widget.CalculatePlanningMask.click()

        # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
        qtbot.waitUntil(babelbrain_widget.Widget.tabWidget.isEnabled,timeout=900000)

        # Placeholder test. Need to find a way to grab the specific error generated.
        assert babelbrain_widget.testing_error == True