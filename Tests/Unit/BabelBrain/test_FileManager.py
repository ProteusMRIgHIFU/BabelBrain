import hashlib
import os
import sys
import logging
from unittest import mock

import pytest
import nibabel
import numpy as np
import SimpleITK as sitk

@pytest.mark.step1
class TestFileManager:
    def test_wait_for_file(self,set_up_file_manager,get_example_data,tmpdir,caplog):
        
        # Set up file manager
        fm,_,_ = set_up_file_manager['blank']()

        # Save random numpy array in temp file
        random_array = get_example_data['numpy']()
        temp_file_path = tmpdir.join("random_array.npy")

        # Start saving file
        fm.save_file(random_array,temp_file_path)

        # Call wait_for_file method
        with caplog.at_level(logging.INFO):
            fm.wait_for_file(temp_file_path)

        assert f"{temp_file_path} needs to save first" in caplog.text

    def test_generate_hash(self,set_up_file_manager, tmpdir):
            
            # Set up file manager
            fm,_,_ = set_up_file_manager['blank']()

            # Create a temporary file for testing
            temp_file = tmpdir.join("test_file.txt")
            msg = "Sample content for hashing"
            with open(temp_file,'w') as file:
                file.write(msg)

            # Calculate the hash using generate_hash method
            result = fm.generate_hash(str(temp_file))

            # Calculate the hash manually for comparison
            blake2s_hash = hashlib.blake2s(digest_size=4)
            blake2s_hash.update(msg.encode('utf-8'))
            expected_hash = blake2s_hash.hexdigest()

            assert result == expected_hash, "Generated hash does not match expected hash"

    def test_generate_hash_file_not_found(self,set_up_file_manager):
        
        # Set up file manager
        fm,_,_ = set_up_file_manager['blank']()

        # Test if the function raises FileNotFoundError when file does not exist
        with pytest.raises(FileNotFoundError, match="No such file: 'non_existent_file.txt'"):
            fm.generate_hash("non_existent_file.txt")

    def test_generate_hash_read_error(self,set_up_file_manager, tmpdir):
            
            # Set up file manager
            fm,_,_ = set_up_file_manager['blank']()

            # Create a temporary file for testing
            temp_file = tmpdir.join("test_file.txt")
            msg = "Sample content for hashing"
            with open(temp_file,'w') as file:
                file.write(msg)

            # Simulate an OSError when trying to read the file
            with mock.patch("builtins.open", side_effect=OSError("Simulated read error")):
                with pytest.raises(RuntimeError, match="An error occurred while reading the file"):
                    fm.generate_hash(str(temp_file))

    def test_nibabel_to_sitk(self,dataset,set_up_file_manager,extract_sitk_info,compare_data):
        
        # Set up file manager
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run nibabel_to_sitk function
        output_data = fm.nibabel_to_sitk(fm.saved_objects['T1_nib'])

        # Extract info
        output_spacing,output_direction,output_origin,output_array = extract_sitk_info(output_data)
        truth_spacing,truth_direction,truth_origin,truth_array = extract_sitk_info(fm.saved_objects['T1_sitk'])

        # Compare info
        dice_coeff_spacing = compare_data['dice_coefficient'](output_spacing, truth_spacing)
        dice_coeff_direction = compare_data['dice_coefficient'](output_direction, truth_direction)
        dice_coeff_origin = compare_data['dice_coefficient'](output_origin, truth_origin)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_origin + dice_coeff_array)/4
        
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_sitk_to_nibabel(self,dataset,set_up_file_manager,extract_nib_info,compare_data):
        
        # Set up file manager
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run sitk_to_nibabel function
        output_data = fm.sitk_to_nibabel(fm.saved_objects['T1_sitk'])

        # Extract info
        output_zooms,output_affine,output_array = extract_nib_info(output_data)
        truth_zooms,truth_affine,truth_array = extract_nib_info(fm.saved_objects['T1_nib'])

        # Compare info
        dice_coeff_spacing = compare_data['dice_coefficient'](output_zooms, truth_zooms)
        dice_coeff_direction = compare_data['dice_coefficient'](output_affine, truth_affine)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_array)/3
        
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"
        
    def test_nibabel_to_sitk_and_back(self,dataset,set_up_file_manager,extract_nib_info,compare_data):
        
        # Set up file manager
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run nibabel_to_sitk function
        output_data = fm.nibabel_to_sitk(fm.saved_objects['T1_nib'])
        
        # Run sitk_to_nibabel function
        output_data = fm.sitk_to_nibabel(output_data)

        # Extract info
        output_zooms,output_affine,output_array = extract_nib_info(output_data)
        truth_zooms,truth_affine,truth_array = extract_nib_info(fm.saved_objects['T1_nib'])

        # Compare info
        dice_coeff_spacing = compare_data['dice_coefficient'](output_zooms, truth_zooms)
        dice_coeff_direction = compare_data['dice_coefficient'](output_affine, truth_affine)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_array)/3
        
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_sitk_to_nibabel_and_back(self,dataset,set_up_file_manager,extract_sitk_info,compare_data):
        
        # Set up file manager
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run sitk_to_nibabel function
        output_data = fm.sitk_to_nibabel(fm.saved_objects['T1_sitk'])
        
        # Run nibabel_to_sitk function
        output_data = fm.nibabel_to_sitk(output_data)

        # Extract info
        output_spacing,output_direction,output_origin,output_array = extract_sitk_info(output_data)
        truth_spacing,truth_direction,truth_origin,truth_array = extract_sitk_info(fm.saved_objects['T1_sitk'])

        # Compare info
        dice_coeff_spacing = compare_data['dice_coefficient'](output_spacing, truth_spacing)
        dice_coeff_direction = compare_data['dice_coefficient'](output_direction, truth_direction)
        dice_coeff_origin = compare_data['dice_coefficient'](output_origin, truth_origin)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_origin + dice_coeff_array)/4
        
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_check_reuse_files_no_files(self,scan_type,set_up_file_manager,capfd):
        
        # Set up file manager for 'previous run'
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Set up file manager for 'second run'
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements, _ = capfd.readouterr()

        assert reuse == False, "Files were reused even though they don't exist"
        assert "Previous files don't exist" in print_statements, "Reason for not reusing files was incorrect"

    def test_check_reuse_files_valid(self,scan_type,set_up_file_manager,get_example_data):
        
        # Set up file manager for 'previous run'
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)

        assert reuse == True, f"Files aren't reused even though they can be (reuse = {reuse})"

    def test_check_reuse_files_different_CT_type(self,scan_type,second_scan_type,set_up_file_manager,get_example_data,capfd):
        
        if scan_type == second_scan_type:
            pytest.skip("Not a different scan type")
        
        # Set up file manager for 'previous run'
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](second_scan_type)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements,_ = capfd.readouterr()

        assert reuse == False, f"Files are reused even though they shouldn't be (reuse = {reuse})"
        assert any(msg in print_statements for msg in ["Previous files used different CT Type", "Previous files didn't use CT"]), "Reason for not reusing files was incorrect"

    def test_check_reuse_files_different_HU_threshold(self,scan_type,set_up_file_manager,get_example_data,capfd):
        
        if scan_type == 'NONE':
            pytest.skip("No CT therefore no HUT value")

        # Set up file manager for 'previous run'
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type,HUT=400.0)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements,_ = capfd.readouterr()

        assert reuse == False, f"Files are reused even though they shouldn't be (reuse = {reuse})"
        assert any(msg in print_statements for msg in ["Previous files used different HUThreshold","Previous files didn't have HU Threshold value"]), "Reason for not reusing files was incorrect"

    def test_check_reuse_files_different_pCT_range(self,scan_type,set_up_file_manager,get_example_data,capfd):
        if scan_type == 'NONE' or scan_type == 'CT':
            pytest.skip("No CT or real CT therefore no pseudo CT range value")

        # Set up file manager for 'previous run'
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type,pCT_range=(0.1,0.5))
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements,_ = capfd.readouterr()

        assert reuse == False, f"Files are reused even though they shouldn't be (reuse = {reuse})"
        assert any(msg in print_statements for msg in ["Previous files used different ZTE/PETRA Range","Previous files didn't have ZTE/PETRA Range values"]), "Reason for not reusing files was incorrect"