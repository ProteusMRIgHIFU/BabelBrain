import pytest
import os
import sys
sys.path.append('./BabelBrain/')
import glob
import shutil
import numpy as np
import nibabel
import trimesh
import re
from unittest.mock import patch
import warnings

from config import test_data_path
from .Dataset import Dataset
from BabelBrain.BabelBrain import BabelBrain
from BabelBrain.SelFiles.SelFiles import SelFiles

# FOLDER/FILE PATHS
root_path = os.getcwd()
test_path = root_path + os.sep + 'Tests' + os.sep
# test_data_path = "" 
thermal_profile_file = test_data_path + 'Thermal_Profiles' + os.sep + 'Profile_1.yaml'

# PARAMETERS
trajectory_type = {
    'brainsight': 0,
    'slicer': 1
}

trajectories = [
    'Deep_Target',
    'Superficial_Target',
    'Skull_Target',
    'Outside_Target'
]

SimNIBS_type = {
    'charm': 0,
    'headreco': 1
}

test_datasets = [
    Dataset(test_data_path,'SDR_0p01',trajectories),      # ZTE Dataset
    # Dataset(test_data_path,'SDR_0p02',trajectories),      # ZTE Dataset
    # Dataset(test_data_path,'SDR_0p03',trajectories),      # ZTE Dataset
    # Dataset(test_data_path,'SDR_0p31',trajectories),      # CT Dataset
    # Dataset(test_data_path,'SDR_0p42',trajectories),      # CT Dataset
    Dataset(test_data_path,'SDR_0p55',trajectories),      # CT Dataset
    # Dataset(test_data_path,'SDR_0p67',trajectories),      # CT Dataset
    # Dataset(test_data_path,'SDR_0p79',trajectories),      # CT Dataset
    # Dataset(test_data_path,'PETRA_TEST',trajectories),
]

CT_type = {
    'NONE': 0,      # T1W Only
    'CT': 1,
    'ZTE': 2,
    # 'PETRA': 3    # TO BE ADDED LATER
}

coregistration = {
    'no': 0, 
    'yes': 1
}

transducers = {
    'Single': 0,
    'CTX_500': 1,
    'H317': 2,
    'H246': 3,
    'BSonix': 4,
}

# Not currently being tested
computing_backend = [
    'CPU',
    'OpenCL',
    'CUDA',
    'Metal'
]

# CREATE TEST COMBINATIONS
test_parameters_valid_cases = []
test_parameters_invalid_cases = []
ids_valid_cases = []
ids_invalid_cases = []

for dataset in test_datasets:
    for scan_type in CT_type.keys():
        if scan_type == 'NONE' or scan_type in dataset.CT_type:
            for tx in transducers.keys():
                for trajectory in trajectories:
                    # Valid Test Cases
                    if trajectory in dataset.get_valid_trajectories():
                        valid_case = {'dataset': dataset, 
                                      'scan_type': scan_type, 
                                      'tx': tx, 
                                      'target': trajectory, 
                                      'target_file': dataset.trajectories[trajectory]}
                        
                        test_parameters_valid_cases.append(valid_case)
                        ids_valid_cases.append(f'{dataset.id} Data, Add. Scan: {scan_type}, {tx}, {trajectory}')

                    # Invalid Test Cases
                    if trajectory in dataset.get_invalid_trajectories():
                        # Test should fail before coregistration step therefore don't need to test for different scan types
                        invalid_case = {'dataset': dataset, 
                                        'scan_type': 'NONE', 
                                        'tx': tx, 
                                        'target': trajectory, 
                                        'target_file': dataset.trajectories[trajectory]}

                        if invalid_case not in test_parameters_invalid_cases:
                            test_parameters_invalid_cases.append(invalid_case)
                            ids_invalid_cases.append(f'{dataset.id} Data, Add. Scan: NONE, {tx}, {trajectory}')

# TEST FIXTURES
@pytest.fixture()
def selfiles_widget(qtbot):

    sf_widget = SelFiles()
    sf_widget.show()
    qtbot.addWidget(sf_widget) # qtbot will handle sf_widget teardown

    return sf_widget
                          
@pytest.fixture()
def babelbrain_widget(qtbot,test_parameters,selfiles_widget,tmp_path):

    test_dataset = test_parameters['dataset']
    additional_scan_type = test_parameters['scan_type']
    transducer = test_parameters['tx']
    trajectory_value = test_parameters['target_file']

    # Set SelFiles Parameters
    selfiles_widget.ui.TrajectoryTypecomboBox.setCurrentIndex(trajectory_type['slicer'])
    selfiles_widget.ui.TrajectorylineEdit.setText(trajectory_value)
    selfiles_widget.ui.SimbNIBSTypecomboBox.setCurrentIndex(SimNIBS_type['charm'])
    selfiles_widget.ui.SimbNIBSlineEdit.setText(test_dataset.SimNIBS_folder)
    selfiles_widget.ui.T1WlineEdit.setText(test_dataset.T1W_file)
    selfiles_widget.ui.CTTypecomboBox.setCurrentIndex(CT_type[additional_scan_type])
    if CT_type[additional_scan_type] != 0:
        selfiles_widget.ui.CoregCTcomboBox.setCurrentIndex(coregistration['yes'])
        selfiles_widget.ui.CTlineEdit.setText(test_dataset.CT_file)
    selfiles_widget.ui.ThermalProfilelineEdit.setText(thermal_profile_file)
    selfiles_widget.ui.TransducerTypecomboBox.setCurrentIndex(transducers[transducer])
    selfiles_widget.ui.ContinuepushButton.click()

    # Create BabelBrain widget
    bb_widget = BabelBrain(selfiles_widget)
    bb_widget.show()
    qtbot.addWidget(bb_widget) # qtbot will handle bb_widget teardown

    # Copy T1W file and additional scan over to temporary folder
    shutil.copy(bb_widget.Config['T1W'],os.path.join(tmp_path,os.path.basename(bb_widget.Config['T1W'])))
    if additional_scan_type != "NONE":
        shutil.copy(test_dataset.CT_file,os.path.join(tmp_path,os.path.basename(test_dataset.CT_file)))

    # Copy SimbNIBs input file over to temporary folder
    if bb_widget.Config['SimbNIBSType'] == 'charm':
        SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'final_tissues.nii.gz'
    else:
        SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'skin.nii.gz'
    os.makedirs(os.path.join(tmp_path,os.path.basename(os.path.dirname(bb_widget.Config['simbnibs_path']))),exist_ok=True)
    shutil.copy(SimbNIBSInput,os.path.join(tmp_path,re.search("m2m.*",SimbNIBSInput)[0]))

    # Copy Trajectory file over to temporary folder
    trajectory_new_file = os.path.join(tmp_path,os.path.basename(bb_widget.Config['Mat4Trajectory']))
    shutil.copy(bb_widget.Config['Mat4Trajectory'],trajectory_new_file)

    # Edit file paths so new data is saved in temporary folder
    bb_widget.Config['Mat4Trajectory'] = trajectory_new_file
    bb_widget.Config['T1WIso'] = os.path.join(tmp_path,os.path.basename(bb_widget.Config['T1WIso']))
    bb_widget.Config['simbnibs_path'] = os.path.join(tmp_path,os.path.split(os.path.split(bb_widget.Config['simbnibs_path'])[0])[1])
    if bb_widget.Config['bUseCT']:
        bb_widget.Config['CT_or_ZTE_input'] = os.path.join(tmp_path,os.path.basename(bb_widget.Config['CT_or_ZTE_input']))

    # Set Sim Parameters
    if transducer == 'H317':
        freq = '700'
    elif transducer == 'BSonix':
        freq = '650'
    else:
        freq = '500'

    freq_index = bb_widget.Widget.USMaskkHzDropDown.findText(freq)
    bb_widget.Widget.USMaskkHzDropDown.setCurrentIndex(freq_index)
    bb_widget.Widget.USPPWSpinBox.setProperty('UserData',6) # 6 PPW
    if additional_scan_type != "NONE":
        bb_widget.Widget.HUThresholdSpinBox.setValue(300)

    yield bb_widget
    
    # Teardown Code
    if tmp_path.exists():
        shutil.rmtree(tmp_path) # Remove all files created in tmp folder

@pytest.fixture()
def mock_NotifyError(babelbrain_widget,monkeypatch):
    # Tests hang up when a messagebox is created during testing (e.g. an error box)
    # To prevent this from happening, we mock the function responsible for creating the error box

    def _mock_NotifyError(self):
        self.Widget.tabWidget.setEnabled(True) # Set to True to prevent timeout from occurring
        self.testing_error = True
    
    monkeypatch.setattr(BabelBrain,'NotifyError',_mock_NotifyError)

    return _mock_NotifyError

@pytest.fixture()
def mock_UpdateMask(babelbrain_widget,monkeypatch):
    # Occasionally get errors during UpdateMask call, we'll temporarily mock it until
    # more in depth testing can reveal the cause

    def _mock_UpdateMask(self):
        self.Widget.tabWidget.setEnabled(True) # Set to True to prevent timeout from occurring
        self.testing_error = False
    
    monkeypatch.setattr(BabelBrain,'UpdateMask',_mock_UpdateMask)

    return _mock_UpdateMask

class TestStep1:

    @pytest.mark.parametrize('test_parameters',test_parameters_valid_cases, ids=ids_valid_cases)
    def test_step1_valid_cases(self,qtbot,babelbrain_widget,test_parameters,tmp_path,mock_NotifyError,mock_UpdateMask):

        test_dataset = test_parameters['dataset']
        additional_scan_type = test_parameters['scan_type']
        transducer = test_parameters['tx']
        trajectory_name = test_parameters['target']

        # Run Step 1
        babelbrain_widget.testing_error = False
        babelbrain_widget.Widget.CalculatePlanningMask.click()

        # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
        qtbot.waitUntil(babelbrain_widget.Widget.tabWidget.isEnabled,timeout=900000)

        # Check if step 1 failed
        if babelbrain_widget.testing_error == True:
            pytest.fail(f"Test failed due to error in execution")

        # Load Truth Data and compare against generated results
        if additional_scan_type == "NONE":
            truth_folder = test_dataset.truth_data_filepath + 'T1W_Only' + os.sep + transducer + os.sep + trajectory_name + os.sep
        elif additional_scan_type == "CT":
            truth_folder = test_dataset.truth_data_filepath + 'T1W_with_CT' + os.sep + transducer + os.sep + trajectory_name + os.sep
        elif additional_scan_type == "ZTE":
            truth_folder = test_dataset.truth_data_filepath + 'T1W_with_ZTE' + os.sep + transducer + os.sep + trajectory_name + os.sep
        elif additional_scan_type == "PETRA":
            truth_folder = test_dataset.truth_data_filepath + 'T1W_with_PETRA' + os.sep + transducer + os.sep + trajectory_name + os.sep
        else:
            truth_folder = ""
        
        cumulative_error = 0
        for truth_file in glob.glob(truth_folder + '*'):
            
            truth_file_name = os.path.basename(truth_file)
            gen_file = os.path.join(tmp_path,truth_file_name)

            if '.nii.gz' in truth_file:
                truth_nib = nibabel.load(truth_file)
                gen_nib = nibabel.load(gen_file)

                truth_data = truth_nib.get_fdata()
                gen_data = gen_nib.get_fdata()
            elif '.npz' in truth_file:
                truth_data = np.load(truth_file)['UniqueHU']
                gen_data = np.load(gen_file)['UniqueHU']
            elif '.stl' in truth_file:
                truth_stl = trimesh.load(truth_file)
                gen_stl = trimesh.load(gen_file)

                truth_data = truth_stl.vertices
                gen_data = gen_stl.vertices
            else:
                pass

            if len(truth_data) != len(gen_data):
                pytest.fail(f"\n{truth_file_name}\nNumber of voxels in truth data ({len(truth_data)}) does not match number in generated data ({len(gen_data)})\n")

            current_mse = np.mean((gen_data - truth_data) ** 2)
            current_range = np.max(truth_data) - np.min(truth_data)
            current_norm_mse = current_mse / current_range

            if current_norm_mse > 0:
                warnings.warn(f"{truth_file_name} had a mean square error of {current_mse}, range of {current_range}, and a normal MSE of {current_norm_mse}")

            cumulative_error += current_norm_mse
        
        assert cumulative_error == 0, f"Cumulative error was {cumulative_error}"

    @pytest.mark.parametrize('test_parameters',test_parameters_invalid_cases, ids=ids_invalid_cases)
    def test_step1_invalid_cases(self,qtbot,babelbrain_widget,tmp_path,test_parameters,mock_NotifyError,mock_UpdateMask):

        # Run Step 1
        babelbrain_widget.Widget.CalculatePlanningMask.click()

        # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
        qtbot.waitUntil(babelbrain_widget.Widget.tabWidget.isEnabled,timeout=900000)

        # Placeholder test. Need to find a way to grab the specific error generated.
        assert babelbrain_widget.testing_error == True