import pytest
import os
import sys
sys.path.append('./BabelBrain/')
import glob
import numpy as np
import trimesh
from unittest.mock import patch
import pyvista as pv
import base64
from io import BytesIO

from .config import test_data_path
import BabelDatasetPreps as bdp

# FOLDER/FILE PATHS
root_path = os.getcwd()
test_path = root_path + os.sep + 'Tests' + os.sep
# test_data_path = ""
thermal_profile_file = test_data_path + 'Thermal_Profiles' + os.sep + 'Profile_1.yaml'

# PARAMETERS
trajectories = [
    'Deep_Target',
    'Superficial_Target',
    'Skull_Target',
    'Outside_Target'
]

# test_datasets = [
#     # Dataset(test_data_path,'SDR_0p01',trajectories),      # ZTE Dataset
#     # Dataset(test_data_path,'SDR_0p02',trajectories),      # ZTE Dataset
#     # Dataset(test_data_path,'SDR_0p03',trajectories),      # ZTE Dataset
#     Dataset(test_data_path,'SDR_0p31',trajectories),      # CT Dataset
#     # Dataset(test_data_path,'SDR_0p42',trajectories),      # CT Dataset
#     Dataset(test_data_path,'SDR_0p55',trajectories),      # CT Dataset
#     # Dataset(test_data_path,'SDR_0p67',trajectories),      # CT Dataset
#     Dataset(test_data_path,'SDR_0p79',trajectories),      # CT Dataset
#     # Dataset(test_data_path,'PETRA_TEST',trajectories),
# ]

test_datasets = {
    'SDR_0p31': test_data_path + 'SDR_0p31' + os.sep,
    # 'SDR_0p42': test_data_path + 'SDR_0p42' + os.sep,
    'SDR_0p55': test_data_path + 'SDR_0p55' + os.sep,    
    # 'SDR_0p67': test_data_path + 'SDR_0p67' + os.sep,     
    'SDR_0p79': test_data_path + 'SDR_0p79' + os.sep,     
}

transducers = {
    'Single': 0,
    'CTX_500': 1,
    'H317': 2,
    'H246': 3,
    'BSonix': 4,
}

# CREATE TEST COMBINATIONS
test_parameters_valid_cases = []
test_parameters_invalid_cases = []
ids_valid_cases = []
ids_invalid_cases = []

for id in test_datasets.keys():
    for tx in transducers.keys():
        for trajectory in trajectories:
            # Valid Test Cases
            if trajectory in ['Deep_Target','Superficial_Target','Skull_Target']:
                valid_case = {'dataset': id, 
                              'tx': tx, 
                              'target': trajectory, 
                              'target_file': test_datasets[id] + 'Trajectories' + os.sep + trajectory + '.txt'}
                
                test_parameters_valid_cases.append(valid_case)
                ids_valid_cases.append(f'{id} Data, {tx}, {trajectory}')

            # Invalid Test Cases
            if trajectory in ['Outside_Target']:
                invalid_case = {'dataset': id, 
                                'tx': tx, 
                                'target': trajectory, 
                                'target_file': test_datasets[id] + 'Trajectories' + os.sep + trajectory + '.txt'}

                if invalid_case not in test_parameters_invalid_cases:
                    test_parameters_invalid_cases.append(invalid_case)
                    ids_invalid_cases.append(f'{id} Data, {tx}, {trajectory}')

def save_pyvista_plot(mesh1,mesh2,mesh3):
    # Create pyvista plot
    plotter = pv.Plotter(window_size=(800, 600),off_screen=True)
    plotter.add_mesh(mesh1, opacity=0.2)
    plotter.add_mesh(mesh2, opacity=0.2)
    plotter.add_mesh(mesh3, opacity=0.5, color='red')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plotter.show(screenshot=buffer)
    
    # Encode the image data as base64 string
    base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return base64_plot

# TEST FIXTURES
@pytest.fixture()
def load_input_meshes(test_parameters):
    
    #Parameters
    dataset_test_path = test_datasets[test_parameters['dataset']] + 'TestingDoIntersect' + os.sep
    transducer = test_parameters['tx']
    target = test_parameters['target']

    # Load input meshes
    # RELOADED EACH TEST, LOOK AT REMOVING THIS REDUNDANCY
    skin_mesh = trimesh.load(dataset_test_path + 'skin.stl')
    skull_mesh = trimesh.load(dataset_test_path + 'bone.stl')
    csf_mesh = trimesh.load(dataset_test_path + 'csf.stl')

    for file in glob.glob(dataset_test_path + '*cone.stl'):
            if transducer in file:
                if target in file:
                    cone_mesh = trimesh.load(file)

    box_mesh = ""
    for file in glob.glob(dataset_test_path + '*box.stl'):
        if transducer in file:
            if target in file:
                box_mesh = trimesh.load(file)

    return [skin_mesh,skull_mesh,csf_mesh,cone_mesh,box_mesh]
                          
class TestBabelDatasetPreps:

    @pytest.mark.parametrize('test_parameters',test_parameters_valid_cases, ids=ids_valid_cases)
    def test_do_intersect_valid_cases(self,test_parameters,load_input_meshes,request):

        #Parameters
        dataset_test_path = test_datasets[test_parameters['dataset']] + 'TestingDoIntersect' + os.sep
        transducer = test_parameters['tx']
        target = test_parameters['target']

        freq = '500kHz'
        if transducer == 'BSonix':
            freq = '650kHz'
        if transducer == 'H317':
            freq = '250kHz'

        # Load input meshes
        skin_mesh = load_input_meshes[0]
        skull_mesh = load_input_meshes[1]
        csf_mesh = load_input_meshes[2]
        cone_mesh = load_input_meshes[3]
        box_mesh = load_input_meshes[4]

        # Load truth meshes (blender)
        truth_meshes = {}
        for file in glob.glob(dataset_test_path + '*intersection_truth.stl'):
                if transducer in file:
                    if target in file:
                        if 'skin_cone' in file:
                            truth_meshes['skin_cone_intersection'] = trimesh.load(file)
                        elif 'skin_box' in file:
                            truth_meshes['skin_box_intersection'] = trimesh.load(file)
                        elif 'skull_box' in file:
                            truth_meshes['skull_box_intersection'] = trimesh.load(file)
                        elif 'csf_box' in file:
                            truth_meshes['csf_box_intersection'] = trimesh.load(file)

        # PRODUCE AND SAVE TRUTH MESHES IF NOT ALREADY CREATED
        # truth_meshes['skin_cone_intersection'] = bdp.DoIntersectBlender(skin_mesh,cone_mesh)
        # truth_meshes['skin_cone_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_skin_cone_intersection_truth.stl')
                            
        # truth_meshes['skin_box_intersection'] = bdp.DoIntersectBlender(skin_mesh,box_mesh)
        # truth_meshes['skin_box_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_skin_box_intersection_truth.stl')
                            
        # truth_meshes['skull_box_intersection'] = bdp.DoIntersectBlender(skull_mesh,box_mesh)
        # truth_meshes['skull_box_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_skull_box_intersection_truth.stl')
                            
        # truth_meshes['csf_box_intersection'] = bdp.DoIntersectBlender(csf_mesh,box_mesh)
        # truth_meshes['csf_box_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_csf_box_intersection_truth.stl')

        # Run intersection steps
        output_meshes = {}
        output_meshes['skin_cone_intersection'] = bdp.DoIntersect(skin_mesh,cone_mesh)
        output_meshes['skin_cone_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_skin_cone_intersection_pycork.stl')
        request.node.intersection1_screenshot = save_pyvista_plot(skin_mesh,cone_mesh,output_meshes['skin_cone_intersection'])

        output_meshes['skin_box_intersection'] = bdp.DoIntersect(skin_mesh,box_mesh)
        output_meshes['skin_box_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_skin_box_intersection_pycork.stl')
        request.node.intersection2_screenshot = save_pyvista_plot(skin_mesh,box_mesh,output_meshes['skin_box_intersection'])

        output_meshes['skull_box_intersection'] = bdp.DoIntersect(skull_mesh,box_mesh)
        output_meshes['skull_box_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_skull_box_intersection_pycork.stl')
        request.node.intersection3_screenshot = save_pyvista_plot(skull_mesh,box_mesh,output_meshes['skull_box_intersection'])

        output_meshes['csf_box_intersection'] = bdp.DoIntersect(csf_mesh,box_mesh)
        output_meshes['csf_box_intersection'].export(dataset_test_path + f'{target}_{transducer}_{freq}_6PPW_csf_box_intersection_pycork.stl')
        request.node.intersection4_screenshot = save_pyvista_plot(csf_mesh,box_mesh,output_meshes['csf_box_intersection'])

        # Compare vertices,faces, and area for output meshes
        all_meshes_equal = True
        fail_msg = "\n"
        for key in output_meshes.keys():
            vertices_equal = len(output_meshes[key].vertices) == len(truth_meshes[key].vertices)
            faces_equal = len(output_meshes[key].faces) == len(truth_meshes[key].faces)
            percent_error_area = abs((output_meshes[key].area - truth_meshes[key].area)/truth_meshes[key].area)
            area_equal = percent_error_area < 0.01 # Less than 1% deviation
            
            if vertices_equal and faces_equal and area_equal:
                pass
            else:
                all_meshes_equal = False
                if not vertices_equal:
                    fail_msg += f"{key}: # of vertices does not match ({len(output_meshes[key].vertices)} vs {len(truth_meshes[key].vertices)})\n"
                if not faces_equal:
                    fail_msg += f"{key}: # of faces does not match ({len(output_meshes[key].faces)} vs {len(truth_meshes[key].faces)})\n"
                if not area_equal:
                    fail_msg += f"{key}: area size does not match ({output_meshes[key].area} vs {truth_meshes[key].area}), percent error = {percent_error_area*100}%\n"


        assert all_meshes_equal == True, fail_msg

    @pytest.mark.parametrize('test_parameters',test_parameters_invalid_cases, ids=ids_invalid_cases)
    def test_do_intersect_invalid_cases(self,tmp_path,test_parameters,load_input_meshes):

        # Load input meshes
        skin_mesh = load_input_meshes[0]
        cone_mesh = load_input_meshes[3]

        # Run intersection steps
        with pytest.raises(ValueError) as exc_info:
            bdp.DoIntersectPycork(skin_mesh,cone_mesh)

        assert str(exc_info.value) == "Trajectory is outside headspace", "Test did not result in error when it should have."