import os
import sys
sys.path.append('.BabelBrain')
import logging

import numpy as np
import pytest
import trimesh

import BabelBrain.BabelDatasetPreps as bdp

@pytest.mark.step1
class TestBabelDatasetPreps:

    @pytest.mark.skip(reason="Need truth data + better metric for pass condition")
    def test_do_intersect_valid_case(self,trajectory,transducer,dataset,load_files,get_pyvista_plot,compare_data,request):
        
        # Parameters
        truth_folder = dataset['folder_path'] + 'Truth' + os.sep + 'Unit' + os.sep + 'DoIntersect' + os.sep
        input_folder = dataset['m2m_folder_path']

        # Load inputs
        input_files = {}
        for tissue in ['skin','bone','csf']:
            input_files[tissue] = input_folder + f"{tissue}.stl"
        for object in ['cone','box']:
            input_files[object] = truth_folder + f"{trajectory}_{transducer['name']}_{object}.stl"
        input_data = load_files(input_files)

        # Load truth data
        truth_files = {}
        for intersection_type in ['skin_cone','skin_box','bone_box','csf_box']:
            truth_files[intersection_type] = truth_folder + f"{trajectory}_{transducer['name']}_{intersection_type}_intersection_truth.stl"
        truth_data = load_files(truth_files)

        # Run intersection steps
        save_intersection_plot = get_pyvista_plot['intersection_plot']
        output_meshes = {}
        request.node.screenshots = []
        for intersection_type in truth_data.keys():
            mesh1_name, mesh2_name = intersection_type.split('_')
            mesh1 = input_data[mesh1_name]
            mesh2 = input_data[mesh2_name]
            output_meshes[intersection_type] = bdp.DoIntersect(mesh1,mesh2)

            # Save plot screenshot to be added to html report later
            screenshot = save_intersection_plot(mesh1,mesh2,output_meshes[intersection_type])
            request.node.screenshots.append(screenshot)

        # Compare stl data
        compare_stl_area_func = compare_data['stl_area']
        compare_stl_vertices_func = compare_data['array_data']
        all_meshes_equal = True
        for key in output_meshes.keys():
            logging.info(f"Intersection type: {key}")
            # Check number of vertices
            array_length_same, _ = compare_stl_vertices_func(output_meshes[key].vertices,truth_data[key].vertices)

            # Compare area
            percent_error_area = compare_stl_area_func(output_meshes[key],truth_data[key])
            area_equal = percent_error_area < 0.01
            if not area_equal:
                logging.error(f"Area for output mesh had a percent error of {percent_error_area}%")
            
            # Test will fail if vertex number or stl area don't match
            all_meshes_equal *= array_length_same
            all_meshes_equal *= area_equal

        assert all_meshes_equal == True
    
    def test_do_intersect_invalid_case(self,dataset,load_files,get_pyvista_plot,request):

        # Parameters
        input_folder = dataset['m2m_folder_path']

        # Load skin stl or create if it doesn't exist
        skin_file = input_folder + f"skin.stl"
        if os.path.exists(skin_file):
            skin_stl = load_files([skin_file])[0]
        else:
            final_tissues = load_files([input_folder+"final_tissues.nii.gz"])[0]
            skin = final_tissues.get_fdata()[:,:,:,0] > 0
            skin_stl = bdp.MaskToStl(skin,final_tissues.affine)

        # Create box (representing domain) and place outside head region
        skin_bounds = skin_stl.bounds
        skin_dims = np.diff(skin_bounds,axis=0)[0]
        max_skin_dim = np.max(skin_dims)
        box_radius = max_skin_dim//6
        offset = 10
        box_transform = np.eye(4)
        box_transform[0,3] = -max_skin_dim/2 - box_radius - offset
        box_transform[1,3] = max_skin_dim/2 + box_radius + offset
        box_transform[2,3] = max_skin_dim/2 + box_radius + offset

        box_stl = trimesh.creation.box(extents=[box_radius*2]*3,transform=box_transform)

        # Save plot screenshot to be added to html report later
        request.node.screenshots = []
        save_mesh_plot = get_pyvista_plot['mesh_plot']
        screenshot = save_mesh_plot([skin_stl,box_stl])
        request.node.screenshots.append(screenshot)
            
        # Run intersection steps
        with pytest.raises(ValueError) as exc_info:
            bdp.DoIntersect(skin_stl,box_stl)

        logging.info(exc_info)
        assert str(exc_info.value) == "Trajectory is outside headspace", "Test did not result in error when it should have."