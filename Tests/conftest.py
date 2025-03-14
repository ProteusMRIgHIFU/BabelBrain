
import datetime
import os
from glob import glob
import sys
sys.path.append('./BabelBrain/')
from pathlib import Path
import platform
import shutil
import re
import configparser
import logging

import pytest
import pytest_html
import trimesh
import pyvista as pv
import base64
from io import BytesIO
import numpy as np
np.random.seed(42) # RNG is same every time
import nibabel
from nibabel import processing, nifti1, affines
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is noninteractive
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error

from BabelBrain.BabelBrain import BabelBrain
from BabelBrain.SelFiles.SelFiles import SelFiles
from BabelBrain.FileManager import FileManager

# FOLDER/FILE PATHS
config = configparser.ConfigParser()
config.read('Tests' + os.sep + 'config.ini')
test_data_folder = config['Paths']['data_folder_path']
gpu_device = config['GPU']['device_name']
print('gpu_device',gpu_device)

# PARAMETERS
test_trajectory_type = {
    'brainsight': 0,
    'slicer': 1
}
valid_trajectories = [
    'Deep_Target',
    'Superficial_Target',
    'Skull_Target',
]
invalid_trajectories = [
    'Outside_Target'
]
SimNIBS_type = {
    'charm': 0,
    'headreco': 1
}
test_datasets = [
    {'id': 'SDR_0p31','folder_path': test_data_folder + os.sep + 'SDR_0p31' + os.sep},
    {'id': 'SDR_0p42','folder_path': test_data_folder + os.sep + 'SDR_0p42' + os.sep},
    {'id': 'SDR_0p55','folder_path': test_data_folder + os.sep + 'SDR_0p55' + os.sep},    
    {'id': 'SDR_0p67','folder_path': test_data_folder + os.sep + 'SDR_0p67' + os.sep},     
    {'id': 'SDR_0p79','folder_path': test_data_folder + os.sep + 'SDR_0p79' + os.sep},
    {'id': 'ID_0082','folder_path': test_data_folder + os.sep + 'ID_0082' + os.sep}
]
for ds in test_datasets:
    ds['m2m_folder_path'] = ds['folder_path'] + f"m2m_{ds['id']}" + os.sep
    ds['T1_path'] = ds['folder_path'] + "T1W.nii.gz"
    ds['T1_iso_path'] = ds['folder_path'] + "T1W-isotropic.nii.gz"

    if os.path.exists(ds['m2m_folder_path'] + 'charm_log.html'):
        ds['simbNIBS_type'] = 'charm'
    else:
        ds['simbNIBS_type'] = 'headreco'

CT_types = {
    'NONE': 0, # T1W Only
    'CT': 1,
    'ZTE': 2,
    'PETRA': 3
}
coregistration = {
    'no': 0,
    'yes': 1
}
thermal_profiles = {
    'thermal_profile_1': test_data_folder + os.sep + 'Profiles' + os.sep + 'Thermal_Profile_1.yaml',
    'thermal_profile_2': test_data_folder + os.sep + 'Profiles' + os.sep + 'Thermal_Profile_2.yaml',
    'thermal_profile_3': test_data_folder + os.sep + 'Profiles' + os.sep + 'Thermal_Profile_3.yaml'
}
transducers = [
    {'name': 'Single', 'dropdown_index': 0, 'diameter': 0}, # EDIT DIAMETER
    {'name': 'CTX_500', 'dropdown_index': 1, 'diameter': 0},
    {'name': 'CTX_250', 'dropdown_index': 2, 'diameter': 0},
    {'name': 'DPX_500', 'dropdown_index': 3, 'diameter': 0},
    {'name': 'H317', 'dropdown_index': 4, 'diameter': 0},
    {'name': 'H246', 'dropdown_index': 5, 'diameter': 0},
    {'name': 'BSonix', 'dropdown_index': 6, 'diameter': 0},
    {'name': 'REMOPD', 'dropdown_index': 7, 'diameter': 0},
    {'name': 'I12378', 'dropdown_index': 8, 'diameter': 0},
    {'name': 'ATAC', 'dropdown_index': 9, 'diameter': 0},
    {'name': 'R15148', 'dropdown_index': 10, 'diameter': 0}
]
computing_backends = [
    # {'type': 'CPU','supported_os': ['Mac','Windows','Linux']},
    {'type': 'OpenCL','supported_os': ['Mac','Windows','Linux']},
    {'type': 'CUDA','supported_os': ['Windows','Linux']},
    {'type': 'Metal','supported_os': ['Mac']}
]
spatial_step = {
    'Spatial_Step_0_919': 0.919,  # 200 kHz,   6 PPW
    # 'Spatial_Step_0_613': 0.613,  # 200 kHz,   9 PPW
    # 'Spatial_Step_0_459': 0.459,  # 200 kHz,  12 PPW
    'Spatial_Step_0_306': 0.306,  # 600 kHz,   6 PPW
    # 'Spatial_Step_0_204': 0.204,  # 600 kHz,   9 PPW
    # 'Spatial_Step_0_153': 0.153,  # 600 kHz,  12 PPW
    'Spatial_Step_0_184': 0.184,  # 1000 kHz,  6 PPW
    # 'Spatial_Step_0_123': 0.123,  # 1000 kHz,  9 PPW
    'Spatial_Step_0_092': 0.092,  # 1000 kHz, 12 PPW
}

# PYTEST FIXTURES
@pytest.fixture()
def check_files_exist():

    def _check_files_exist(fnames):
        missing_files = []
        for file in fnames:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            return False, missing_files
        else:
            return True, ""

    return _check_files_exist

@pytest.fixture()
def load_files(check_files_exist):
    
    def _load_files(fnames,nifti_load_method='nibabel',skip_test=True):

        if isinstance(fnames,dict):
            datas = fnames.copy()
            fnames_list = fnames.values()
        else:
            datas = []
            fnames_list = fnames

        # Check files exist
        files_exist, missing_files = check_files_exist(fnames_list)

        if not files_exist:
            if skip_test:
                logging.warning(f"Following files are missing: {', '.join(missing_files)}")
                pytest.skip(f"Skipping test because the following files are missing: {', '.join(missing_files)}")
            else:
                raise FileNotFoundError(f"Following files are missing: {', '.join(missing_files)}")

        # Load files based on their extensions
        if isinstance(datas, dict):
            # Iterate over dictionary keys and values directly
            for key, fname in fnames.items():
                datas[key] = _load_file(fname, nifti_load_method)
        else:
            # For lists, just iterate over file names
            for fname in fnames_list:
                datas.append(_load_file(fname,nifti_load_method))

        return datas
    
    def _load_file(fname, nifti_load_method='nibabel'):
        """Helper function to load a single file based on its extension."""
        
        # Get file extension type
        base, ext = os.path.splitext(fname)
        
        # Repeat for compressed files
        if ext == '.gz':
            base, ext = os.path.splitext(base)

        # Load file using appropriate method
        if ext == '.npy':
            return np.load(fname)
        elif ext == '.stl':
            return trimesh.load(fname)
        elif ext == '.nii':
            if nifti_load_method == 'nibabel':
                return nibabel.load(fname)
            elif nifti_load_method == 'sitk':
                return sitk.ReadImage(fname)
            else:
                raise ValueError(f"Invalid nifti load method specified: {nifti_load_method}")
        elif ext == '.txt':
            with open(fname, 'r') as file:
                content = file.read()
                return content
        else:
            logging.warning(f"Unsupported file extension, {fname} not loaded")

    return _load_files

@pytest.fixture()
def check_os(computing_backend):
    sys_os = None
    sys_platform = platform.platform(aliased=True)
    
    if 'macOS' in sys_platform:
        sys_os = 'Mac'
    elif 'Windows' in sys_platform:
        sys_os = 'Windows'
    elif 'Linux' in sys_platform:
        sys_os = 'Linux'
    else:
        logging.warning("No idea what os you're using")

    if sys_os not in computing_backend['supported_os']:
        pytest.skip("Skipping test because the selected computing backend is not available on this system")

@pytest.fixture(scope="session")
def get_gpu_device():
    
    def _get_gpu_device():
        return gpu_device

    return _get_gpu_device

@pytest.fixture()
def get_rmse():
    def _get_rmse(output_points, truth_points):
        rmse = np.sqrt(np.mean((output_points - truth_points) ** 2))
        data_range = np.max(truth_points) - np.min(truth_points)
        norm_rmse = rmse / data_range

        return rmse, data_range, norm_rmse
        
    return _get_rmse

@pytest.fixture()
def get_resampled_input(load_files):
    def _get_resampled_input(input,new_zoom,output_fname):

        if input.ndim > 3:
            tmp_data = input.get_fdata()[:,:,:,0]
            tmp_affine = input.affine
            input = nifti1.Nifti1Image(tmp_data,tmp_affine)

        # Determine new output dimensions and affine
        zooms = np.asarray(input.header.get_zooms())
        new_zooms = np.full(3,new_zoom)
        logging.info(f"Original zooms: {zooms}")
        logging.info(f"New zooms: {new_zooms}")
        new_x_dim = int(input.shape[0]/(new_zooms[0]/zooms[0]))
        new_y_dim = int(input.shape[1]/(new_zooms[1]/zooms[1]))
        new_z_dim = int(input.shape[2]/(new_zooms[2]/zooms[2]))
        new_affine = affines.rescale_affine(input.affine.copy(),
                                                input.shape,
                                                new_zooms,
                                                (new_x_dim,new_y_dim,new_z_dim))

        # Create output
        output_data = np.zeros((new_x_dim,new_y_dim,new_z_dim),dtype=np.uint8)
        output_nifti = nifti1.Nifti1Image(output_data,new_affine)
        logging.info(f"New Dimensions: {output_data.shape}")
        logging.info(f"New Size: {output_data.size}")

        try:
            logging.info('Reloading resampled input')
            resampled_nifti = load_files([output_fname],skip_test=False)[0]
            resampled_data = resampled_nifti.get_fdata()
        except:
            logging.info("File doesn't exist")
            logging.info('Generating resampled input')
            resampled_nifti = processing.resample_from_to(input,output_nifti,mode='constant',order=0,cval=input.get_fdata().min()) # Truth method
            resampled_data = resampled_nifti.get_fdata()
            logging.info('Saving file for future use')
            nibabel.save(resampled_nifti,output_fname)

        # Check data is contiguous
        if not resampled_data.flags.contiguous:
            logging.info("Changing resampled input data to be a contiguous array")
            resampled_data = np.ascontiguousarray(resampled_data)

        return resampled_nifti, resampled_data
    
    return _get_resampled_input

@pytest.fixture()
def check_data():
    def isometric_check(nifti):
        logging.info('Running isometric check')
        zooms = nifti.header.get_zooms()
        logging.info(f"Zooms: {zooms}")
        diffs = np.abs(np.subtract.outer(zooms, zooms))
        isometric = np.all(diffs <= 1e-6)

        return isometric

    # Return the fixture object with the specified attribute
    return {'isometric': isometric_check}

@pytest.fixture()
def compare_data(get_rmse):

    def array_data(output_array,truth_array):
        logging.info('Calculating root mean square error')

        array_rmse = array_range = array_norm_rmse = None

        # Check array size
        if len(output_array) == len(truth_array):
            logging.info(f"Number of array points are equal: {len(output_array)}")
            array_length_same = True

            array_rmse, array_range, array_norm_rmse = get_rmse(output_array,truth_array)
            if array_norm_rmse > 0:
                logging.warning(f"Array had a root mean square error of {array_rmse}, range of {array_range}, and a normalized RMSE of {array_norm_rmse}")
        else:
            logging.error(f"# of array points in output ({len(output_array)}) vs truth ({len(truth_array)})")
            array_length_same = False
        
        return array_length_same, array_norm_rmse
    
    def bhattacharyya_distance(arr1,arr2,num_bins):

        min_val = int(np.floor(min(arr1.min(),arr2.min())))
        max_val = int(np.ceil(max(arr1.max(),arr2.max())))
        hist1,_ = np.histogram(arr1,bins=num_bins,range=(min_val,max_val))
        hist2,_ = np.histogram(arr2,bins=num_bins,range=(min_val,max_val))
        norm_hist1 = hist1 / np.sum(hist1)
        norm_hist2 = hist2 / np.sum(hist2)

        logging.info('Calculating Bhattacharyya Distance')
        s1 = np.sum(norm_hist1)
        s2 = np.sum(norm_hist2)
        s1 *= s2
        result = np.sum(np.sqrt(norm_hist1*norm_hist2))

        if abs(s1) > np.finfo(np.float32).tiny:
            s1 = 1/np.sqrt(s1)
        else:
            s1 = 1

        bhatt_distance = np.sqrt(np.max(1-(s1*result),0))
        logging.info(f"Bhattacharyya distance : {bhatt_distance}")

        return bhatt_distance

    def dice_coefficient(output_array,truth_array,abs_tolerance=1e-6,rel_tolerance=0):
        logging.info('Calculating dice coefficient')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")
        
        if output_array.dtype == bool:
            matches = output_array == truth_array
        else:
            matches = np.isclose(output_array,truth_array,atol=abs_tolerance,rtol=rel_tolerance)
        matches_count = len(matches[matches==True])

        dice_coeff = 2 * matches_count / (output_array.size + truth_array.size)
        logging.info(f"DICE Coefficient: {dice_coeff}")
        return dice_coeff
    
    def mse(output_array,truth_array):
        logging.info('Calculating mean square error')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")
        
        mean_square_error = mean_squared_error(output_array, truth_array)
        return mean_square_error
    
    def ssim(output_array,truth_array,win_size=7,data_range=None):
        logging.info('Calculating structural similarity')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")

        score = structural_similarity(output_array, truth_array, win_size=win_size,data_range=data_range)
        return score
    
    def stl_area(output_stl,truth_stl):
        logging.info('Calculating percent error of stl area')
        
        # Check STL area
        percent_error_area = abs((output_stl.area - truth_stl.area)/truth_stl.area)
        if percent_error_area > 0:
            logging.warning(f"STL area had a percent error of {percent_error_area*100}%")
        else:
            logging.info(f"STL area is identical ({output_stl.area})")
        
        return percent_error_area

    # Return the fixture object with the specified attribute
    return {'array_data': array_data,'bhatt_distance': bhattacharyya_distance,'dice_coefficient': dice_coefficient,'mse': mse,'ssim': ssim,'stl_area': stl_area}

@pytest.fixture()
def extract_nib_info():
    def _extract_nib_info(nifti_nib):
        zooms = np.asarray(nifti_nib.header.get_zooms())[:3]
        affine = nifti_nib.affine
        data = np.squeeze(nifti_nib.get_fdata())

        return zooms, affine, data
    
    return _extract_nib_info

@pytest.fixture()
def extract_sitk_info():
    def _extract_sitk_info(nifti_sitk):
        spacing = np.asarray(nifti_sitk.GetSpacing())
        direction = np.asarray(nifti_sitk.GetDirection())
        origin = np.asarray(nifti_sitk.GetOrigin())
        data = sitk.GetArrayFromImage(nifti_sitk)

        return spacing, direction, origin, data
    
    return _extract_sitk_info

@pytest.fixture()
def image_to_base64():
    def _image_to_base64(image_path):
        # Ensure the file exists
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"File {image_path} does not exist.")
        
        # Open the image file in binary mode
        with image_path.open("rb") as image_file:
            # Read the binary data from the file
            image_data = image_file.read()
            
            # Encode the binary data to a base64 string
            base64_string = base64.b64encode(image_data).decode('utf-8')
        
        return base64_string
    
    return _image_to_base64

@pytest.fixture()
def get_mpl_plot():
    def _get_mpl_plot(datas,axes_num=1,titles=None, color_map='viridis'):

        data_num = len(datas)
        fig, axs = plt.subplots(axes_num, data_num, figsize = (data_num * 2.5, axes_num * 2.5))

        for axis in range(axes_num):
            for num in range(data_num):
                midpoint = datas[num].shape[axis]//2

                try:
                    if axis == 0:
                        axs[axis,num].imshow(np.rot90(datas[num][midpoint,:,:]), cmap=color_map)
                    elif axis == 1:
                        axs[axis,num].imshow(np.rot90(datas[num][:,midpoint,:]), cmap=color_map)
                    else:
                        axs[axis,num].imshow(datas[num][:,:,midpoint], cmap=color_map)

                    if titles is not None and axis == 0:
                        axs[axis,num].set_title(titles[num])
                except:
                    if axis == 0:
                        axs[num].imshow(np.rot90(datas[num][midpoint,:,:]), cmap=color_map)
                    elif axis == 1:
                        axs[num].imshow(np.rot90(datas[num][:,midpoint,:]), cmap=color_map)
                    else:
                        axs[num].imshow(datas[num][:,:,midpoint], cmap=color_map)

                    if titles is not None and axis == 0:
                        axs[num].set_title(titles[num],fontsize=14)

        # Adjust plots
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
        plt.tight_layout()

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_plot
    
    return _get_mpl_plot

@pytest.fixture
def get_pyvista_plot():

    def intersection_plot(mesh1,mesh2,mesh3):
        # Create pyvista plot
        plotter = pv.Plotter(window_size=(400, 400),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(mesh1, opacity=0.2)
        plotter.add_mesh(mesh2, opacity=0.2)
        plotter.add_mesh(mesh3, opacity=0.5, color='red')

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_plot
    
    def mesh_plot(mesh,title=''):

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(pv.wrap(mesh),opacity=0.5)
        plotter.add_title(title, font_size=12)
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)

        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_plot
    
    def voxel_plot(mesh,Points,title=''):
        # Create points mesh
        step = Points.shape[0]//1000000 # Plot 1000000 points
        points_mesh =  pv.PolyData(Points[::step,:])

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(pv.wrap(mesh),opacity=0.5)
        plotter.add_mesh(points_mesh,color='blue',opacity=0.1)
        plotter.add_title(title,font_size=12)
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)

        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_plot
    
    # Return the fixture object with the specified attribute
    return {'intersection_plot': intersection_plot,'mesh_plot': mesh_plot,'voxel_plot': voxel_plot}

@pytest.fixture()
def selfiles_widget(qtbot):

    sf_widget = SelFiles()
    sf_widget.show()
    qtbot.addWidget(sf_widget) # qtbot will handle sf_widget teardown

    return sf_widget

@pytest.fixture()
def get_freq():

    def _get_freq(tx):
        if tx == 'Single':
            freq = '400'
        elif tx in ['CTX_500','DPX_500','H246','R15148']:
            freq = '500'
        elif tx == 'CTX_250':
            freq = '250'
        elif tx == 'H317':
            freq = '250'
        elif tx in ['BSonix','I12378']:
            freq = '650'
        elif tx in ['ATAC']:
            freq = '1000'
        elif tx == 'REMOPD':
            freq = '300'
        return freq

    return _get_freq

@pytest.fixture()
def get_extra_scan_file():
    def _get_extra_scan_file(extra_scan_type,ds_folder_path):
        scan_file_path = ""

        if extra_scan_type != 'NONE':
            scan_file_path = ds_folder_path + os.sep + extra_scan_type + '.nii.gz'

            if not os.path.exists(scan_file_path):
                pytest.skip(f"{ds_folder_path} does not possess a {extra_scan_type} file")

        return scan_file_path
    
    return _get_extra_scan_file

@pytest.fixture()
def babelbrain_widget(qtbot,trajectory_type,
                      scan_type,
                      trajectory,
                      dataset,
                      transducer,
                      selfiles_widget,
                      get_freq,
                      get_extra_scan_file,
                      tmp_path):

    # Folder paths
    input_folder = dataset['folder_path']
    simNIBS_folder = dataset['m2m_folder_path']
    trajectory_folder = input_folder + 'Trajectories' + os.sep

    # Filenames
    T1W_file = dataset['T1_path']
    if scan_type != 'NONE':
        CT_file = get_extra_scan_file(scan_type,input_folder)
    thermal_profile_file = thermal_profiles['thermal_profile_1']
    trajectory_file = trajectory_folder + f"{trajectory}.txt"

    # Set SelFiles Parameters
    selfiles_widget.ui.TrajectoryTypecomboBox.setCurrentIndex(test_trajectory_type[trajectory_type])
    selfiles_widget.ui.TrajectorylineEdit.setText(trajectory_file)
    selfiles_widget.ui.SimbNIBSTypecomboBox.setCurrentIndex(SimNIBS_type['charm'])
    selfiles_widget.ui.SimbNIBSlineEdit.setText(simNIBS_folder)
    selfiles_widget.ui.T1WlineEdit.setText(T1W_file)
    selfiles_widget.ui.CTTypecomboBox.setCurrentIndex(CT_types[scan_type])
    if scan_type != 'NONE':
        selfiles_widget.ui.CoregCTcomboBox.setCurrentIndex(coregistration['yes'])
        selfiles_widget.ui.CTlineEdit.setText(CT_file)
    selfiles_widget.ui.ThermalProfilelineEdit.setText(thermal_profile_file)
    selfiles_widget.ui.TransducerTypecomboBox.setCurrentIndex(transducer['dropdown_index'])
    selfiles_widget.ui.ContinuepushButton.click()

    # Create BabelBrain widget
    os.environ['BABEL_PYTEST']='1'
    bb_widget = BabelBrain(selfiles_widget,AltOutputFilesPath=str(tmp_path))
    bb_widget.show()
    qtbot.addWidget(bb_widget) # qtbot will handle bb_widget teardown

    # Copy T1W file and additional scan over to temporary folder
    shutil.copy(bb_widget.Config['T1W'],os.path.join(tmp_path,os.path.basename(bb_widget.Config['T1W'])))
    if scan_type != 'NONE':
        shutil.copy(CT_file,os.path.join(tmp_path,os.path.basename(CT_file)))

    # Copy SimbNIBs input file over to temporary folder
    if bb_widget.Config['SimbNIBSType'] == 'charm':
        SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'final_tissues.nii.gz'
    else:
        SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'skin.nii.gz'
    os.makedirs(os.path.join(tmp_path,os.path.basename(os.path.dirname(bb_widget.Config['simbnibs_path']))),exist_ok=True)
    shutil.copy(SimbNIBSInput,os.path.join(tmp_path,re.search('m2m.*',SimbNIBSInput)[0]))

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
    freq = get_freq(transducer['name'])

    freq_index = bb_widget.Widget.USMaskkHzDropDown.findText(freq)

    bb_widget.Widget.USMaskkHzDropDown.setCurrentIndex(freq_index)
    bb_widget.Widget.USPPWSpinBox.setProperty('UserData',6) # 6 PPW
    if scan_type != 'NONE':
        bb_widget.Widget.HUThresholdSpinBox.setValue(300)

    yield bb_widget
    
    # Teardown Code
    if tmp_path.exists():
        shutil.rmtree(tmp_path) # Remove all files created in tmp folder

    os.environ.pop('BABEL_PYTEST')

@pytest.fixture()
def set_up_file_manager(load_files,tmpdir,get_example_data,get_extra_scan_file):

    def existing_dataset(ds,extra_scan_type="NONE",HUT=300.0,pCT_range=(0.1,0.6)):
        T1_iso_path = ds['folder_path'] + f"T1W-isotropic.nii.gz"
        extra_scan_path = get_extra_scan_file(extra_scan_type,ds['folder_path'])
        prefix = ""

        # Instantiate FileManager class object
        file_manager = FileManager(ds['m2m_folder_path'],
                                   ds['simbNIBS_type'],
                                   ds['T1_path'],
                                   T1_iso_path,
                                   extra_scan_path,
                                   prefix,
                                   CT_types[extra_scan_type],
                                   current_HUT=HUT,
                                   current_pCT_range=pCT_range)
        
        # Load T1 using nibabel and save to file manager
        file_manager.saved_objects['T1_nib'] = load_files([ds['T1_path']],nifti_load_method='nibabel')[0]

        # Load T1 using sitk and save to file manager
        file_manager.saved_objects['T1_sitk'] = load_files([ds['T1_path']],nifti_load_method='sitk')[0]

        return file_manager
    
    def blank(CT_type='NONE',HUT=300.0,pCT_range=(0.1,0.6)):

        # Instantiate blank FileManager class object
        file_manager = FileManager(simNIBS_dir="",
                                   simbNIBS_type="",
                                   T1_fname="",
                                   T1_iso_fname="",
                                   extra_scan_fname="",
                                   prefix="",
                                   current_CT_type=CT_types[CT_type],
                                   current_HUT=HUT,
                                   current_pCT_range=pCT_range)
        
        # Set file paths
        input_1_path = str(tmpdir.join('input1.npy'))
        output_1_path = str(tmpdir.join('output1.nii.gz'))
        output_2_path = str(tmpdir.join('output2.nii.gz'))
        input_fnames = {'input1': input_1_path}
        output_fnames = {'output1': output_1_path,'output2': output_2_path}

        # Get random example data
        example_input_data = get_example_data['numpy']()

        # Save input files
        if not os.path.exists(input_1_path):
            np.save(input_1_path,example_input_data)

        return file_manager, input_fnames, output_fnames
    
    return {'existing_dataset':existing_dataset,'blank':blank}

@pytest.fixture()
def get_example_data():
    def numpy_data(dims = (4,4)):
        return np.random.random(dims)
    
    def nifti_nib_data(dims=(256,256,128)):
        affine = np.random.rand(4,4)
        data = np.random.random(dims)
        nibabel_nifti = nibabel.nifti1.Nifti1Image(data,affine)
        nibabel_nifti.header.set_zooms(np.random.rand(3))
        
        return nibabel_nifti
    
    def nifti_sitk_data():
        data = np.random.rand(256,256,128)
        nibabel_sitk = sitk.GetImageFromArray(data)

        # Set the spacing, direction, and origin in the SimpleITK image
        nibabel_sitk.SetSpacing(np.random.rand(3))
        nibabel_sitk.SetDirection(np.random.rand(3,3))
        nibabel_sitk.SetOrigin(np.random.rand(3))
        
        return nibabel_sitk
    
    # Return the fixture object with the specified attribute
    return {'numpy': numpy_data,
            'nifti_nib':nifti_nib_data,
            'nifti_sitk':nifti_sitk_data}
        
# PYTEST HOOKS
def pytest_generate_tests(metafunc):
    # Parametrize tests based on arguments
    if 'trajectory_type' in metafunc.fixturenames:
        metafunc.parametrize('trajectory_type', tuple(test_trajectory_type)) 

    if 'scan_type' in metafunc.fixturenames:
        metafunc.parametrize('scan_type',tuple(CT_types.keys()))

    if 'second_scan_type' in metafunc.fixturenames:
        metafunc.parametrize('second_scan_type',tuple(CT_types.keys()))

    if 'trajectory' in metafunc.fixturenames and 'invalid' in metafunc.function.__name__:
        metafunc.parametrize('trajectory', tuple(invalid_trajectories))
    elif 'trajectory' in metafunc.fixturenames and ('valid' in metafunc.function.__name__ or
                                                    'normal' in metafunc.function.__name__):
        metafunc.parametrize('trajectory', tuple(valid_trajectories)) 
    
    if 'dataset' in metafunc.fixturenames:
        metafunc.parametrize('dataset',tuple(test_datasets),ids=tuple(ds['id'] for ds in test_datasets))
    
    if 'transducer' in metafunc.fixturenames:
        metafunc.parametrize('transducer', tuple(transducers),ids=tuple(tx['name'] for tx in transducers))
    
    if 'computing_backend' in metafunc.fixturenames:
        metafunc.parametrize('computing_backend',tuple(computing_backends),ids=tuple(cb['type'] for cb in computing_backends))

    if 'spatial_step' in metafunc.fixturenames:
        metafunc.parametrize('spatial_step',tuple(spatial_step.values()),ids=tuple(spatial_step.keys()))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome = yield
    report = outcome.get_result()
    
    if report.when == 'call':
        extras = getattr(report, 'extras', [])

        # Add saved screenshots to html report
        if hasattr(item, 'screenshots'):
            img_tags = ''
            for screenshot in item.screenshots:
                img_tags += "<td><img src='data:image/png;base64,{}'></td>".format(screenshot)
            extras.append(pytest_html.extras.html(f"<tr>{img_tags}</tr>"))
            
        report.extras = extras

@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Hook to modify inline final report"""
    terminalreporter.write_line(f"Total tests run: {terminalreporter._numcollected}")
    terminalreporter.write_line(f"Total failures: {len(terminalreporter.stats.get('failed', []))}")
    terminalreporter.write_line(f"Total passes: {len(terminalreporter.stats.get('passed', []))}")

    if os.path.isfile(os.path.join('PyTest_Reports','report.html')):
    # Change report name to include time of completion
        report_name = f"report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
        os.rename(os.path.join('PyTest_Reports','report.html'), os.path.join('PyTest_Reports',report_name))
        print(f"Report saved as {report_name}")