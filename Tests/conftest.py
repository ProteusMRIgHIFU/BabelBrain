
import datetime
import os
import sys
sys.path.append('./BabelBrain/')
import platform
import shutil
import re
import configparser
import logging

import base64
import h5py
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is noninteractive
import matplotlib.pyplot as plt
import nibabel
from nibabel import processing, nifti1, affines
import numpy as np
np.random.seed(42) # RNG is same every time
from PySide6.QtCore import Qt
import pytest
import pytest_html
import pyvista as pv
import SimpleITK as sitk
from skimage.metrics import structural_similarity, mean_squared_error
import trimesh

from BabelBrain.BabelBrain import BabelBrain
from BabelBrain.SelFiles.SelFiles import SelFiles
from BabelBrain.FileManager import FileManager

# ================================================================================================================================
# FOLDER/FILE PATHS
# ================================================================================================================================
config = configparser.ConfigParser()
config.read('Tests' + os.sep + 'config.ini')
gpu_device = config['GPU']['device_name']               # GPU device used for test
print('Using GPU device: ',gpu_device)
test_data_folder = config['Paths']['data_folder_path']  # Folder containing input test data
ref_output_dir = config['Paths']['ref_output_folder_1']   # Folder containing previously generated BabelBrain outputs. Used in regression tests
ref_output_dir_2 = config['Paths']['ref_output_folder_2']   # Folder containing previously generated BabelBrain outputs. Used in test_full_pipeline_two_outputs tests
gen_output_dir = config['Paths']['gen_output_folder']   # Folder to store newly generated BabelBrain outputs. Used for generate_outputs "test"
REPORTS_DIR = "PyTest_Reports"

# ================================================================================================================================
# PARAMETERS
# ================================================================================================================================
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
    {'id': 'ID_0082' ,'folder_path': test_data_folder + os.sep + 'ID_0082'  + os.sep}
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
    {'name': 'Single',      'dropdown_index': 0,  'diameter': 0, 'freqs':[200000.0,250000.0,300000.0,350000.0,400000.0,450000.0,500000.0,550000.0,600000.0,650000.0,700000.0,750000.0,800000.0,850000.0,900000.0,950000.0,1000000.0]}, # EDIT DIAMETER
    {'name': 'CTX_500',     'dropdown_index': 1,  'diameter': 0, 'freqs':[500000.0,545000.0]},
    {'name': 'CTX_250',     'dropdown_index': 2,  'diameter': 0, 'freqs':[250000.0]},
    {'name': 'CTX_250_2ch', 'dropdown_index': 3,  'diameter': 0, 'freqs':[250000.0]},
    {'name': 'DPX_500',     'dropdown_index': 4,  'diameter': 0, 'freqs':[500000.0]},
    {'name': 'DPXPC_300',   'dropdown_index': 5,  'diameter': 0, 'freqs':[300000.0]},
    {'name': 'H317',        'dropdown_index': 6,  'diameter': 0, 'freqs':[250000.0,700000.0,825000.0]},
    {'name': 'H246',        'dropdown_index': 7,  'diameter': 0, 'freqs':[500000.0]},
    {'name': 'BSonix',      'dropdown_index': 8,  'diameter': 0, 'freqs':[650000.0]},
    {'name': 'REMOPD',      'dropdown_index': 9,  'diameter': 0, 'freqs':[300000.0,480000.0,490000.0]},
    {'name': 'I12378',      'dropdown_index': 10, 'diameter': 0, 'freqs':[650000.0]},
    {'name': 'ATAC',        'dropdown_index': 11, 'diameter': 0, 'freqs':[1000000.0]},
    {'name': 'R15148',      'dropdown_index': 12, 'diameter': 0, 'freqs':[500000.0]},
    {'name': 'R15287',      'dropdown_index': 13, 'diameter': 0, 'freqs':[300000.0]},
    {'name': 'R15473',      'dropdown_index': 14, 'diameter': 0, 'freqs':[300000.0]},
    {'name': 'R15646',      'dropdown_index': 15, 'diameter': 0, 'freqs':[650000.0]},
    {'name': 'IGT64_500',   'dropdown_index': 16, 'diameter': 0, 'freqs':[500000.0]}
]
computing_backends = [
    # {'type': 'CPU','supported_os': ['Mac','Windows','Linux']},
    {'type': 'OpenCL','supported_os': ['Windows','Linux']},
    {'type': 'CUDA',  'supported_os': ['Windows','Linux']},
    {'type': 'Metal', 'supported_os': ['Mac']},
    {'type': 'MLX',   'supported_os': ['Mac']} # Linux too?
]
spatial_step = {
    'Low_Res': 0.919,  # 200 kHz,   6 PPW
    'Med_Res': 0.306,  # 600 kHz,   6 PPW
    'High_Res': 0.184,  # 1000 kHz,  6 PPW
    'Stress_Res': 0.092,  # 1000 kHz, 12 PPW
}

# ================================================================================================================================
# PYTEST FIXTURES
# ================================================================================================================================
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
    return gpu_device

@pytest.fixture(scope="session")
def get_config_dirs():
    config_dirs = {}
    config_dirs["test_data_dir"] = test_data_folder
    config_dirs["ref_dir_1"] = ref_output_dir
    config_dirs["ref_dir_2"] = ref_output_dir_2
    config_dirs["gen_output_dir"] = gen_output_dir
    return config_dirs

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
    
    def bhattacharyya_coefficient(arr1,arr2,num_bins=None):

        # Check arrays are not empty
        if arr1.size == 0 or arr2.size == 0:
            pytest.fail("One or both arrays are empty")

        # Determine range of values. We extended the range slightly so bins are divided at 0.5 marks 
        # instead of 1.0 (e.g. -0.5, 0.5, 1.5,...) as array values are more likely to exist at integer 
        # values and helps prevent errors when values lie exactly at bin edge
        min_val = int(np.floor(min(arr1.min(),arr2.min()))) - 0.5
        max_val = int(np.ceil(max(arr1.max(),arr2.max()))) + 0.5
        logging.debug(f"Using {min_val} to {max_val} range for bhatt coeff calculation")
        
        
        # Determine number of bins if argument is not supplied
        if num_bins is None:
            num_bins = int(max_val - min_val)
        logging.debug(f"Using {num_bins} bins for bhatt coeff calculation")
        
        # Get and normalize histograms
        hist1,_ = np.histogram(arr1,bins=num_bins,range=(min_val,max_val))
        hist2,_ = np.histogram(arr2,bins=num_bins,range=(min_val,max_val))
        norm_hist1 = hist1 / np.sum(hist1)
        norm_hist2 = hist2 / np.sum(hist2)

        # Compute Bhattacharyya coefficient
        logging.info('Calculating Bhattacharyya Coefficient')
        bhatt_coefficent = np.sum(np.sqrt(norm_hist1 * norm_hist2))
        logging.info(f"Bhattacharyya coefficient : {bhatt_coefficent}")

        return bhatt_coefficent

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
    
    def h5_data(h5_ref_path,h5_test_path,tolerance=0):
        mismatches = []
        
        def compare_items(name, obj1):
            if name not in f2:
                logging.warning(f"{name} missing in test file")
                mismatches.append(name)
                return
            obj2 = f2[name]
            if isinstance(obj1, h5py.Dataset):
                data1, data2 = obj1[()], obj2[()]
                if not np.allclose(data1, data2, rtol=tolerance, atol=0,equal_nan=True):
                    if data1.size > 1:
                        logging.warning(f"Dataset {name} differs")
                    else:
                        logging.warning(f"Dataset {name} differs: {data1} vs {data2}")
                    mismatches.append(name)
            elif isinstance(obj1, h5py.Group):
                pass  # groups are containers, children checked recursively
                
        with h5py.File(h5_ref_path, "r") as f1, h5py.File(h5_test_path, "r") as f2:
            exact_match = f1.visititems(lambda name, obj: compare_items(name, obj1=obj))
            
        return len(mismatches) == 0
    
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
    return {'array_data': array_data,'bhatt_coeff': bhattacharyya_coefficient,'dice_coefficient': dice_coefficient,'h5_data': h5_data,'mse': mse,'ssim': ssim,'stl_area': stl_area}

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
    
    def mesh_plot(meshes,title=''):

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        for mesh in meshes:
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
def get_freq():
    def _get_default_freq(tx):
        tx = tx['name']
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
    
    def _get_low_freq(tx):
        return tx['freqs'][0]
        
    def _get_high_freq(tx):
        return tx['freqs'][-1]

    return {'default': _get_default_freq,'low': _get_low_freq,'high': _get_high_freq}

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
def selfiles_widget(qtbot):

    sf_widget = SelFiles()
    sf_widget.show()
    qtbot.addWidget(sf_widget) # qtbot will handle sf_widget teardown

    yield sf_widget
    
    sf_widget.close()
    sf_widget.deleteLater()

@pytest.fixture()
def babelbrain_widget(request,qtbot,
                      trajectory_type,
                      scan_type,
                      trajectory,
                      dataset,
                      transducer,
                      selfiles_widget,
                      frequency,
                      get_extra_scan_file,
                      computing_backend,
                      check_os,
                      load_files,
                      tmp_path):
    created_widgets = []
    
    def _babelbrain_widget(generate_outputs=False):
        
        # Convert frequency to string
        freq = str(int(frequency/1000)
                   )
        # Folder paths
        input_folder = dataset['folder_path']
        simNIBS_folder = dataset['m2m_folder_path']
        trajectory_folder = input_folder + 'Trajectories' + os.sep
        if generate_outputs:
            if not os.path.exists(gen_output_dir):
                pytest.fail(f"output folder does not exist:\n{gen_output_dir}")
            output_folder = gen_output_dir + f"{os.sep}{trajectory_type}_CT={scan_type}_{trajectory}_{transducer['name']}_Freq={freq}kHz_{computing_backend['type']}{os.sep}"
        else:
            output_folder = str(tmp_path) + f"{os.sep}{trajectory_type}_CT={scan_type}_{trajectory}_{transducer['name']}_Freq={freq}kHz_{computing_backend['type']}{os.sep}"

        try:
            os.makedirs(output_folder)
        except:
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
                
        # Filenames
        T1W_file = dataset['T1_path']
        if scan_type != 'NONE':
            CT_file = get_extra_scan_file(scan_type,input_folder)
        thermal_profile_file = thermal_profiles['thermal_profile_1']
        trajectory_file = trajectory_folder + f"{trajectory_type}_{dataset['id']}_{trajectory}.txt"
        load_files([trajectory_file]) # Use to ensure trajectory file exists otherwise skip

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
        cb_index = selfiles_widget.ui.ComputingEnginecomboBox.findText(computing_backend['type'],Qt.MatchContains)
        selfiles_widget.ui.ComputingEnginecomboBox.setCurrentIndex(cb_index)
        if selfiles_widget.ui.MultiPointTypecomboBox.isEnabled():
            selfiles_widget.ui.MultiPointTypecomboBox.setCurrentIndex(0) # Only single focus
            # selfiles_widget.ui.MultiPointlineEdit.setText() # Fill out when implementing test for multi-point sims
        selfiles_widget.ui.ContinuepushButton.click()

        # Create BabelBrain widget
        os.environ['BABEL_PYTEST']='1'
        bb_widget = BabelBrain(selfiles_widget,AltOutputFilesPath=str(output_folder))
        bb_widget.show()
        qtbot.addWidget(bb_widget) # qtbot will handle bb_widget teardown
        created_widgets.append(bb_widget)

        # Copy T1W file and additional scan over to output folder
        # Not needed?
        shutil.copy(bb_widget.Config['T1W'],os.path.join(output_folder,os.path.basename(bb_widget.Config['T1W'])))
        if scan_type != 'NONE':
            shutil.copy(CT_file,os.path.join(output_folder,os.path.basename(CT_file)))

        # Copy SimbNIBs input file over to output folder
        if bb_widget.Config['SimbNIBSType'] == 'charm':
            SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'final_tissues.nii.gz'
        else:
            SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'skin.nii.gz'
        os.makedirs(os.path.join(output_folder,os.path.basename(os.path.dirname(bb_widget.Config['simbnibs_path']))),exist_ok=True)
        shutil.copy(SimbNIBSInput,os.path.join(output_folder,re.search('m2m.*',SimbNIBSInput)[0]))

        # Copy Trajectory file over to output folder
        trajectory_new_file = os.path.join(output_folder,os.path.basename(bb_widget.Config['Mat4Trajectory']))
        shutil.copy(bb_widget.Config['Mat4Trajectory'],trajectory_new_file)
        bb_widget.Config['ID'] = os.path.splitext(os.path.basename(bb_widget.Config['Mat4Trajectory']))[0] # Affects trajectory naming in output files

        # Edit file paths so new data is saved in output folder
        bb_widget.Config['Mat4Trajectory'] = trajectory_new_file
        bb_widget.Config['T1WIso'] = os.path.join(output_folder,os.path.basename(bb_widget.Config['T1WIso']))
        bb_widget.Config['simbnibs_path'] = os.path.join(output_folder,os.path.split(os.path.split(bb_widget.Config['simbnibs_path'])[0])[1])
        if bb_widget.Config['bUseCT']:
            bb_widget.Config['CT_or_ZTE_input'] = os.path.join(output_folder,os.path.basename(bb_widget.Config['CT_or_ZTE_input']))

        # Set Sim Parameters
        freq_index = bb_widget.Widget.USMaskkHzDropDown.findText(freq)

        bb_widget.Widget.USMaskkHzDropDown.setCurrentIndex(freq_index)
        bb_widget.Widget.USPPWSpinBox.setProperty('UserData',6) # 6 PPW
        if scan_type != 'NONE':
            bb_widget.Widget.HUThresholdSpinBox.setValue(300)
        
        return bb_widget

    yield _babelbrain_widget

    for w in created_widgets:
        w.close()
        w.deleteLater()
    
    if tmp_path.exists():
        shutil.rmtree(tmp_path) # Remove all files created in tmp folder
    
    if 'BABEL_PYTEST' in os.environ:
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
        
# ================================================================================================================================
# PYTEST HOOKS
# ================================================================================================================================
def pytest_generate_tests(metafunc):
    # Parametrize + mark tests based on fixtures used
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
        if 'frequency' in metafunc.fixturenames:
            # Parametrize both transducer and freq
            params = []
            for tx in transducers:
                for freq in tx['freqs']:
                    params.append(pytest.param(tx, freq, id=f"{tx['name']}-{int(freq/1000)}kHz"))
            metafunc.parametrize("transducer,frequency", params)
        else:
            # Only parametrize transducer
            metafunc.parametrize(
                "transducer",
                [pytest.param(tx, id=tx['name']) for tx in transducers]
            )
    
    if 'computing_backend' in metafunc.fixturenames:
        params = [pytest.param(cb, id=cb['type'], marks=pytest.mark.gpu) for cb in computing_backends]
        metafunc.parametrize("computing_backend", params)

    if 'spatial_step' in metafunc.fixturenames:
        # metafunc.parametrize('spatial_step',tuple(spatial_step.values()),ids=tuple(spatial_step.keys()))
        params = []
        for ss_key,ss_value in spatial_step.items():
            if "low" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=pytest.mark.low_res))
            elif "med" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=pytest.mark.medium_res))
            elif "high" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=[pytest.mark.slow,pytest.mark.high_res]))
            elif "stress" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=[pytest.mark.slow,pytest.mark.stress_res]))
            else:
                params.append(pytest.param(ss_value, id=ss_key))
        metafunc.parametrize('spatial_step',params)
        
    if 'tolerance' in metafunc.fixturenames:
        metafunc.parametrize('tolerance',
                             [pytest.param(0, marks=pytest.mark.tol_0, id="0%_tolerance"),
                              pytest.param(0.01, marks=pytest.mark.tol_1, id="1%_tolerance"),
                              pytest.param(0.05, marks=pytest.mark.tol_5, id="5%_tolerance")])

def pytest_collection_modifyitems(config, items):
    for item in items:
        # Add markers for basic babelbrain param tests
        if "Deep_Target" in item.name and \
            "ID_0082" in item.name and \
            "H317" in item.name and \
            ("NONE" in item.name or "CT" in item.name or "ZTE" in item.name) and \
            ("250kHz" in item.name or "825kHz" in item.name):
            item.add_marker(pytest.mark.basic_babelbrain_params)
            
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
                img_tags += "<td><img src='data:image/png;base64,{}' width='500'>></td>".format(screenshot)
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