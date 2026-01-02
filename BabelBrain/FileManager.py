from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import glob
import hashlib
import logging
import os
import re
import threading

import nibabel
import numpy as np
import SimpleITK as sitk
import trimesh

MAX_WORKERS = os.cpu_count() * 4
logger = logging.getLogger()

class FileManager:
    def __init__(self, simNIBS_dir, simbNIBS_type, T1_fname, T1_iso_fname, extra_scan_fname, prefix, current_CT_type,coreg=None, current_HUT=None, current_pCT_range=None, max_workers=MAX_WORKERS):
        self._lock = threading.Lock()
        self._saving_file = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.CT_type = current_CT_type
        self.HU_threshold = None
        self.pseudo_CT_range = None
        self.simNIBS_dir = simNIBS_dir
        self.simNIBS_type = simbNIBS_type
        self.T1_fname = T1_fname
        self.T1_iso_fname = T1_iso_fname
        self.extra_scan_fname = extra_scan_fname
        self.prefix = prefix
        self.coreg = coreg
        self.saved_objects = {}

        if current_CT_type != 0: # Not T1W Only
            self.HU_threshold = current_HUT

            if current_CT_type > 1: # Pseudo CT Methods
                self.pseudo_CT_range = current_pCT_range

        input_files, output_files = self._generate_filenames()
        self.input_files = input_files
        self.output_files = output_files

    def _return_copy(self,value,nifti_load_method='nibabel',sitk_dtype=None):

        # Handle nibabel NIfti images 
        if isinstance(value, nibabel.Nifti1Image):

            if nifti_load_method == 'sitk':
                return self.nibabel_to_sitk(value,sitk_dtype=sitk_dtype) # nibabel_to_sitk automatically makes a copy
            else:
                data_copy = nibabel.Nifti1Image(value.get_fdata(), value.affine, value.header.copy())
                return data_copy
        
        # Handle sitk NIfti images 
        elif isinstance(value,sitk.Image):
            if nifti_load_method == 'nibabel':
                return self.sitk_to_nibabel(value) # sitk_to_nibabel automatically makes a copy
            else:
                data_copy = sitk.Image(value)
                return data_copy

        # Handles trimesh meshes
        elif isinstance(value, trimesh.Trimesh):
            return value.copy()
        
        # Handle NumPy arrays
        elif isinstance(value, np.ndarray):
            return np.copy(value)  # Shallow copy of NumPy array

        # Handle plain text (string)
        elif isinstance(value, str):
            return value  # Strings are immutable, return directly
        
        else:
            logger.warning('Loading unknown type of object')
            # For any other type, fallback to deep copy
            return copy.deepcopy(value)

    def load_file(self, filename, **kwargs):
        logger.info(f"{filename} started loading")

        file_saved = self.saved_objects.get(filename)

        if file_saved:
            data_copy = self._return_copy(file_saved,**kwargs)
            logger.info(f"{filename} saved object returned instead")
            return data_copy
        else:
            # Determine file extension
            ext = self.get_file_type(filename)

            # Load file using appropriate method
            if ext == '.npy' or ext == '.npz':
                loaded_data = np.load(filename)
            elif ext == '.stl':
                loaded_data = trimesh.load_mesh(filename)
            elif ext == '.nii':
                loaded_data = self._load_nifti_file(filename,**kwargs)
            elif ext == '.txt':
                with open(filename, 'r') as file:
                    loaded_data = file.read()
            else:
                raise ValueError(f"Not able to load this file type ({ext})")
            
            logger.info(f"{filename} finished loading")
            return loaded_data
    
    def save_file(self, file_data, filename, precursor_files=None, **kwargs):
        logger.info(f"{filename} preparing to save")

        # Save object for future use
        self.saved_objects[filename] = file_data

        file_being_saved = self._saving_file.get(filename)

        if file_being_saved:
            logger.info(f"current {filename} save has been stopped, restarting save")
            self._saving_file.pop(filename, None)

        self.set_saving(filename)
        self._executor.submit(self._save_file, file_data, filename, precursor_files, **kwargs)
        # self._save_file(file_data, filename, precursor_files,  **kwargs)

    def wait_for_file(self, filename):
        # Check if file is currently being saved and wait if it is
        with self._lock:
            saving = self._saving_file.get(filename)
            if saving:
                logger.info(f"{filename} needs to save first")
                saving.wait()

    def _file_saving(self, filename):
        # Check if file is currently being saved and wait if it is
        with self._lock:
            saving = self._saving_file.get(filename)
            if saving:
                return True
            else:
                return False

    def set_saving(self, filename):
        with self._lock:
            self._saving_file[filename] = threading.Condition(self._lock)

    def notify_saving_complete(self, filename):
        with self._lock:
            file_saved = self._saving_file.pop(filename, None)
            if file_saved:
                file_saved.notify_all()

    def get_file_type(self,filename):
        # Determine file extension
        base, extension = os.path.splitext(filename)
        # Extra check for compressed files
        if extension == '.gz':
            base, extension = os.path.splitext(base)

        return extension.lower()
    
    def generate_hash(self,filename):
        # Check file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No such file: '{filename}'")
    
        # Create a Blake2s hash object of size 4 bytes
        blake2s_hash = hashlib.blake2s(digest_size=4)
        
        try:
            # Open the file in binary mode and read it in chunks
            with open(filename, 'rb') as file:
                while True:
                    data = file.read(65536)  # Read data in 64KB chunks
                    if not data:
                        break
                    blake2s_hash.update(data)
        except OSError as e:
            raise RuntimeError(f"An error occurred while reading the file: {e}")
        
        # Return string that is double the hash object size (i.e. 8 bytes)
        return blake2s_hash.hexdigest()
    
    def generate_checksum(self, precursor_files):
        # CT type info to be saved
        if self.CT_type == 0:
            saved_CT_type = ""
        elif self.CT_type == 1:
            saved_CT_type = "CTType=CT,"
        elif self.CT_type == 2:
            saved_CT_type = "CTType=ZTE,"
        elif self.CT_type == 3:
            saved_CT_type = "CTType=PETRA,"
        elif self.CT_type == 4:
            saved_CT_type = "CTType=Density,"
        else:
            logger.warning("hash saving method not set up for this pseudo CT type")
            saved_CT_type = "CTType=Unknown,"
        
        # HU Threshold info to be saved
        if self.HU_threshold is None:
            saved_HUT = ""
        else:
            saved_HUT = f"HUT={int(self.HU_threshold)},"

        # ZTE range info to be saved
        if self.pseudo_CT_range is None:
            saved_pCT_range = ""
        else:
            saved_pCT_range = f"pCTR={self.pseudo_CT_range},"

        # checksum info to be saved
        saved_hashes = ""
        for f in precursor_files:
            self.wait_for_file(f)
            saved_hashes += self.generate_hash(f)+","
        
        saved_info = (saved_CT_type + saved_HUT + saved_pCT_range + saved_hashes).encode('utf-8')
        
        return saved_info

    def check_reuse_files(self, input_fnames, output_fnames):
        ''' 
        This function grabs checksum information stored in the header of intermediate nifti files generated 
        in previous BabelBrain runs, and determines if those files can be reused to speed up current run.
        '''

        current_hashes = ""
        expected_hashes = []
        expected_CT_types = []
        expected_HU_thresholds = []
        expected_pCT_ranges = []
        prev_files = {}
        pattern_CT_type = "CTType=\w+"
        pattern_HU_threshold = "HUT=\d+"
        pattern_pCT_range = "pCTR=.+\)"

        if self.CT_type == 0:
            current_CT_type = ""
        elif self.CT_type == 1:
            current_CT_type = "CT"
        elif self.CT_type == 2:
            current_CT_type = "ZTE"
        elif self.CT_type == 3:
            current_CT_type = "PETRA"
        elif self.CT_type == 4:
            current_CT_type = "Density"
        else:
            logger.warning("checksum verification not set up for this pseudo CT type")

        if self.HU_threshold:
            current_HUT = int(self.HU_threshold)
        else:
            current_HUT = None

        if self.pseudo_CT_range:
            current_pCT_range = str(self.pseudo_CT_range)
        else:
            current_pCT_range = None

        # Grab checksum and parameter information from existing nifti file headers
        for key,file in output_fnames.items():
            prev_files[key] = output_fnames[key]

            # Check if files already exist
            if not os.path.isfile(file):

                # If file doesn't exist for current target, use existing file for a different target
                target = re.search("\w+(?=_[a-zA-Z0-9]+_[a-zA-Z0-9]+Hz.+)",file)

                if target is not None:
                    prev_target_search = re.sub(target.group(),"*",file)
                    prev_targets = glob.glob(prev_target_search)

                    if len(prev_targets) > 0:
                        prev_files[key] = prev_targets[0] # Use first equivalent file at different target 
                    else:
                        print(f"Previous files use different transducer, frequency, or PPW")
                        return False, output_fnames
                else:
                    print(f"Previous files don't exist")
                    return False, output_fnames

            # Load checksum info from existing files
            try:
                # Load stored information
                if ".nii.gz" in prev_files[key]:
                    prev_file = nibabel.load(prev_files[key])
                    description = str(prev_file.header['descrip'].astype('str'))
                elif "babelbrain_reuse.npy" in prev_files[key]:
                    prev_file = np.load(prev_files[key])
                    description = str(prev_file.astype('str'))
                else:
                    continue # No information stored in stl, npz, etc files
            except:
                print(f"{prev_files[key]} was corrupted")
                continue

            # Extract parameter information 
            CT_type = re.search(pattern_CT_type,description)
            HU_threshold = re.search(pattern_HU_threshold,description)
            pCT_range = re.search(pattern_pCT_range,description)

            if CT_type:
                expected_CT_types += [CT_type.group()]
                description = re.sub(pattern_CT_type + ",", "",description)

            if HU_threshold:
                expected_HU_thresholds += [HU_threshold.group()]
                description = re.sub(pattern_HU_threshold + ",", "",description)

            if pCT_range:
                expected_pCT_ranges += [pCT_range.group()]
                description = re.sub(pattern_pCT_range + ",", "",description)
            
            # Extract checksum information
            expected_hashes += [description]
        
        # Grab current checksum information for both input and output files
        for f in (input_fnames | prev_files).values():
            current_hashes += self.generate_hash(f)+","

        # Check all expected hashes match current hashes
        for expected_hash in expected_hashes:
            if expected_hash not in current_hashes:
                print(f"Missing expected hash")
                return False, output_fnames

        # Check all CT Type of previous files match current ones
        if current_CT_type and len(expected_CT_types) == 0:
            print(f"Previous files didn't use CT")
            return False, output_fnames
        else:
            for expected_CT_type in expected_CT_types:
                if expected_CT_type != f"CTType={current_CT_type}":
                    print(f"Previous files used different CT Type")
                    return False, output_fnames
                    
        # Check all HU Threshold of previous files match current ones
        if current_HUT and len(expected_HU_thresholds) == 0:
            print(f"Previous files didn't have HU Threshold value")
            return False, output_fnames
        else:
            for expected_HU_threshold in expected_HU_thresholds:
                if int(expected_HU_threshold[4:]) != current_HUT:
                    print(f"Previous files used different HUThreshold")
                    return False, output_fnames
            
        # Check all pseudo CT ranges of previous files match current ones
        if current_pCT_range and len(expected_pCT_ranges) == 0:
            print(f"Previous files didn't have ZTE/PETRA Range values")
            return False, output_fnames
        else:
            for expected_pCT_range in expected_pCT_ranges:
                if expected_pCT_range[5:] != current_pCT_range:
                    print(f"Previous files used different ZTE/PETRA Range")
                    return False, output_fnames
        
        print("Reusing previously generated files")
        return True, prev_files

    def nibabel_to_sitk(self,original_nib_image,sitk_dtype=None):
        """
        Convert a nibabel.Nifti1Image to a SimpleITK.Image with correct spacing, direction, and origin.

        Parameters:
        - nib_image: nibabel.Nifti1Image
            The NIfTI image loaded using nibabel.

        Returns:
        - sitk_img: SimpleITK.Image
            The corresponding SimpleITK image.
        """

        # Make copy of original image
        nib_image = nibabel.Nifti1Image(original_nib_image.get_fdata(), original_nib_image.affine, original_nib_image.header.copy())

        # Get the image data array and affine matrix from the nibabel image
        image_data = nib_image.get_fdata()
        if len(image_data.shape) > 3:
            image_data = np.squeeze(image_data, axis=-1)
        affine = nib_image.affine

        # Transpose image data from (x, y, z) to (z, y, x) for SimpleITK
        image_data = image_data.transpose(2,1,0)

        # Convert the NumPy array to a SimpleITK image
        sitk_img = sitk.GetImageFromArray(image_data)

        # Extract spacing from the affine matrix (norm of the columns)
        # spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        spacing = np.asanyarray(nib_image.header.get_zooms()[:3],np.float64)

        # Compute the direction cosines (columns of the affine normalized by spacing)
        direction = affine[:3, :3] / spacing
        direction[:2,:] *= -1
        
        # Compute the origin (translation part of the affine)
        origin = affine[:3, 3]
        origin[:2] *= -1

        # Set the spacing, direction, and origin in the SimpleITK image
        sitk_img.SetSpacing(spacing)
        sitk_img.SetDirection(direction.flatten())
        sitk_img.SetOrigin(origin)

        if sitk_dtype:
            sitk_img = sitk.Cast(sitk_img,sitk_dtype)

        return sitk_img

    def sitk_to_nibabel(self, original_sitk_img):
        """
        Convert a SimpleITK.Image to a nibabel.Nifti1Image with correct affine matrix.

        Parameters:
        - sitk_img: SimpleITK.Image
            The SimpleITK image to be converted.

        Returns:
        - nib_image: nibabel.Nifti1Image
            The corresponding nibabel NIfTI image.
        """

        # Make copy of original
        sitk_img = sitk.Image(original_sitk_img)

        # Get the image data as a NumPy array and transpose from (z, y, x) to (x, y, z)
        image_data = sitk.GetArrayFromImage(sitk_img)
        image_data = image_data.transpose(2, 1, 0)

        # Retrieve the spacing, direction, and origin from the SimpleITK image
        spacing = np.array(sitk_img.GetSpacing())
        direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
        origin = np.array(sitk_img.GetOrigin())

        # Recompute the affine matrix
        direction[:2, :] *= -1
        origin[:2] *= -1
        affine = np.eye(4)
        affine[:3, :3] = direction * spacing
        affine[:3, 3] = origin

        # Create a nibabel NIfTI image with the NumPy array and affine matrix
        nib_image = nibabel.Nifti1Image(image_data, affine)

        return nib_image

    def shutdown(self):
        logger.info("Waiting for ongoing tasks to complete...")
        self._executor.shutdown(wait=True)
        logger.info("FileManager has been shut down.")

    def _load_nifti_file(self, filename,nifti_load_method='nibabel',sitk_dtype=sitk.sitkFloat32):
        if nifti_load_method == 'nibabel':
            loaded_file = nibabel.load(filename)
        elif nifti_load_method == 'sitk':
            loaded_file = sitk.ReadImage(filename,sitk_dtype)
        else:
            raise ValueError('Invalid load method specified')
        
        return loaded_file
    
    def _save_file(self, file_data, filename, precursor_files=None, **kwargs):
        logger.info(f"{filename} saving started in separate thread")
        
        # Ensure precursor values is iterable if provided
        if precursor_files is not None:
            if not isinstance(precursor_files, (list,set,tuple)):
                if isinstance(precursor_files,(str)):
                    precursor_files = [precursor_files]
                else:
                    precursor_files = list(precursor_files)

        # Get file type
        ext = self.get_file_type(filename)

        # Generate checksum so that we can verify if this file can be reused for future sims
        if precursor_files:

            file_checksum = self.generate_checksum(precursor_files)

            if ext == '.npy':
                file_data = np.array(file_checksum)
            elif ext == '.nii':
                if len(file_checksum) <= 80: # Nifiti descrip header can only accept string < 80 bytes (80 utf-8 chars)
                    file_data.header['descrip'] = file_checksum
                else:
                    print(f"'descrip' string not saved in nifti header as it exceeds text limit ({len(file_checksum)} > 80)")  
            else:
                print("checksum data not saved, invalid output file type specified")      

        # Save file using appropriate method
        # logger.info(f"{filename} started saving")
        try:
            if ext == '.npy':
                np.save(filename,file_data)
            elif ext == '.npz':
                np.savez_compressed(filename,**kwargs)
            elif ext == '.stl':
                file_data.export(filename)
            elif ext == '.nii':
                nibabel.save(file_data, filename)
            elif ext == '.txt':
                with open(filename, 'w') as file:
                    file.write(file_data)
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
        finally:
            logger.info(f"{filename} finished saving")
            self.notify_saving_complete(filename)

    def _generate_filenames(self):
        input_files = {}
        output_files = {}

        # T1W file name
        input_files['T1'] = self.T1_fname

        # CT,ZTE, or PETRA file name if a file is given
        if self.extra_scan_fname:
            input_files['ExtraScan'] = self.extra_scan_fname

        # SimbNIBS input file name
        if self.simNIBS_type == 'charm':
            input_files['SimbNIBSinput'] = self.simNIBS_dir + os.sep + 'final_tissues.nii.gz'
            output_files['ReuseSimbNIBS'] = self.simNIBS_dir + os.sep + 'final_tissues_babelbrain_reuse.npy' # This file is only used to confirm if inputs can be reused
        else:
            input_files['SimbNIBSinput'] = self.simNIBS_dir + os.sep + 'skin.nii.gz'
            output_files['ReuseSimbNIBS'] = self.simNIBS_dir + os.sep + 'skin_babelbrain_reuse.npy' # This file is only used to confirm if inputs can be reused

        # SimbNIBS stl file names
        output_files['Skull_STL'] = self.simNIBS_dir + os.sep + 'bone.stl'
        output_files['CSF_STL'] = self.simNIBS_dir + os.sep + 'csf.stl'
        output_files['Skin_STL'] = self.simNIBS_dir + os.sep + 'skin.stl'
        
        # BiasCorrec and Coreg File Names
        if self.extra_scan_fname:
            BaseNameInT1 = os.path.splitext(self.T1_iso_fname)[0]
            if '.nii.gz' in self.T1_iso_fname:
                BaseNameInT1 =os.path.splitext(BaseNameInT1)[0]

            if self.CT_type in [2,3]: # ZTE/PETRA
                output_files['T1fnameBiasCorrec'] = BaseNameInT1 + '_BiasCorrec.nii.gz'
                BaseNameInZTE=os.path.splitext(self.extra_scan_fname)[0]
                if '.nii.gz' in self.extra_scan_fname:
                    BaseNameInZTE=os.path.splitext(BaseNameInZTE)[0]
                
                output_files['ZTEfnameBiasCorrec'] = BaseNameInZTE + '_BiasCorrec.nii.gz'
                output_files['ZTEInT1W'] = BaseNameInZTE + '_InT1.nii.gz'
                output_files['pCTfname'] = BaseNameInZTE + '_BiasCorrec_pCT.nii.gz'

            else: # CT
                if self.coreg==0:
                    pass
                elif self.coreg == 1:
                    output_files['T1fnameBiasCorrec'] = BaseNameInT1 + '_BiasCorrec.nii.gz'
                    output_files['T1fname_CTRes'] = BaseNameInT1 + '_BiasCorrec_CT_res.nii.gz'

                    CTInT1W = os.path.splitext(self.extra_scan_fname)[0]
                    if '.nii.gz' in self.extra_scan_fname:
                        CTInT1W=os.path.splitext(CTInT1W)[0]
                    
                    CTInT1W += '_InT1.nii.gz'
                    output_files['CTInT1W'] = CTInT1W
                else:
                    output_files['T1WinCT'] = BaseNameInT1 + '_InCT.nii.gz'

            # Intermediate mask file name
            output_files['ReuseMask'] = os.path.dirname(self.T1_iso_fname) + os.sep + self.prefix + 'ReuseMask.nii.gz' # This file is only used to confirm if previously generated intermediate mask can be reused
            
            # CT file name
            output_files['CTfname'] = os.path.dirname(self.T1_iso_fname) + os.sep + self.prefix + 'CT.nii.gz'

        return input_files, output_files