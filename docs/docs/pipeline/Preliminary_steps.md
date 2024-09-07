# Preliminary steps
## Anatomical Imaging 
* Collect T1W (and optionally T2W) imaging of a participant. T1W scan **must** be 1-mm isotropic scans.
* *Optional*: CT scan of the participant. Depending on the study being conducted, counting with a CT scan improves the precision of the simulation. 
* *Optional*:: ZTE or PETRA scan of the participant. A pseudo-CT scan can be reconstructed using an ultrashort echo time  (ZTE in GE, PETRA in Siemens) MRI scan. Details on MRI scan parameters and methods for pseudo-CT reconstruction (using the "classical" approach) can be found in the work presented by 
<a href="https://ieeexplore.ieee.org/document/9856605/" target="_blank">Miscouridou et al.</a> (DOI: 10.1109/TUFFC.2022.3198522) and in the GitHub repository <a href="https://github.com/ucl-bug/petra-to-ct" target="_blank">petra-to-ct</a>, both from the UCL group. The user needs only to provide the Nifti file of the ZTE/PETRA scan. BabelBrain will do the transformation to pseudo-CT. A Nifti file with the pseudo-CT will be generated. Consult [MRI Sequences](../MRISequences/MRISequences.md) for GE and Siemens scan settings recomendations 

## Pre-processing 
* Execute <a href="https://simnibs.github.io/simnibs/build/html/index.html" target="_blank">SimNIBS</a> 4.x `charm` processing tool:

    ```
    charm <ID> <Path to T1W Nifti file> <Path to T2W Nifti file>
    ```

    `<ID>` is a string for identification. A subdirectory `m2m_<ID>` will be created. Take note of this directory, this will be referred to as the **SimNIBS output** directory in the following of this manual.
    
    **Note**: Sometimes, `charm` may complain that the qform and sform matrices are inconsistent. We have observed this when converting DICOM datasets with `dcm2niix`. If `charm` complains, you can try passing the  `--forceqform` parameter when executing `charm`.  

* Identify the coordinates of the target of focus ultrasound in T1W space. If you need to start in standardized space (e.g. MNI), there are many tools (FSL, SPM12, etc) that can be used to convert from standardized space to T1W space. 

    For example, with FSL, a simple csv file (`mni.csv`) can be created with the coordinates in MNI such as `-32.0 -20.0 65.0`. Then run the following commands

    `flirt -in <path to T1W nifti> -ref $FSLDIR/data/standard/MNI152_T1_1mm -omat anat2mni.xfm -out anat_norm`

    `std2imgcoord -img <path to T1W nifti>  -std $FSLDIR/data/standard/MNI152_T1_1mm.nii -xfm anat2mni.xfm mni.csv > natspace.csv`

    The file `natspace.csv` will contain the MNI coordinates converted to T1W space. Please note that often visual inspections could be required to confirm the location.
    


If no CT or ZTE scans are available, a mask representing the skull bone will be generated from the `headreco` or `charm` tools output. Be sure of inspecting the generated mask Nifti file to ensure the mask is correctly calculated. Our experience indicates that `charm` tool produces a better skull mask extraction. When using only T1W and T2W as inputs, BabelBrain uses a generic mask to represent the skull bone (including regions of trabecular and cortical bone). Average values of speed of sound and attenuation are assigned to these bone layers. Consult the appendix section for details on the values used.

If a CT or ZTE scan is provided, a mapping of density, speed of sound and attenuation will be produced. Consult the appendix section for details on the mapping procedure.
