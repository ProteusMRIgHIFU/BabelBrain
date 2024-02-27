BabelBrain v0.3.2
=============
Samuel Pichardo, Ph.D  
Associate Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca


[![License: BSD-3](https://img.shields.io/badge/BSD-3-Clause.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://proteusmrighifu.github.io/BabelBrain)


**GUI application for the modeling of transcranial ultrasound for neuromodulation applications**
BabelBrain is a frontend application specially designed to work in tandem with neuronavigation software to perform focused ultrasound research. BabelBrain uses [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) extensively for calculations. BabelViscoFDTD is optimized for multiple GPU backends (Metal, OpenCL and CUDA). In its initial inception, BabelViscoFDTD is focused on MacOS systems based on Apple ARM64 processors. However, BabelViscoFDTD can run on any system (Mac, Linux, Windows) that has a decent GPU from NVidia or AMD. 

# Disclaimer
This software is provided "as is" and it is intended exclusively for research purposes.

# Hardware requirements
* A decent GPU (AMD, Nvidia or Apple Silicon). For AMD and Nvidia GPUs, 4 GB or more is highly recommended. For Apple Silicon, M1 Max/Ultra or M2 Max with 16 GB RAM or more is highly recommended. Intel-based Mac systems need a dedicated AMD GPU (internal or external). Intel-based iMac Pro and MacPro systems have internal GPUs suitable for sub 650 kHz simulations (i.e., Vega 56, Vega 64). An external GPU (i.e., AMD W6800 Pro) offers excellent performance and capability for high-frequency simulations.
* 16 GB RAM or more for main CPU memory for Intel-based systems.

# Standalone application
Ready-to-use applications (no need for Python installation) for macOS and Windows are available in the [Releases](https://github.com/ProteusMRIgHIFU/BabelBrain/releases) section. 
* For macOS, download the correct DMG image according to your CPU architecture (Intel X64 or ARM64),  and double-click the PKG installer. The first time you use you will be prompted to authorize to run and access directories.
* For Windows, download the MSI file and run the installer.

**Note for Windows:** CUDA 11.7 or up must be installed.

# Instructions for use
Please consult the [online manual](https://proteusmrighifu.github.io/BabelBrain/) for details on instructions for use.

# Manual Installation for Development 
If you prefer to run the code in a Python environment, the requirements are roughly a clean Python 3.9-3.10 environment, a healthy XCode installation in macOS, or CUDA (up to v11.8) + Visual Studio/gcc in Windows/Linux. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for details on what is needed for the FDTD solvers

*  Python 3.9-3.10. Anaconda/miniconda is recommended. - if running in Apple new ARM64 processors (M1, M1 Max, etc.), be sure to use a native ARM64 version. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for further details.
* [Blender](www.blender.org) installed

## Recommended settings
* All OS: create a conda environment using the appropriate yaml file. 

## Running
If running from the GitHub source code, just change to the BabelBrain directory and execute

`python BabelBrain.py`

## Building an standalone application
A Pyinstaller specification file is ready for use. To build either the macOS or Windows application, just change to the BabelBrain directory and run

`pyinstaller BabelBrain.spec --noconfirm`

A new application ready to use will be created at `BabelBrain/BabelBrain/dist/`

# Citation
If you find BabelBrain useful for your research, please consider adding a citation to:  
Pichardo S. BabelBrain: An Open-Source Application for Prospective Modeling of Transcranial Focused Ultrasound for Neuromodulation Applications.  
IEEE Trans Ultrason Ferroelectr Freq Control. 2023 Jul;70(7):587-599.  
doi: [10.1109/TUFFC.2023.3274046](https://doi.org/10.1109/TUFFC.2023.3274046). Epub 2023 Jun 29. PMID: 37155375.


# Version log
- 0.3.2 - March 3rd, 2024
  - New: Full integration with Brainsight (Rogue Research) version XXX is now operational. In Brainsight, you need to start a new Simbnibs project and during the planning stage, a new "FUS" button in the Brainsight GUI can be used to invoke BabelBrain. Once simulations are completed, the results of the normalized transcranial simulation (normalized between 0 and 1 in the brain region) will be loaded automatically in Brainsight. 
  - New: Export of thermal maps into Nifti format. In Step 3, the current thermal simulation on display can be now exported in Nifti for inspection in neuronavigation and visualization software.
  - New: Smart reuse of preexisting middle files. Previous versions recalculate in Step 1 every single middle file required to generate the domain for simulations. Some of these files can be perfectly reused in recalculations with the same subject if frequency and resolution are the same.  However, there is an important safety aspect to preserve; that is why the first versions we opted to always recalculate everything. In this new version, we developed a method to use hash signatures that help to detect any external changes to the files. If no changes are detected, then the middle files can be reused. This approach can save from 20% up to 80 % time or more (especially when using CT) in the execution time of Step 1. 
  - New: Support for PETRA scans. We adopted the formulas proposed by Brad Treeby's lab at UCL ([petra-to-ct](https://github.com/ucl-bug/petra-to-ct)). Now in the first screen of BabelBrain you can select between real CT, ZTE, PETRA or none in the options to use CT-type data for the simulations. 
  - New: Add 35-mm focusing devices to the list of devices associated with [Schafer *et al.*](https://doi.org/10.1109/TUFFC.2020.3006781)
  - Improved precision for all transducers. The method that couples the Rayleigh integral component, which models the transducer sources, to the FDTD domain was revised, improving precision of calculations.
  - Fix: Saved modified trajectory had an incorrect sign direction. 
  - Fix: Far field PML had an issue in Metal backend (solved in BabelViscoFDTD 1.0.5)
- 0.3.0 - Nov 5, 2023
  - New: Add checkboxes to hide marks on plots.
  - New: Replace labels in Step 3 with a table to organize better output metrics.
  - New: Add metrics in Step 3 of temperature and thermal dose at the target, which often does match the maximal temperature in the brain. Also, a metric of distance from maximal peak intensity in the brain to the target.
  - Fix an issue with display results in Step 3. Results were scaled to the intensity at the target instead of maximal intensity in the brain region, as done when calculating the required Isppa in water.
  - Fix an issue with the display results of the H317 Transducer.
  - Fix the remaining "bleeding" of the skin label in the brain region.
  - Fix NaN calculations when the target is accidentally in the bone region.
  - Fix passing version to macOS bundle.
  
- 0.2.9-b - Oct 14, 2023
  - macOS applications are now fully signed, and no more warnings from macOS Gatekeeper appear.
  - PKG installer in macOS DMG distribution files replaces the "drag" macOS app into the application.
  - Use of the latest pyinstaller library (6.1.0) and new scripts to sign macOS applications.
  - Small fix for Step 3 the AllCombinations file that was not saving correctly an index based on the combinations of DC, PRF and duration. 
- 0.2.9-a - Sep 21, 2023
  - Add fix for Apple Silicon systems with latest versions of MacOS. In some systems, occasional crashes were occurring. Fix was addressed at the underlying library at https://github.com/ProteusMRIgHIFU/py-metal-compute.
  - Address slow BHTE calculations in Step 3 in external AMD GPUs in Apple X64 system. Fix done in underlying library BabelViscoFDTD 1.0.1
  - Extra GPU optimizations in Step 1
  - Correct handling of dark mode display settings
- 0.2.9 - Aug 1, 2023
  - Add scrolling of imaging planes of acoustic fields in Step 2
  - Add the possibility in Step 2 of adjusting the positioning of the transducer to recreate the transducer pressed on the scalp. Use with caution. The processing will remove tissue layers. Limited to 10 mm.
  - Add multiple behind-the-scenes new GPU code to accelerate calculations for Step 1. Because now operations are float32, some very minor differences can be expected for the produced mask for simulations with previous versions.
- 0.2.7 - May 3, 2023
  - New devices as detailed in [Schafer *et al.*](https://doi.org/10.1109/TUFFC.2020.3006781) 
  - Option to show water-only modeling results added in Step 2
  - Adjustment of Z mechanical for CTX 500 device added for scenarios where a pad is added
  - New Nifti output (with extension `_Sub_NORM.nii.gz`) containing normalized pressure output (0 to 1.0) in the brain region. All other tissue regions are set to 0. This simplifies visualization and thresholding in tools such as fsleyes.
- 0.2.6 - Apr 8, 2023:
  - Windows standalone available 
  - BUG fix: Issue #14 - Reduced precision with single Elem Tx with high F#
  - BUG fix: Issue #13 - Inadequate use of SimpleITK SetSpacing to create isotropic scans 
  - Minor bugs for minimal requirements
- 0.2.5-1. March 15, 2023:
  - BUG fix: Issue #7 - Matplotlib backends are now included in macOS application 
  - BUG fix: Remove CPU from engine
- 0.2.5 -  Jan 10, 2023:
  - First public release

