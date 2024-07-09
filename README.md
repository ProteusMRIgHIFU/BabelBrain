BabelBrain v0.3.5
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
BabelBrain is a frontend application specially designed to work in tandem with neuronavigation and visualization software to perform transcranial focused ultrasound research. BabelBrain uses [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) extensively for calculations. BabelViscoFDTD is optimized for multiple GPU backends (Metal, OpenCL and CUDA). In its initial inception, BabelViscoFDTD was focused on MacOS systems based on Apple ARM64 processors. However, BabelViscoFDTD can run on any system (Mac, Linux, Windows) that has a decent GPU from NVidia or AMD. 

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
## Recommended settings
* All OS: create a conda environment using the appropriate yaml file for macOS Intel, macOS ARM64, Windows or Linux. 

Besides the recommended conda environment, a healthy XCode installation in macOS, or CUDA (up to v11.8) + Visual Studio/gcc in Windows/Linux will be required. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for details on what is needed for the FDTD solvers

*  CSG Python `pycork` library needs to be installed manually. Clone the repository in a BabelBrain environment. 
   
   In macOS, install the GMP library; for example with `homebrew`
   ```
   brew install gmp
   ```
   Install the `pycork` library with:
   ```
   git clone https://github.com/drlukeparry/pycork.git
   cd pycork
   git checkout d9efcd1da212c685345f65503ba253373dcdece0 
   git submodule update --init --recursive
   pip install .
   ```



## Running
If running from the GitHub source code, just change to the BabelBrain directory and execute

`python BabelBrain.py`Building a standalone application
A Pyinstaller specification file is ready for use. To build either the macOS or Windows application, just change to the BabelBrain directory and run

`pyinstaller BabelBrain.spec --noconfirm`

A new application ready to use will be created at `BabelBrain/BabelBrain/dist/`

# Citation
If you find BabelBrain useful for your research, please consider adding a citation to:  
Pichardo S. BabelBrain: An Open-Source Application for Prospective Modeling of Transcranial Focused Ultrasound for Neuromodulation Applications.  
IEEE Trans Ultrason Ferroelectr Freq Control. 2023 Jul;70(7):587-599.  
doi: [10.1109/TUFFC.2023.3274046](https://doi.org/10.1109/TUFFC.2023.3274046). Epub 2023 Jun 29. PMID: 37155375.


# Version log
- 0.3.5 - July 8thth, 2024
  - New: Selection of mapping method for acoustic properties based on CT scanner model, kernel variant and resolution is now available when selecting the CT input in the first dialog. The selected scanner combination will be saved for future sessions. However, default values will be applied when changing the type from real CT, PETRA or ZTE. A reset button to restore default values is also available.
  - New: Add advanced options for fine-tuning the processing. In this first iteration, these parameters are for the control of the domain generation in Step 1.  A new button for "Advanced options" is now available.
     - **Selection of Elastix Optimizer**. The default optimizer for Elastix, AdaptiveStochasticGradientDescent, works *almost* for every combination of CT->T1W and ZTE/PETRA->T1W. However, as much more testing has started to accumulate, we noticed that the CT->T1W was not working in some cases. We tested different options in Elastix, and the FiniteDifferenceGradientDescent and QuasiNewtonLBFGS seem to work well as a replacement. If your coregistration from CT to T1W is not working as expected, try one of these two other options. FiniteDifferenceGradientDescent seems to work better.
     - **Control of fraction of trabecular**. By default, the percentage of trabecular is 80% using the line of sight of the trajectory. If the trajectory crosses very thin regions like the parietal bone, there may be a chance all region is considered trabecular, which may not be desired. A warning message will be printed out in the terminal output in that case. In such cases as the parietal bone, consider reducing a lower value of trabecular, such as 0.1 or 0.2.
     - **Capability of specifying a subvolume region**. This feature is intended to give a bit more control for scenarios where the trajectory is regions where the skin mask may cause some issues, especially in targets near the back of the head. For most targets, there is no need to make any adjustments, but we have spotted a few cases (i.e., EEG-P7 location) in which the skin mask of the back of the head was pushing the whole domain and causing issues to place correctly the transducer. For these scenarios, adjusting a subvolume to extract for the simulations can mitigate these issues. This can also help to reduce image size for high-resolution cases.
      - **Force use of blender**. While pycork seems to do a great replacement, we identified that in some instances when using the subvolume feature there was a crash in the low pycork library. Forcing using Blender instead can help to run those special cases.
  - New: Legend of tissue type added in step 1.
  - Improvement: Do a watertight fix of STL masks before saving.  Before, a watertight fix for the charm-derived STL masks was being applied all the time, while this could have done only once when creating the STL files the first time. This will save computing time when recalculating trajectories.
  - Improvement: Replacement of Mechanical Adjustment in Z by Distance from Tx outplane to skin. This change is mostly aesthetic to make the GUI a bit clearer when adjusting the position of the Tx along the acoustic axis. Previously, the mechanical position in Z was shown relative to the target and a label showed the distance from the outplane to the skin. This setup can be confusing, so the control was changed to adjust rather directly the distance from the outplane to the skin, which is more intuitive during the TUS experiments.
  - Fix: In cases where the trajectory was crossing a very thin skull region, the degree of erosion used to assign the trabecular bone was calculated as 0, meaning all the bone should be considered trabecular. However, the Scipy function for erosion keeps eroding until the mask does not change, which in our case meant all bone was assigned cortical.
  - Fix: GPU filter to quantification had a bug for CUDA and OPENCL versions. 
- 0.3.4-2 - May 22nd, 2024
  - Fix: Correct handling of cases with a large number of sonication points in Step 3 when using low duty cycle.
  - Fix: Address hanging in Step 2 when running inside Brainsight with a large list of sonication points.
- 0.3.4-1 - Apr 19th, 2024
  - Fix: Export CSV for BSonix transducers had a bug preventing the export.
- 0.3.4 - Apr 5th, 2024
  - Improvement: Significantly faster calculations in Step 2. Improvements to the modeling of acoustic sources in r0.3.2 allowed the elimination of the two-step calculations used in previous versions. Computational cost savings should range between 48% to 40%. A large numerical study was executed to ensure the precision of calculations was not affected.
  -  Improvement. No more need for Blender. We finally found a native Python CSG library that is robust enough to perform the geometry tasks we have been using with Blender until now. This has only a minor implication for those users running BabelBrain in their own Python environment (see details above about installing the `pycork` library). For those using the stand-alone applications, there is no impact other than Blender can be safely uninstalled if there is no more need for it.
  - New: Support to new transducers. We added three new phased array devices: Two concave arrays and one flat 2D array.  The I12378 transducer is a 128-element device operating at 650 kHz with a focal length of 72 mm and a diameter of 103 mm. The ATAC transducer is a 128-element device operating at 1 MHz with a focal length 53.2 mm and a diameter of 58 mm.  The REMOPD transducer is a 256-element flat 2D array operating at 300 kHz with a diameter of 58 mm. We thank the team at Vanderbilt University for sharing the transducer definitions of their concave arrays (I12378 and ATAC transducers). We thank the team at Toronto Western Hospital and Fraunhofer IBMT for sharing the transducer definition of their REMOPD transducer.
  - New: Automatic calculation for mechanical corrections in X and Y directions. After running a first pass of simulation in Step 2, a new button action is now available to calculate the distance from the target to the center of mass of the focal spot at -6dB and suggest applying the required mechanical corrections in X and Y directions. This action should help to minimize the number of iterations in simulations looking to ensure the focal spot is aligned in the X and Y directions to the intended target.
  - New: Possibility to update thermal profile in Step 3. The initial design of BabelBrain assumed that the parameters of the timing of LIFU exposures would vary little in a study. That is why it is asked as initial input when initiating BabelBrain. However, some users have expressed their need to have more flexibility to explore variated settings without having to restart BabelBrain. For this purpose, we added in Step 3 a new action button that can be used to load an updated version of the thermal profile file. 
  - New: Multi-point LIFU exposures. BabelBrain now offers the possibility to execute electronic steering over a list of points if a concave array is selected. In the first dialog, if any of the three concave arrays is selected (H317, I12378 and ATAC), the user can select a new profile definition specific to multi-point steering. You can consult an example in the [Profiles](https://github.com/ProteusMRIgHIFU/BabelBrain/blob/main/Profiles/MultiFocus_Profile1.yaml) subdirectory in BabelBrain. Coordinates in the profile are relative to the user-specified steering in Step 2. Be aware that the selected duty cycle for the thermal simulations is split among all multi-point entries. It is assumed that a single burst is applied per steering location, before steering to the next location. For example, if simulating three-point steering with a duty cycle of 30%, this implies that each point will represent 10% of the duty cycle.
  - New: Examples for offline batch processing. We added a couple of Jupyter notebooks in the [OfflineBatchExamples](https://github.com/ProteusMRIgHIFU/BabelBrain/tree/main/OfflineBatchExamples) directory in the BabelBrain repository, along with a README file indicating steps to create an environment ready to run these examples. These notebooks can be very useful when running very large parametric studies. For example, the [CompareRayleightWithFDTD](https://github.com/ProteusMRIgHIFU/BabelBrain/tree/main/OfflineBatchExamples/CompareRayleightWithFDTD) case is the one we used to study the acoustic simulations in water-only conditions that facilitate reducing the computational costs in Step 2.
  - Fix: Occasional mask generation issues with high-resolution conditions. In some cases, Step 1 would produce incomplete tissue masks when using high PPW values (9 and higher) or with high frequency ( 1 MHz) when using Metal-based devices. The GPU filtering functions were rewritten to cover better these cases or at least provide a more meaningful error message to notify the users. 
  - Fix: Enforce float type in profile files. Entries in thermal profiles such as duty cycle must be float values. Entries such as "1" should be "1.0". An error will be shown when loading profiles that do not adhere to this convention. This should imply minor changes to old profile files.
- 0.3.2 - March 3rd, 2024
  - New: Full integration with Brainsight (Rogue Research) version 2.5.3 is now operational. In Brainsight, you need to start a new Simbnibs project and during the planning stage, a new "FUS" button in the Brainsight GUI can be used to invoke BabelBrain. Once simulations are completed, the results of the normalized transcranial simulation (normalized between 0 and 1 in the brain region) will be loaded automatically in Brainsight. 
  - New: Export of thermal maps into Nifti format. In Step 3, the current thermal simulation on display can now be exported in Nifti for inspection in neuronavigation and visualization software.
  - New: Smart reuse of preexisting middle files. Previous versions recalculate in Step 1 every single middle file required to generate the domain for simulations. Some of these files can be perfectly reused in recalculations with the same subject if frequency and resolution are the same.  However, there is an important safety aspect to preserve; that is why, in the first versions, we opted always to recalculate everything. In this new version, we developed a method to use hash signatures that help to detect any external changes to the files. If no changes are detected, then the middle files can be reused. This approach can save from 20% to 80 % of time or more (especially when using CT) in the execution time of Step 1. 
  - New: Support for PETRA scans. We adopted the formulas proposed by Brad Treeby's lab at UCL ([petra-to-ct](https://github.com/ucl-bug/petra-to-ct)). Now in the first screen of BabelBrain you can select between real CT, ZTE, PETRA or none in the options to use CT-type data for the simulations. 
  - New: Add 35-mm focusing devices to the list of devices associated with [Schafer *et al.*](https://doi.org/10.1109/TUFFC.2020.3006781)
  - Improvement: The precision for all transducers was improved. The method that couples the Rayleigh integral component, which models the transducer sources, to the FDTD domain was revised, improving precision for all calculations.
  - Fix: Re-enable support for older versions of macOS (Monterey and up) for ARM64.
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

