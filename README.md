BabelBrain v0.2.5
=============
Samuel Pichardo, Ph.D  
Associate Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca

**GUI application for the modeling of transcranial ultrasound for neuromodulation applications**
BabelBrain is a frontend application specially designed to work in tandem with neuronavigation software to perform focused ultrasound research. BabelBrain uses [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) extensively for calculations. BabelViscoFDTD is optimized for multiple GPU backends (Metal, OpenCL and CUDA). In its initial inception, BabelViscoFDTD is focused on MacOS systems based on Apple ARM64 processors. However, BabelViscoFDTD can run on any system (Mac, Linux, Windows) that has a decent GPU from NVidia or AMD. 

# Disclaimer
This software is provided "as is" and it is intended exclusively for research purposes.

# Standalone application
Ready-to-use applications (no need for Python installation) for MacOS are available in the Releases section. Download, open and drag "BabelBrain" into the Applications folder. The first time you use you will be prompted to authorize to run and access directories. You may also need to authorize it in the Security settings of MacOS.

# Instructions for use
Please consult the [online manual](https://proteusmrighifu.github.io/BabelBrain/) for details on instructions for use.

# Manual Installation for Development 
If you prefer to run the code in a Python environment, the requirements are roughly a clean Python 3.9-3.10 environment, a healthy XCode installation in macOS, or CUDA (up to v11.8) + Visual Studio/gcc in Windows/Linux. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for details on what is needed for the FDTD solvers
# Requirements
* Python 3.9-3.10. Anaconda/miniconda is recommende. - if running in Apple new ARM64 processors (M1, M1 Max, etc.), be sure to use a native ARM64 version. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for further details.
* [Blender](www.blender.org)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) 


Be sure FSL init scripts are properly activated in your .bash_profile or .zsh_profile.

## Recommended settings
macOS/Linux: Create first and activate a new environment with some basics libraries from Conda

  `conda create --name babel python=3.10 numpy scipy`

  `activate babel`

In Windows only, install  cupy and numba from 
  `conda install -c conda-forge cupy numba`

Install the rest of dependencies with either

`pip install -r requirements_mac.txt`
or
`pip install -r requirements_linux.txt`
or
`pip install -r requirements_win.txt`

## Running
If running from the Github source code, just change to the BabelBrain directory and run

`python BabelBrain.py`

## Building an standalone application for macOS
A Pyinstaller specification file is ready for use. To build the MacOS application, just change to the BabelBrain directory and run

`pyinstaller BabelBrain.spec`

A new application ready to use will be created at `BabelBrain/BabelBrain/dist/Babelbrain.app`

