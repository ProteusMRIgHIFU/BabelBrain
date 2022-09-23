# Standalone application
Ready-to-use applications (no need of Python installation) for macOS are available in the Releases section. Download, open and drag "BabelBrain" into the Applications folder. The first time you will use you will be prompted to authorize it to run and to access directories. You may also need to authorize it in the Security settings of macOS.

# Manual Installation for Development 
If you prefer to run the code in a Python environment, the requirements are roughly a clean Python 3.9 environment and a healthy XCode installation. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for details on what is needed for the FDTD solvers
# Requirements
* Python 3.9. Anaconda/miniconda is recommended. - if running in Apple new ARM64 processors (M1, M1 Max, etc.), be sure of using a native ARM64 version. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for further details.
* [Blender](www.blender.org)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) 


Be sure FSL init scripts are properly activated in your .bash_profile or .zsh_profile.

## Recommended settings
Create first and activate a new environment with some basics libraries from Conda

  `conda create --name babel python=3.9.13 numpy=1.23.3 scipy=1.9.1 matplotlib=3.6.0`

  `activate babel`

Install dependencies with 

`pip install -r requirements.txt`

## Running
If running from the Github source code, just change to the BabelBrain directory and run

`python BabelBrain.py`

## Building standalone application
A Pyinstaller specification file is ready for use. To build the macOS application, just change to the BabelBrain directory and run

`pyinstaller BabelBrain.spec`

A new application ready to use will be created at `BabelBrain/BabelBrain/dist/`