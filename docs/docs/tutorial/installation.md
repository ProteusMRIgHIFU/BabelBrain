# Standalone application
Ready-to-use applications (no need for Python installation) for macOS are available in the Releases section. Download the installer and drag "BabelBrain" into the Applications folder. The first time you will use you may be prompted to authorize it to run and access directories. You may also need to authorize it in the Security settings of macOS.

# Manual Installation for Development 
If you prefer to run the code in a Python environment, the requirements are roughly a clean Python 3.9 environment and a healthy XCode installation. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for details on what is needed for the FDTD solver.
## Requirements
* Python 3.9. Anaconda/miniconda is recommended. - if running in Apple new ARM64 processors (M1, M1 Max, etc.), be sure of using a native ARM64 version. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for further details.



Be sure FSL init scripts are properly activated in your .bash_profile or .zsh_profile.

### Recommended settings
Create first and activate a new environment with some basics libraries from Conda

  `conda create --name babel python=3.9.13 numpy=1.23.3 scipy=1.9.1 matplotlib=3.6.0`

  `activate babel`

Install dependencies with 

`pip install -r requirements.txt`

** BabelBrain uses `antspyx`, which can take longtime to compile.
### Running
Change to the `BabelBrain/BabelBrain` directory and run

`python BabelBrain.py`

### Building standalone application
A Pyinstaller specification file is provided to build the macOS application. In the `BabelBrain/BabelBrain` directory run

`pyinstaller BabelBrain.spec`

A new application will be created at `BabelBrain/BabelBrain/dist/`

A dmg image installer can be created with the ``BabelBrain/BabelBrain/create_dmg.sh` script. 