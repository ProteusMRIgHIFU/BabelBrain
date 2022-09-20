BabelBrain
=============
Samuel Pichardo, Ph.D  
Associate Professor  
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute  
Cumming School of Medicine,  
University of Calgary   
samuel.pichardo@ucalgary.ca  
www.neurofus.ca

**GUI application for the modeling of transcranial ultrasound for neuromodulation applications**
BabelBrain is a frontend application especially designed to work in tandem with neuronavigation  software to perform focused ultrasound research. BabelBrain uses extensively [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for calculations. BabelViscoFDTD is optimized to run in multiple GPU backends (Metal, OpenCL and CUDA). In its initial inception, BabelViscoFDTD is focused for MacOS systems based on Apple ARM64 processors. However, BabelViscoFDTD can run in any system (Mac, Linux, Windows) that has a decent GPU from NVidia or AMD. 

# Disclaimer
This software is provided "as is" and it is intended exclusively for research purposes.

## Licensing
BSD 3-Clause License

Copyright (c) 2022, Samuel Pichardo
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Standalone application
Ready to use applications (no need of Python installation) for MacOS are available in the Releases section. Download, open and drag "BabelBrain" into the Applications folder. The first time you will use you will be prompted  to authorize to run and to access directories. You may also need to authorize in the Security settings of MacOS.

# Instructions for use
Please consult 

# Manual Installation for Development 
If you prefer to run the code in a Python environment, the requirements are roughly a clean Python 3.9 environment and healthy XCode installation. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for details what is needed for the FDTD solvers
# Requirements
* Python 3.9. Anaconda/miniconda is recommende. - if running in Apple new ARM64 processors (M1, M1 Max, etc.), be sure of using a native ARM64 version. Consult [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for further details.
* [Blender](www.blender.org)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) 


Be sure FSL init scripts are properly activated in your .bash_profile or .zsh_profile.

## Recommended settings
Create first and activate an new environment with some basics libraries from Conda

  `conda create --name babel python=3.9.13 numpy=1.23.3 scipy=1.9.1 matplotlib=3.6.0`

  `activate babel`

Install dependencies with 

`pip install -r requirements.txt`

## Running
If running from the Github source code, just change to the BabelBrain directory and run

`python BabelBrain.py`

## Building standalone application
A Pyinstaller specification file is ready for use. To build the MacOS application, just change to the BabelBrain directory and run

`pyinstaller BabelBrain.spec`

A new application ready to use will be created at `BabelBrain/BabelBrain/dist/Babelbrain.app`
