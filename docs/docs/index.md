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

<img src="Target-demo.png">
BabelBrain is a frontend application for research purposes only in the study of applications transcranial focused ultrasound. BabelBrain calculates the transmitted acoustic field in the brain, taking into account the distortion effects caused by the skull barrier. BabelBrain also calculates the thermal effects that a given ultrasound regime, which is regulated mainly by the total duration of ultrasound exposure, the duty cycle of ultrasound use and the peak acoustic intensity,

BabelBrain is designed to work in tandem with neuronavigation and/or visualization software (such as 3DSlicer). BabelBrain uses extensively [BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) for calculations. BabelViscoFDTD is a finite-difference time-difference solver of isotropic viscoelastic equation, which was implemented to support multiple GPU backends (Metal, OpenCL and CUDA). In its initial inception, BabelBrain is focused for use of MacOS systems based on Apple ARM64 processors and Linux-based systems using high-end GPUs from AMD and NVidia. 


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


