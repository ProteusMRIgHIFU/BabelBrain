# Instructions to prepare and run PCA fit ZTE/PETRA to CT 
---

Follow the filename convention strictly. 

1.	Create a conda environment using one of the YML environment files in BabelBrain's root directory.

2. Actiate the environment and add the `jupyterlab` package with\
    `pip install jupyterlab`

    In the directory where the code and instructions are located, create a directory called **DATA**

2.	Inside the **DATA** directory, create another subdir with a clean ID scheme like “ID_001”. Inside this subdirectory, copy the **T1W.nii.gz**, **T2W.nii.gz**, **ZTE.nii.gz** and **CT.nii.gz**. Every time there is new subject data, follow this scheme. Even if using PETRA scans, rename it as ZTE.nii.gz,

3.	Open a terminal in the ID subdir and run charm, use the same ID for charm as the subdirectory, for example\
    `charm ID_001 T1W.nii.gz T2W.nii.gz --forceqform`

4.	In Brainsight, create a new project for the subject and create a dummy trajectory; any trajectory inside the brain will work.
<img width="1089" alt="image" src="https://github.com/user-attachments/assets/33ac9911-3b6d-41a1-8a7d-34bcce4e1701" />


5.	Export the trajectory inside the ID subdir (i.e. DATA/ID_001) with the name **Target.txt**.

6.	In Finder, make a copy of **RunPCA_FIT_ZTE_PETRA.ipynb** to something like **RunPCA_FIT_ID_001.ipynb**. Do not modify/re run  RunPCA_FIT_ZTE_PETRA.ipynb so you can see how results should look. There is also the PDF file RunPCA_FIT_ZTE_PETRA.pdf with a hard copy of a successful run.

7.	Open a terminal in the directory where you unzipped the code and activate conda environment

    `conda activate BabelJup`

8.	Initiate Jupyter lab with

    `jupyter lab`

9.	Open the recently copied notebook (i.e., RunPCA_FIT_ID_001.ipynb) and follow the instructions there. If all works ok, the density plot with the new PCA fit should like this.
<img width="1037" alt="image" src="https://github.com/user-attachments/assets/2c12b794-fc96-4da9-8e40-ec92b99e8a10" />



The **RunAllPCA.ipynb** Notebook can be used to run a batch of cases and calculate the PCA fit when considering all cases.
