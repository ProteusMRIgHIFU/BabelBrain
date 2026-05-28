# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
from PyInstaller import compat
from os import listdir
import platform
from glob import glob
import os

is_mac = "Darwin" in platform.system()

# ==============================================================================
# Helper Functions
# ==============================================================================

def collect_external_bin_binaries():

    collected_binaries = []

    # Recursively loop through external bin directory and add binaries
    for path in glob("ExternalBin" + os.sep + "**", recursive=True):
        if os.path.isfile(path):
            # Common binaries
            if ".txt" in path:
                collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
            elif ".py" in path:
                collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]

            # Platform specific binaries
            if "Darwin" in platform.system():
                if "mac" in path:
                    collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
                if ".sh" in path:
                    collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
            elif "Windows" in platform.system():
                if "windows" in path:
                    collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
                if ".bat" in path:
                    collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
            elif "Linux" in platform.system():
                if "linux" in path:
                    collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
                if ".sh" in path:
                    collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
            else:
                raise ValueError("Only MAC, Windows, and Linux systems are supported")

    print("\nExternal Bin Binaries:\n" + "\n".join(map(str, collected_binaries)))  # print list of binaries

    return collected_binaries


def collect_missing_package_info(missing_packages):

    missing_datas = []
    missing_binaries = []
    missing_hiddenimports = []

    print("\nMissing Info From The Following Packages:")
    for mp in missing_packages:
        # Collect missing package info
        modinfo = collect_all(mp)

        # Print list of missing packages
        if modinfo is not []:
            print(mp)

        # Add info
        missing_datas += modinfo[0]
        missing_binaries += modinfo[1]
        missing_hiddenimports += modinfo[2]

    # print("\nAdding Missing Data Files:\n" + "\n".join(map(str, missing_datas)))
    print("\nAdding Missing Binaries:\n" + "\n".join(map(str, missing_binaries)))
    print("\nAdding Missing Hidden Imports:\n"+ "\n".join(map(str, missing_hiddenimports)))
    return missing_datas, missing_binaries, missing_hiddenimports


# ==============================================================================
# Base Parameters
# ==============================================================================

with open("version.txt") as f:
    version = f.readline().strip()

block_cipher = None

datas = []
binaries = []
hiddenimports = []
upx_exclude_list = []

# ==============================================================================
# Add Common Data Files
# ==============================================================================

commonDatas = [
    ("Babel_H317/default.yaml", "./Babel_H317"),
    ("Babel_H246/default.yaml", "./Babel_H246"),
    ("Babel_CTX500/default.yaml", "./Babel_CTX500"),
    ("Babel_CTX250/default.yaml", "./Babel_CTX250"),
    ("Babel_CTX250_2ch/default.yaml", "./Babel_CTX250_2ch"),
    ("Babel_DPX500/default.yaml", "./Babel_DPX500"),
    ("Babel_DPXPC300/default.yaml", "./Babel_DPXPC300"),
    ("Babel_SingleTx/default.yaml", "./Babel_SingleTx"),
    ("Babel_SingleTx/defaultBSonix.yaml", "./Babel_SingleTx"),
    ("Babel_REMOPD/default.yaml", "./Babel_REMOPD"),
    ("Babel_I12378/default.yaml", "./Babel_I12378"),
    ("Babel_ATAC/default.yaml", "./Babel_ATAC"),
    ("Babel_R15148/default.yaml", "./Babel_R15148"),
    ("Babel_R15287/default.yaml", "./Babel_R15287"),
    ("Babel_R15473/default.yaml", "./Babel_R15473"),
    ("Babel_R15646/default.yaml", "./Babel_R15646"),
    ("Babel_IGT64_500/default.yaml", "./Babel_IGT64_500"),
    ("Babel_DomeTx/default.yaml", "./Babel_DomeTx"),
    ("Babel_H301/default.yaml", "./Babel_H301"),
    ("default.yaml", "./"),
    ("version.txt", "./"),
    ("version-gui.txt", "./"),
    ("icons8-hourglass.gif", "./"),
    ("form.ui", "./"),
    ("ExampleHistogram.h5", "./"),
    ("rigid_template.txt", "./"),
    ("Babel_H317/form.ui", "./Babel_H317"),
    ("Babel_H246/form.ui", "./Babel_H246"),
    ("_Babel_RingTx/form.ui", "./_Babel_RingTx"),
    ("Babel_SingleTx/form.ui", "./Babel_SingleTx"),
    ("Babel_SingleTx/formBx.ui", "./Babel_SingleTx"),
    ("Babel_REMOPD/form.ui", "./Babel_REMOPD"),
    ("Babel_DomeTx/form.ui", "./Babel_DomeTx"),
    ("GUIComponents/scrollbars.ui", "./GUIComponents"),
    ("../TranscranialModeling/H-317 XYZ Coordinates_revB update 1.18.22.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/I12378.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/ATACArray.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/DomeTxTransducerGeometry.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/R15148_1001.mat", "./TranscranialModeling"),
    ("../TranscranialModeling/R15646.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/MapPichardo.h5", "./TranscranialModeling"),
    ("../TranscranialModeling/WebbHU_SoS.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/WebbHU_Att.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/ct-calibration-low-dose-30-March-2023-v1.h5", "./TranscranialModeling"),
    ("../TranscranialModeling/ct_to_density_calibration_cph2025_line_v1.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/REMOPD_ElementPosition.mat", "./TranscranialModeling"),
    ("../TranscranialModeling/IGT64_500.csv", "./TranscranialModeling"),
    ("../TranscranialModeling/H301.csv", "./TranscranialModeling"),
    ("Babel_Thermal/form.ui", "./Babel_Thermal"),
    ("GPUFunctions/GPUBinaryClosing/binary_closing.cpp", "./GPUFunctions/GPUBinaryClosing"),
    ("GPUFunctions/GPULabel/label.cpp", "./GPUFunctions/GPULabel"),
    ("GPUFunctions/GPUMapping/map_filter.cpp", "./GPUFunctions/GPUMapping"),
    ("GPUFunctions/GPUMedianFilter/median_filter.cpp", "./GPUFunctions/GPUMedianFilter"),
    ("GPUFunctions/GPUResample/affine_transform.cpp", "./GPUFunctions/GPUResample"),
    ("GPUFunctions/GPUResample/spline_filter.cpp", "./GPUFunctions/GPUResample"),
    ("GPUFunctions/GPUVoxelize/voxelize.cpp", "./GPUFunctions/GPUVoxelize"),
    ("GPUFunctions/GPUVoxelize/helper_math.h", "./GPUFunctions/GPUVoxelize"),
]

datas += commonDatas
print("\nCommon Data Files:\n" + "\n".join(map(str, commonDatas)))  # print list of common data files

# ==============================================================================
# Add Common Hidden Imports
# ==============================================================================

commonhidden = [
    "Babel_H317.Babel_H317",
    "Babel_H246.Babel_H246",
    "Babel_CTX500.Babel_CTX500",
    "Babel_CTX250.Babel_CTX250",
    "Babel_CTX250_2ch.Babel_CTX250_2ch",
    "Babel_DPX500.Babel_DPX500",
    "Babel_DPXPC300.Babel_DPXPC300",
    "Babel_SingleTx.Babel_SingleTx",
    "Babel_SingleTx.Babel_BSonix",
    "Babel_REMOPD.Babel_REMOPD",
    "Babel_I12378.Babel_I12378",
    "Babel_ATAC.Babel_ATAC",
    "Babel_R15148.Babel_R15148",
    "Babel_R15287.Babel_R15287",
    "Babel_R15473.Babel_R15473",
    "Babel_R15646.Babel_R15646",
    "Babel_IGT64_500.Babel_IGT64_500",
    "Babel_DomeTx.Babel_DomeTx",
    "Babel_H301.Babel_H301",
    "TranscranialModeling.BabelIntegrationCONCAVE_PHASEDARRAY",
    "TranscranialModeling.BabelIntegrationH317",
    "TranscranialModeling.BabelIntegrationH246",
    "TranscranialModeling.BabelIntegrationREMOPD",
    "TranscranialModeling.BabelIntegrationI12378",
    "TranscranialModeling.BabelIntegrationATAC",
    "TranscranialModeling.BabelIntegrationR15148",
    "TranscranialModeling.BabelIntegrationR15646",
    "TranscranialModeling.BabelIntegrationIGT64_500",
    "TranscranialModeling.BabelIntegrationDomeTx",
    "TranscranialModeling.BabelIntegrationH301",
]

hiddenimports += commonhidden
print("\nCommon Hidden Imports:\n" + "\n".join(map(str, commonhidden)))  # print list of common hidden imports

# ==============================================================================
# Platform-specific
# ==============================================================================

if "Darwin" in platform.system():  # For MAC systems

    # BabelViscoFDTD
    tmp_ret = collect_all("BabelViscoFDTD")
    binaries += tmp_ret[1]
    datas += tmp_ret[0]

    # Trimesh
    tmp_ret = collect_all("trimesh")
    hiddenimports += tmp_ret[2]
    datas += tmp_ret[0]

    # ITK
    tmp_ret = collect_all("itk")
    hiddenimports += tmp_ret[2]

    itk_datas = collect_data_files("itk", include_py_files=True)
    datas += [x for x in itk_datas if "__pycache__" not in x[0]]

    # VTK
    hiddenimports += [
        "vtkmodules",
        "vtkmodules.all",
        "vtkmodules.qt.QVTKRenderWindowInteractor",
        "vtkmodules.util",
        "vtkmodules.util.numpy_support",
        "vtkmodules.numpy_interface",
        "vtkmodules.numpy_interface.dataset_adapter",
    ]

    binaries += collect_external_bin_binaries()

    # PyDICOM
    tmp_ret2 = collect_all("pydicom")
    hiddenimports += tmp_ret2[2]

    # Histoprint and MKL
    if "arm64" not in platform.platform():
        hiddenimports += ["histoprint"]
        libdir = compat.base_prefix + "/lib"
        mkllib = filter(lambda x: x.startswith("libmkl_"), listdir(libdir))
        if mkllib != []:
            print("MKL installed as part of numpy, importing that!")
            binaries += map(lambda l: (libdir + "/" + l, "./"), mkllib)
            print(binaries)

    # MLX
    if "arm64" in platform.platform():
        hiddenimports += collect_submodules("mlx")

elif "Windows" in platform.system():  # For windows systems

    # CUDA paths
    conda_prefix = os.environ["CONDA_PREFIX"]
    cuda_include = os.path.join(conda_prefix, "Library", "include")
    cuda_bin = os.path.join(conda_prefix, "Library", "bin")

    # ==========================================================================
    # Windows Specific Data Files
    # ==========================================================================

    # CUDA headers
    cuda_headers = []
    cuda_headers += [(os.path.join(cuda_include, "cuda_runtime.h"), "./Library/include")]
    cuda_headers += [(os.path.join(cuda_include, "cuda_fp16.h"), "./Library/include")]

    # Add to datas
    datas += cuda_headers
    print("\nData Files:\n" + "\n".join(map(str, cuda_headers)))  # print list of window specific data files

    # ==========================================================================
    # Windows Specific Binaries
    # ==========================================================================

    # binaries += [(f, ".") for f in glob(os.path.join(cuda_bin, "*.dll"))] # CUDA dlls
    binaries += [(conda_prefix + "/Library/bin/nvrtc-builtins64_*.dll", ".")]
    print("\nBinaries:\n" + "\n".join(map(str, binaries)))  # print list of binaries

    binaries += collect_external_bin_binaries()

    # ==========================================================================
    # Windows Specific Missing Packages
    # ==========================================================================

    missing_package_info = [
        "BabelViscoFDTD",
        "cupy",
        "cupy_backends",
        "fastrlock",
    ]
    md, mb, mhi = collect_missing_package_info(missing_package_info)

    datas += md
    binaries += mb
    hiddenimports += mhi

else:  # For linux systems

    # ==========================================================================
    # Linux Specific Binaries
    # ==========================================================================

    binaries += [(conda_prefix + "/lib64/libnvrtc-builtins.so", "./")]
    print("\nBinaries:\n" + "\n".join(map(str, binaries)))  # print list of binaries

    binaries += collect_external_bin_binaries()

    # ==========================================================================
    # Linux Specific Missing Packages
    # ==========================================================================

    missing_package_info = [
        "BabelViscoFDTD",
        "cupy",
        "cupyx",
        "cupy_backends",
        "fastrlock",
        "skimage",
        "pyopencl",
    ]

    md, mb, mhi = collect_missing_package_info(missing_package_info)

    datas += md
    binaries += mb
    hiddenimports += mhi

# ==============================================================================
# Dependency Analysis
# ==============================================================================

print("\nPerforming Dependency Analysis:")
a = Analysis(
    ["BabelBrain.py"],
    pathex=["./"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["TkAgg","Qt5Agg","AGG","PDF","PS","SVG","PGF"],  # collect multiple backends
        },
    },
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ==============================================================================
# Python Bytecode Packaging
# ==============================================================================

print("\nPerforming Bytecode Packaging:")
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ==============================================================================
# Executable Configuration
# ==============================================================================

print("\nPerforming Executable Configuration:")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BabelBrain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=upx_exclude_list,
    console=not is_mac,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    entitlements_file=None,
    icon=None if is_mac else ["Proteus-Alciato-logo.ico"],
)

# ==============================================================================
# Bundle Assembly
# ==============================================================================

print("\nPerforming Bundle Assembly:")
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=is_mac,
    upx_exclude=[],
    name="BabelBrain",
)

# ==============================================================================
# macOS App Bundle
# ==============================================================================

if is_mac:
    print("\nPerforming macOS App Bundle Creation:")
    app = BUNDLE(
        coll,
        name="BabelBrain.app",
        version=version,
        bundle_identifier="com.ucalgary.babelbrain",
        icon="./Proteus-Alciato-logo.png",
    )
