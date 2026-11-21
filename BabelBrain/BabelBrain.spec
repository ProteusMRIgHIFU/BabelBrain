# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
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
            # PlanTUS is a pinned submodule: bundle its whole tree (yaml configs,
            # .scene templates, .gii transducer models, code), not just .py files.
            # glob("**") already skips dotfiles, so .git contents are excluded.
            if os.path.join("PlanTUS", "PlanTUS") + os.sep in path:
                if "__pycache__" in path:
                    continue
                collected_binaries += [(path, "." + os.sep + os.path.dirname(path))]
                continue

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


def collect_transducer_configs():
    """``default.yaml`` for every built-in transducer.

    Utils.paths.resource_path() resolves a frozen module's resources as
    ``_MEIPASS/<name of the module's own folder>``, so each config has to land
    in a folder named after its transducer, e.g.
    ``babel_transducers/focused_array/H317/default.yaml`` -> ``./H317``.
    Globbing keeps this in step with babel_transducers/ as devices are added.
    """

    collected = []
    for path in glob(os.path.join("babel_transducers", "*", "*", "*.yaml")):
        collected += [(path, "." + os.sep + os.path.basename(os.path.dirname(path)))]

    print("\nTransducer Configs:\n" + "\n".join(map(str, collected)))

    return collected


def collect_gpu_kernels():
    """GPU kernel sources and headers.

    ``GPUFunctions/<Op>/<kernel>.cpp`` is read by ``GPUFunctions/<Op>/*.py``,
    which resolves to ``_MEIPASS/<Op>``. The second copy under
    ``./GPUFunctions/<Op>`` is what keeps GPUUtils' nvrtc include path
    (``_MEIPASS/GPUFunctions``) valid for the CUDA backend.
    """

    collected = []
    for pattern in ("*.cpp", "*.h"):
        for path in glob(os.path.join("GPUFunctions", "*", pattern)):
            op = os.path.basename(os.path.dirname(path))
            collected += [(path, "." + os.sep + op),
                          (path, "." + os.sep + os.path.join("GPUFunctions", op))]

    print("\nGPU Kernels:\n" + "\n".join(map(str, collected)))

    return collected


def collect_local_submodules(package_dir, package_name):
    """Every module under a first-party package, named as it is imported.

    ``collect_submodules()`` is not usable for these: it imports the package in
    an isolated subprocess whose sys.path does not contain the repository, and
    it skips sub-folders without an ``__init__.py``. Walking the tree needs no
    imports and cannot miss a module.
    """

    modules = [package_name]
    for path in glob(os.path.join(package_dir, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, package_dir)
        if "__pycache__" in rel:
            continue
        parts = rel[: -len(".py")].split(os.sep)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules += [".".join([package_name] + parts)]

    return sorted(set(modules))


def collect_optional_packages(packages):
    """collect_all() for packages that are not installed on every platform."""

    opt_datas = []
    opt_binaries = []
    opt_hiddenimports = []

    for pkg in packages:
        try:
            modinfo = collect_all(pkg)
        except Exception as e:  # not installed in this environment
            print(f"\nSkipping optional package {pkg}: {e}")
            continue
        opt_datas += modinfo[0]
        opt_binaries += modinfo[1]
        opt_hiddenimports += modinfo[2]

    return opt_datas, opt_binaries, opt_hiddenimports


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
# 'scripting' and 'server' are imported lazily inside BabelBrain.main() for the
# --execute / --code and --serve CLI entries; 'ArtifactIO' is imported by the
# server and (guarded) by the pipeline modules. List them so the frozen binary
# always bundles them.
hiddenimports = ['scripting', 'server', 'ArtifactIO', 'FastGzip',
                 'RemoteServers', 'RunServerCalculation',
                 'client_functions', 'GUIComponents.RemoteServerDialog']
upx_exclude_list = []

# ==============================================================================
# Add Common Data Files
# ==============================================================================

# Everything Utils.paths.resource_path() / bundle_root() reaches at runtime.
# resource_path(__file__) is _MEIPASS/<name of the calling module's folder> and
# bundle_root(__file__) is _MEIPASS itself, so destinations below are named
# after the folder the reading module lives in -- not after its source path.
commonDatas = collect_transducer_configs() + collect_gpu_kernels() + [
    # Read by Utils/transducer_registry.py via bundle_root()
    ("SelFiles/transducer_list.yaml", "./SelFiles"),
    # Jinja sources for user-created transducers, rendered by
    # CreateTransducers/transducer_creator.py via bundle_root()
    ("babel_transducers/transducer_templates/babel_integration_tx.py.jinja", "./babel_transducers/transducer_templates"),
    ("babel_transducers/transducer_templates/babel_tx.py.jinja", "./babel_transducers/transducer_templates"),
    ("babel_transducers/transducer_templates/tx_form.py.jinja", "./babel_transducers/transducer_templates"),
    # Ready-to-edit custom transducer yaml templates, one per geometry type, shipped for users to copy
    *[(path, "./custom_tx_templates") for path in sorted(glob(os.path.join("..", "custom_tx_templates", "*.yaml")))],
    # Bundle root: read by the top-level modules (BabelBrain.py, ClockDialog.py,
    # CTZTEProcessing.py) and by anything calling bundle_root()
    ("default.yaml", "./"),
    ("version.txt", "./"),
    ("version-gui.txt", "./"),
    ("icons8-hourglass.gif", "./"),
    ("ExampleHistogram.h5", "./"),
    ("rigid_template.txt", "./"),
    ("Proteus-Alciato-logo.png", "./"),
    # Skull-property calibration tables, read by
    # TranscranialModeling/babel_integration/babel_integration_helpers.py
    ("../TranscranialModeling/MapPichardo.h5", "./"),
    ("../TranscranialModeling/WebbHU_SoS.csv", "./"),
    ("../TranscranialModeling/WebbHU_Att.csv", "./"),
    ("../TranscranialModeling/ct-calibration-low-dose-30-March-2023-v1.h5", "./"),
    ("../TranscranialModeling/ct_to_density_calibration_cph2025_line_v1.csv", "./"),
]

# Compiled UI string catalogues (see Localization.py and i18n/README.md). Only
# the .qm files are needed at run time; the .ts sources stay out of the bundle.
# glob so that new languages/modes are picked up without touching this list.
commonDatas += [(f, "./i18n") for f in sorted(glob("i18n" + os.sep + "*.qm"))]
# Fail the build rather than shipping an app that is missing a resource: a
# source path that no longer exists is silently dropped by PyInstaller, which
# is how the pre-UserTx transducer layout survived in this spec unnoticed.
missing_sources = [src for src, _ in commonDatas if not os.path.isfile(src)]
if missing_sources:
    raise FileNotFoundError(
        "BabelBrain.spec references data files that do not exist:\n  "
        + "\n  ".join(missing_sources)
    )

datas += commonDatas
print("\nCommon Data Files:\n" + "\n".join(map(str, commonDatas)))  # print list of common data files

# ==============================================================================
# Add Common Hidden Imports
# ==============================================================================

# Modules PyInstaller's import graph cannot see. BabelBrain.load_ui() imports
# the Step-2 GUI by name (babel_transducers.<type>.<Tx>.babel_<Tx>) and the
# matching solver is pulled in from there, so both trees are collected whole.
commonhidden = [
    "Babel_Thermal.Babel_Thermal",
    "CreateTransducers.transducer_creator",
    "CreateTransducers.transducer_verification_dialog",
    "GUIComponents.TxPanelBase",
    "TranscranialModeling.tx_geometries",
]
commonhidden += collect_local_submodules("babel_transducers", "babel_transducers")
commonhidden += collect_local_submodules(
    os.path.join("..", "TranscranialModeling", "babel_integration"),
    "TranscranialModeling.babel_integration",
)

hiddenimports += commonhidden
print("\nCommon Hidden Imports:\n" + "\n".join(map(str, commonhidden)))  # print list of common hidden imports

# ==============================================================================
# Add Common Optional Packages
# ==============================================================================

# pyvista backs the 3D views in BabelDatasetPreps and the custom-transducer
# verification dialog; pyvistaqt is only listed in some environment files, so
# a missing one is skipped rather than failing the build.
# isal (ISA-L gzip, used by FastGzip) is optional: saves fall back to zlib.
od, ob, ohi = collect_optional_packages(["pyvista", "pyvistaqt", "isal"])
datas += od
binaries += ob
hiddenimports += ohi

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

    # VTK: only the modules the code imports (see Utils/vtk_api.py); never
    # vtkmodules.all, which drags in every VTK module -- see EXCLUDES below.
    hiddenimports += [
        "vtkmodules",
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

    # Histoprint. (MKL used to be copied in here for Intel builds; numpy now
    # uses OpenBLAS on every platform -- see check_no_mkl() below.)
    if "arm64" not in platform.platform():
        hiddenimports += ["histoprint"]

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

# Packages that are installed in the build environments but must not ship.
# Each one is either unused at run time or only reachable through an optional
# import that is guarded with try/except.
EXCLUDES = [
    # `import vtk` == vtkmodules.all: every VTK module plus native deps
    # (VTK-m/viskores, IOSS, CGNS, NetCDF, FFmpeg). BabelBrain imports VTK
    # module by module through Utils/vtk_api.py; pyvista only names `vtk` in
    # TYPE_CHECKING blocks and ImportError fallbacks that never run.
    "vtk",
    "vtkmodules.all",
    # numba + llvmlite (~110 MB): the only user was a popcount in
    # GPUVoxelize/Voxelize.py, now plain NumPy.
    "numba",
    "llvmlite",
    # PyMuPDF (~45 MB): the calibration viewer shows PNG twins of the PDFs
    # (Calibration/plot_preview.py) and imports fitz only as a fallback.
    "fitz",
    "pymupdf",
    # scikit-learn: only an optional, guarded import inside scikit-image.
    "sklearn",
    # Tcl/Tk: matplotlib runs on QtAgg; nothing uses tkinter.
    "tkinter",
    "_tkinter",
]

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
            # QtAgg for the GUI; Agg/PDF/PS/SVG/PGF for savefig()
            "backends": ["QtAgg", "Agg", "PDF", "PS", "SVG", "PGF"],
        },
    },
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ==============================================================================
# Binary Pruning
# ==============================================================================

# CUDA math libraries. PyInstaller finds them as link-time dependencies of
# CuPy's wrapper modules (cupy_backends/cuda/libs/cublas.pyd, cupy/cuda/cufft.pyd,
# ...), but CuPy 13 imports those wrappers lazily, only when cupy.linalg,
# cupy.fft, cupy.random, cupy.sparse or cupy.dot/matmul run on the GPU. Neither
# BabelBrain (cupyx.scipy.ndimage label/binary_closing/median_filter/
# affine_transform, RawModule kernels) nor BabelViscoFDTD (RawModule) calls any
# of those, and the FDTD kernels need only NVRTC and the CUDA runtime. Together
# these libraries are most of the size of the Windows installer.
# Set BABELBRAIN_KEEP_CUDA_MATH_LIBS=1 to bundle them anyway.
CUDA_MATH_LIBS = ("cublas", "cufft", "curand", "cusolver", "cusparse",
                  "cutensor", "nccl", "cudnn")


def is_cuda_math_lib(dest_name):
    name = os.path.basename(dest_name).lower()
    if name.endswith(".dll"):  # Windows: cublas64_13.dll, cublasLt64_13.dll, ...
        stem = name
    elif name.startswith("lib") and ".so" in name:  # Linux: libcublas.so.13, ...
        stem = name[3:]
    else:  # CuPy's own cublas.*.pyd / cublas.*.so wrapper modules stay
        return False
    return stem.startswith(CUDA_MATH_LIBS)


if os.environ.get("BABELBRAIN_KEEP_CUDA_MATH_LIBS") != "1":
    pruned = [b for b in a.binaries if is_cuda_math_lib(b[0])]
    a.binaries = [b for b in a.binaries if not is_cuda_math_lib(b[0])]
    print("\nPruned CUDA math libraries:\n" + "\n".join(b[0] for b in pruned))


def check_no_mkl(toc):
    """MKL is ~600 MB with all its CPU variants. The environments pin OpenBLAS;
    if a solver change brings MKL back, say so loudly in the build log."""
    mkl = [b[0] for b in toc if os.path.basename(b[0]).lower().startswith(("mkl_", "libmkl_"))]
    if mkl:
        print("\n" + "!" * 78 + "\nWARNING: MKL is being bundled (%d files). Pin libblas=*=*openblas "
              "in the environment file.\n" % len(mkl) + "\n".join(mkl) + "\n" + "!" * 78)


check_no_mkl(a.binaries)

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
