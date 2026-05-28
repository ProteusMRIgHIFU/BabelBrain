# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is BabelBrain

BabelBrain is a GUI application for prospective modeling of transcranial focused ultrasound (LIFU) for neuromodulation. It computes acoustic and thermal fields transmitted through the skull using GPU-accelerated FDTD simulations. The external solver is `BabelViscoFDTD` (v1.2.4+).

## Environment Setup

The project uses Conda. Choose the appropriate environment file:

```bash
conda env create -f environment_mac_arm64-311.yml   # macOS ARM64 (Python 3.11)
conda env create -f environment_mac_x64-310.yml     # macOS Intel (Python 3.10)
conda env create -f environment_linux.yml           # Linux
conda env create -f environment_win.yml             # Windows
```

## Running the Application

```bash
python BabelBrain/BabelBrain.py
python BabelBrain/BabelBrain.py -bInUseWithBrainsight  # Brainsight integration mode
```

## Building a Standalone App

```bash
pyinstaller BabelBrain.spec --noconfirm  # output in BabelBrain/dist/
```

## Testing

Tests require `Tests/config.ini` (copy from `Tests/example_config.ini` and configure data folder path, GPU device, reference output folders).

```bash
pytest                                               # all tests
pytest -k "test_FileManager"                        # by function name
pytest -m "step1"                                   # by step marker
pytest -m "gpu and low_res"                         # GPU unit tests at low resolution
pytest -k "test_full_pipeline_normal" -m "basic_babelbrain_params"  # E2E tests
```

Test markers: `slow`, `gpu`, `step1/2/3`, `low_res/medium_res/high_res/stress_res`, `tol_0/1/5`, `basic_babelbrain_params`, `generate_outputs`. HTML reports go to `PyTest_Reports/report.html`.

## Architecture

### Three-Step Workflow

The application implements a three-step pipeline:

1. **Step 1 — Domain Generation** (`CalculateMaskProcess.py`): Medical image preprocessing → tissue segmentation → simulation domain. Runs as a worker thread (`RunMaskGeneration`). Heavy use of `SimpleITK`, `elastix` (coregistration), and GPU functions.

2. **Step 2 — Acoustic Simulation** (`CalculateFieldProcess.py` + `TranscranialModeling/`): Water-only field first, then transcranial field with skull corrections via BabelViscoFDTD FDTD solver.

3. **Step 3 — Thermal Analysis** (`ThermalModeling/CalculateTemperatureEffects.py`): Temperature/dose maps from ultrasound parameters (duty cycle, PRF, duration, intensity).

### Key Modules

- **`BabelBrain/BabelBrain.py`** — Main `QWidget`-based GUI orchestrator (~1490 lines). Drives the 3-step workflow, manages configuration, and spawns worker threads.
- **`BabelBrain/BabelDatasetPreps.py`** — Image registration, tissue segmentation, and domain generation (~1177 lines). Core Step 1 logic.
- **`BabelBrain/FileManager.py`** — Hash-based incremental caching. Hash signatures on input files prevent redundant recalculations when inputs haven't changed.
- **`TranscranialModeling/BabelIntegrationBASE.py`** — Base class for FDTD integration (~2848 lines). Subclassed for each transducer type.
- **`ThermalModeling/CalculateTemperatureEffects.py`** — Thermal effects calculation (~1185 lines).

### Transducer Model Architecture

`TranscranialModeling/` contains implementations for 20+ transducer devices. Each device has:
- A `BabelIntegration<Device>.py` file subclassing `BabelIntegrationBASE`
- A `BabelBrain/Babel_<Device>/` folder with transducer geometry CSVs and config files

Base class hierarchy: `BabelIntegrationBASE` ← `_BabelBaseTx` / `_BabelBasePhasedArray`

### GPU Backend

`BabelBrain/GPUFunctions/` provides GPU-accelerated operations (binary closing, median filter, resampling, voxelization, labeling) with four backends: **CUDA, OpenCL, Metal, MLX**. Each operation has a dispatcher that selects the appropriate backend kernel.

### CT/MRI Input Types

The pipeline supports multiple skull imaging modalities via `CTZTEProcessing.py`:
- **CT** — Direct Hounsfield units
- **ZTE** — Zero echo time MRI → pseudo-CT conversion
- **PETRA** — Pointwise encoding time reduction with radial acquisition → pseudo-CT

### File I/O

- Medical images: NIfTI (`.nii`/`.nii.gz`) via `nibabel` and `SimpleITK`
- Simulation data: HDF5 (`.h5`) via `h5py`
- Transducer geometry: CSV files in `BabelBrain/Babel_<Device>/`
- Thermal profiles: YAML files in `Profiles/`
- Transform matrices: `.mat` (Brainsight) and `.tfm`/`.h5` (3D Slicer)

### Configuration Persistence

User selections (file paths, parameters) are saved to a `.ini` file alongside the input data so sessions can be resumed. Advanced parameters are in `Options/Options.py`.
