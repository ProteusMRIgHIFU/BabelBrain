# Scripting BabelBrain (Coding / API)

BabelBrain can be driven from a Python script, so the whole planning →
acoustic → thermal pipeline can be **automated** without clicking through the
GUI. This is meant for batch processing, reproducible runs, and integration
testing by third parties.

A script drives the *same* operations you would perform by hand in the GUI —
selecting inputs, adjusting the frequency, setting the steering, running each
step — so this section documents those high-level, GUI-equivalent actions
rather than the internal solver functions.

!!! note
    Scripting requires a GPU-capable machine, exactly like the interactive
    application. The script runs *inside* BabelBrain, so it has full access to
    the live application objects.

## Running a script

Use the `--execute` option, both in a source checkout and in the packaged
(frozen) application:

```bash
# Source checkout
python BabelBrain/BabelBrain.py --execute my_script.py

# Frozen distribution
#   Windows
BabelBrain.exe --execute my_script.py
#   macOS
/Applications/BabelBrain.app/Contents/MacOS/BabelBrain --execute my_script.py
```

An inline snippet can be run with `--code` instead of a file:

```bash
python BabelBrain/BabelBrain.py --code "print('hello from BabelBrain')"
```

### Command-line options

| Option | Effect |
| --- | --- |
| `--execute SCRIPT.py` | Run a Python script against the application, then exit. |
| `--code "…"` | Run an inline Python snippet, then exit. |
| `--headless` | Run with no visible window (Qt *offscreen* platform). For CI on a GPU runner. |
| `--keep-open` | Leave BabelBrain open and interactive **after** the script finishes (ignored with `--headless`). By default the application closes when the script ends. |
| `--do-not-use-last-selection` | Do **not** seed inputs from the last GUI session. By default a script is seeded from your last selection and any explicit `launch()` inputs override it. |

### Exit codes

A scripted run returns a process **exit code**, so it drops straight into a CI
pipeline or a shell test:

* `0` — the script ran to completion with no error.
* non-zero — an uncaught exception, a failed `assert`, `fail(...)`, or
  `check_no_error(...)` (a full traceback is printed to `stderr`).

```bash
python BabelBrain/BabelBrain.py --headless --execute integration_test.py && echo PASS || echo FAIL
```

## The scripting API

When your script runs, the following names are already available in its global
scope (no imports needed):

| Name | Purpose |
| --- | --- |
| `launch(**inputs)` | Build a ready-to-drive BabelBrain window from a set of inputs. Returns the application object, referred to below as `bb`. |
| `launch_from_last_selection(**overrides)` | Same, but seeded from your last GUI selection; `**overrides` win. |
| `wait_until(cond, timeout_ms=…)` | Run the event loop until `cond()` is true (e.g. a step finished). Raises `TimeoutError` on timeout. |
| `wait(ms)` | Run the event loop for a fixed time (e.g. let a plot render). |
| `auto_answer_dialogs(question=…)` | Auto-answer pop-up dialogs so an unattended run never blocks (see below). |
| `restore_dialogs()` | Restore normal (interactive) dialog behaviour. |
| `check_no_error(bb)` | Raise if a BabelBrain worker flagged an error (`bb.testing_error`). Call after each step. |
| `reset_advanced_config(bb)` | Reset all advanced-configuration parameters in `bb.Config` to their defaults (see [Advanced configuration](advanced_config.md)). Opt-in; by default the previous session's values are kept. |
| `fail(msg)` | Abort the script (non-zero exit) with a message. |
| `QMessageBox` | The Qt message-box enum, for `auto_answer_dialogs`. |

### `launch()` — selecting inputs

`launch()` mirrors the **input-selection dialog** of the GUI. Every field is
optional; anything you omit falls back to your last GUI selection (unless you
passed `--do-not-use-last-selection`).

```python
bb = launch(
    t1w="/data/sub-01/T1W.nii.gz",
    simbnibs="/data/sub-01/m2m_sub-01/",   # SimNIBS output folder
    simbnibs_type="charm",                 # "charm" | "headreco"
    trajectory="/data/sub-01/target.txt",
    trajectory_type="brainsight",          # "brainsight" | "slicer"
    thermal_profile="/data/profile.yaml",
    transducer="CTX_500",                  # transducer name as in the GUI list
    ct_type="CT",                          # "NONE" | "CT" | "ZTE" | "PETRA"
    ct="/data/sub-01/CT.nii.gz",
    coreg_ct=True,
    frequency_khz=500,                     # Step-1 frequency
    ppw=6,                                 # Step-1 points-per-wavelength
    hu_threshold=300,                      # Step-1 bone threshold (CT only)
    # gpu="Apple M3 Max", backend="Metal", # else the auto-selected GPU is used
    output_path="/data/sub-01/output/",
)
```

| `launch()` argument | GUI equivalent |
| --- | --- |
| `t1w` | T1W NIfTI file |
| `simbnibs`, `simbnibs_type` | SimNIBS folder + tool (`charm`/`headreco`) |
| `trajectory`, `trajectory_type` | Trajectory file + source (`brainsight`/`slicer`) |
| `ct_type`, `ct`, `coreg_ct` | CT/ZTE/PETRA selection, file, and coregistration toggle |
| `ct_mapping` | CT mapping profile (tuple as shown in the CT-mapping list) |
| `thermal_profile` | Thermal profile YAML |
| `transducer` | Transducer model |
| `gpu`, `backend` / `computing` | Computing engine (GPU + backend) |
| `multipoint_type`, `multipoint` | Multi-point steering selection + profile |
| `frequency_khz`, `ppw`, `hu_threshold` | Step-1 frequency / PPW / bone threshold |
| `output_path` | Output folder |

### Handling pop-up dialogs

When a result already exists, BabelBrain asks whether to *recalculate* or
*reload*. In an unattended run you answer these once, up front:

```python
auto_answer_dialogs(question=QMessageBox.Yes)  # always recalculate
auto_answer_dialogs(question=QMessageBox.No)   # always reload existing results
```

## A complete example

The following script runs the full pipeline for one target and exits `0` if it
succeeds. See **[Scriptable operations](operations.md)** for the full catalogue
of GUI actions used here.

```python
# full_pipeline.py  —  python BabelBrain/BabelBrain.py --execute full_pipeline.py
bb = launch(
    t1w="/data/sub-01/T1W.nii.gz",
    simbnibs="/data/sub-01/m2m_sub-01/",
    trajectory="/data/sub-01/target.txt",
    thermal_profile="/data/profile.yaml",
    transducer="CTX_500",
    ct_type="CT", ct="/data/sub-01/CT.nii.gz",
    frequency_khz=500, ppw=6, hu_threshold=300,
    output_path="/data/sub-01/output/",
)
auto_answer_dialogs(question=QMessageBox.Yes)  # deterministic: recalculate

# --- Step 1: domain generation ---
bb.testing_error = False
bb.Widget.CalculatePlanningMask.click()
wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
check_no_error(bb)

# --- Step 2: acoustic field (every trajectory tab) ---
bb.Widget.tabWidget.setCurrentIndex(1)
for i in range(bb.AcSim._txTabs.count()):
    bb.AcSim._txTabs.setCurrentIndex(i)
    # (optional) adjust steering before running — see Scriptable operations
    if hasattr(bb.AcSim.Widget, 'ZSteeringSpinBox'):
        bb.AcSim.Widget.ZSteeringSpinBox.setValue(0.0)
    bb.AcSim.Widget.CalculateAcField.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3_600_000)
    check_no_error(bb)

# --- Step 3: thermal ---
bb.Widget.tabWidget.setCurrentIndex(2)
for i in range(bb.ThermalSim._txTabs.count()):
    bb.ThermalSim._txTabs.setCurrentIndex(i)
    bb.ThermalSim.Widget.CalculateThermal.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
    check_no_error(bb)

print("Pipeline OK.")   # falling off the end -> exit code 0
```

## How a script is structured

Every script follows the same three moves:

1. **`launch(...)`** to open BabelBrain with your inputs and get the `bb` object.
2. **Trigger an action** by clicking a button or setting a control, exactly as
   the GUI does — e.g. `bb.Widget.CalculatePlanningMask.click()`.
3. **`wait_until(...)`** for the step to finish (the step tabs re-enable when a
   worker completes), then `check_no_error(bb)`.

The three "steps" of the GUI are reached through three objects on `bb`:

| Step | Object | Notes |
| --- | --- | --- |
| Step 1 — Planning / domain | `bb.Widget` | The main window controls. |
| Step 2 — Acoustic simulation | `bb.AcSim` | `bb.AcSim.Widget` for controls; `bb.AcSim._txTabs` selects the trajectory tab. |
| Step 3 — Thermal simulation | `bb.ThermalSim` | `bb.ThermalSim.Widget` for controls; `bb.ThermalSim._txTabs` selects the trajectory tab. |

The exact controls available for each step are listed in
**[Scriptable operations](operations.md)**.
