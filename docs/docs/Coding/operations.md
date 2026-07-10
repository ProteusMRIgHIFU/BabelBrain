# Scriptable operations

This page lists the high-level operations that can be automated from a script,
grouped by the three pipeline steps. Each entry maps a **GUI action** to the
equivalent **scripting call**. See
**[Scripting BabelBrain](scripting.md)** for how to run a script and obtain the
`bb` object.

## Conventions

* Controls are reached through `bb.Widget` (Step 1), `bb.AcSim.Widget`
  (Step 2), and `bb.ThermalSim.Widget` (Step 3).
* A control is set with the usual Qt calls:
    * buttons — `.click()`
    * spin boxes — `.setValue(number)`
    * check boxes — `.setChecked(True/False)`
    * drop-downs — `.setCurrentText("label")` or `.setCurrentIndex(n)`
* After triggering a step, **always** wait for it to finish and check for
  errors:

    ```python
    bb.Widget.CalculatePlanningMask.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
    check_no_error(bb)
    ```

!!! warning "Step-2 controls are transducer-specific"
    The acoustic-step controls depend on the selected transducer (a single-element
    device has no electronic steering, a phased array does, etc.). Guard optional
    controls with `hasattr` so one script works across devices:

    ```python
    if hasattr(bb.AcSim.Widget, 'ZSteeringSpinBox'):
        bb.AcSim.Widget.ZSteeringSpinBox.setValue(0.0)
    ```

### Multiple trajectories

When a plan contains several trajectories, Step 2 and Step 3 present one tab per
trajectory. Select a tab before acting on it:

```python
for i in range(bb.AcSim._txTabs.count()):        # Step 2 tabs
    bb.AcSim._txTabs.setCurrentIndex(i)
    bb.AcSim.Widget.CalculateAcField.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3_600_000)
    check_no_error(bb)
```

Step 3 uses `bb.ThermalSim._txTabs` in the same way.

---

## Step 1 — Planning / domain generation

Controls are on `bb.Widget`.

| GUI action | Scripting call |
| --- | --- |
| Set the **frequency** (kHz) | `bb.Widget.USMaskkHzDropDown.setCurrentText("500")` — or, preferably, `launch(frequency_khz=500)` |
| Set **points per wavelength (PPW)** | `bb.Widget.USPPWSpinBox.setProperty('UserData', 6)` — or `launch(ppw=6)` |
| Set the **bone HU threshold** (CT only) | `bb.Widget.HUThresholdSpinBox.setValue(300)` — or `launch(hu_threshold=300)` |
| Adjust the **ZTE/PETRA range** | `bb.Widget.ZTERangeSlider` (range slider) |
| **Run domain generation** | `bb.Widget.CalculatePlanningMask.click()` |
| Set **transparency** of the overlay | `bb.Widget.TransparencyScrollBar.setValue(n)` |
| **Hide/show markers** | `bb.Widget.HideMarkscheckBox.setChecked(True)` |

Frequency, PPW and the HU threshold are best set through `launch()` (they are
applied before Step 1 runs). Setting them on `bb.Widget` afterwards is
equivalent and useful if you re-run Step 1 with different values in one script.

```python
# Sweep the bone threshold and re-run Step 1
for hu in (200, 300, 400):
    bb.Widget.HUThresholdSpinBox.setValue(hu)
    bb.Widget.CalculatePlanningMask.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
    check_no_error(bb)
```

---

## Step 2 — Acoustic simulation

Controls are on `bb.AcSim.Widget`; select the trajectory tab with
`bb.AcSim._txTabs`. Availability depends on the transducer (see the warning
above).

| GUI action | Scripting call |
| --- | --- |
| **Run the acoustic field** | `bb.AcSim.Widget.CalculateAcField.click()` |
| Set **electronic steering** (phased arrays) | `bb.AcSim.Widget.XSteeringSpinBox.setValue(x)` · `YSteeringSpinBox` · `ZSteeringSpinBox` |
| Set **cone-to-focus / TPO distance** | `bb.AcSim.Widget.TPODistanceSpinBox.setValue(d)` (some devices use `DistanceConeToFocusSpinBox`) |
| Set **mechanical adjustment** | `XMechanicSpinBox` · `YMechanicSpinBox` · `ZMechanicSpinBox` · `ZRotationSpinBox`, then `bb.AcSim.Widget.CalculateMechAdj.click()` |
| Set **maximum depth** | `bb.AcSim.Widget.MaxDepthSpinBox.setValue(d)` |
| Toggle **refocusing** | `bb.AcSim.Widget.RefocusingcheckBox.setChecked(True)` |
| Select the **multi-focus point** to display | `bb.AcSim.Widget.SelCombinationDropDown.setCurrentIndex(n)` |
| Change the **display / view plane** | `bb.AcSim.Widget.DisplayDropDown` · `SelViewDropDown` |
| Show **water-only** result | `bb.AcSim.Widget.ShowWaterResultscheckBox.setChecked(True)` |
| Move the **slice** position | `bb.AcSim.Widget.IsppaScrollBar.setValue(n)` |
| **Combine trajectories** (multi-target) | `bb.AcSim.Widget.CombineTrajectories.click()` |

`CombineTrajectories` only exists / is enabled once the acoustic field of every
trajectory has been computed:

```python
if hasattr(bb.AcSim.Widget, 'CombineTrajectories') \
        and bb.AcSim.Widget.CombineTrajectories.isEnabled():
    bb.AcSim.Widget.CombineTrajectories.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3_600_000)
    check_no_error(bb)
```

### Reading values back

Some read-outs are exposed as label "UserData" — e.g. the skin-to-target
distance used to set a cone position:

```python
skin_to_target = bb.AcSim.Widget.DistanceSkinLabel.property('UserData')
```

---

## Step 3 — Thermal simulation

Controls are on `bb.ThermalSim.Widget`; select the trajectory tab with
`bb.ThermalSim._txTabs`.

| GUI action | Scripting call |
| --- | --- |
| **Run the thermal simulation** | `bb.ThermalSim.Widget.CalculateThermal.click()` |
| Select the **timing combination** (DC/PRF/duration) | `bb.ThermalSim.Widget.SelCombinationDropDown.setCurrentIndex(n)` |
| Change **what is displayed** | `bb.ThermalSim.Widget.DisplayDropDown.setCurrentIndex(n)` |
| Change the **view plane** (XZ / YZ / XY) | `bb.ThermalSim.Widget.SelViewDropDown.setCurrentText("YZ")` |
| Set the target **Isppa** (in brain) | `bb.ThermalSim.Widget.IsppaSpinBox.setValue(v)` |
| Set the **Isppa in water** | `bb.ThermalSim.Widget.IsppaWaterSpinBox.setValue(v)` |
| Move the **slice** position | `bb.ThermalSim.Widget.IsppaScrollBar.setValue(n)` |
| **Hide/show markers** | `bb.ThermalSim.Widget.HideMarkscheckBox.setChecked(True)` |
| **Export the summary** (CSV) | `bb.ThermalSim.Widget.ExportSummary.click()` |
| **Export the maps** (NIfTI) | `bb.ThermalSim.Widget.ExportMaps.click()` |
| **Combine trajectories** (multi-target) | `bb.ThermalSim.Widget.CombineTrajectories.click()` |

!!! note "Exports in an unattended run"
    In scripting mode the "save file" dialogs of `ExportSummary` / `ExportMaps`
    are bypassed and the files are written into the run's output folder
    automatically, so an unattended export never blocks.

---

## Full multi-step example

```python
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
auto_answer_dialogs(question=QMessageBox.Yes)

# Step 1
bb.testing_error = False
bb.Widget.CalculatePlanningMask.click()
wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
check_no_error(bb)

# Step 2 (all trajectories)
bb.Widget.tabWidget.setCurrentIndex(1)
for i in range(bb.AcSim._txTabs.count()):
    bb.AcSim._txTabs.setCurrentIndex(i)
    if hasattr(bb.AcSim.Widget, 'ZSteeringSpinBox'):
        bb.AcSim.Widget.ZSteeringSpinBox.setValue(0.0)
    bb.AcSim.Widget.CalculateAcField.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3_600_000)
    check_no_error(bb)

# Step 3 (all trajectories) + export
bb.Widget.tabWidget.setCurrentIndex(2)
for i in range(bb.ThermalSim._txTabs.count()):
    bb.ThermalSim._txTabs.setCurrentIndex(i)
    bb.ThermalSim.Widget.CalculateThermal.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
    check_no_error(bb)
    bb.ThermalSim.Widget.ExportSummary.click()
    bb.ThermalSim.Widget.ExportMaps.click()

print("Done.")
```
