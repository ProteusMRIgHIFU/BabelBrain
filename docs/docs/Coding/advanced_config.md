# Advanced configuration

The **Advanced Options** dialog of the GUI controls a large set of low-level
parameters (domain generation, pseudo-CT mapping, thermal modelling, …). When
scripting, you do **not** open that dialog — every one of those parameters is a
plain entry in the `bb.Config` dictionary, so you set it directly:

```python
bb = launch(...)
bb.Config['TrabecularProportion'] = 0.2       # override one advanced parameter
bb.Config['ElastixOptimizer'] = 'FiniteDifferenceGradientDescent'
```

!!! note "Out of scope"
    Two operations reachable from the Advanced Options dialog — running the
    external **PlanTUS** tool and **transducer calibration** — need heavy user
    interaction and are *not* scriptable for now. The path/parameters they use
    (`PlanTUSRoot`, `ConnectomeRoot`, `TxOptimizedWeights`) are still ordinary
    `bb.Config` entries and are listed below for completeness.

## When to set a parameter

Each parameter affects one of the pipeline steps. Set it **before** you trigger
that step. The simplest, always-correct approach is to apply all your overrides
right after `launch()` and before Step 1:

```python
bb = launch(t1w=..., simbnibs=..., trajectory=..., transducer="CTX_500",
            frequency_khz=500, output_path="/data/out/")

# --- advanced overrides ---
bb.Config['bExtractAirRegions'] = True
bb.Config['TrabecularProportion'] = 0.2
bb.Config['BaselineTemperature'] = 37.0
bb.Config['HomogenousMediumValues']['Perfusion'] = 500.0

# --- then run the pipeline as usual ---
bb.Widget.CalculatePlanningMask.click()
wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
check_no_error(bb)
```

`bb.Config` already contains every key below (initialised to the default value
shown, or to the value you last saved in the GUI). You only assign the ones you
want to change. The `HomogenousMediumValues` entry is a **nested dictionary** —
set its members individually, e.g. `bb.Config['HomogenousMediumValues']['Density'] = 1010.0`.

!!! note "Persistence"
    In the GUI, changing an advanced option persists to the next session. In a
    script, assigning to `bb.Config` only affects the current run.

---

## Domain generation (Step 1)

| `bb.Config` key | Default | Description |
| --- | --- | --- |
| `ElastixOptimizer` | `'AdaptiveStochasticGradientDescent'` | Elastix optimizer for CT/ZTE/PETRA → T1W coregistration. Alternatives that help difficult CT→T1W cases: `'FiniteDifferenceGradientDescent'`, `'QuasiNewtonLBFGS'`. |
| `TrabecularProportion` | `0.8` | Fraction of bone (along the line of sight) modelled as trabecular. Lower it (e.g. `0.1`–`0.2`) for thin bone such as the parietal. |
| `bSegmentBrainTissue` | `False` | Use SimNIBS segmentation of white/grey matter and CSF with tissue-specific acoustic/thermal properties. Requires `SimbNINBSRoot`. |
| `bForceUseBlender` | `False` | Force Blender for constructive solid geometry instead of pycork (works around rare pycork crashes). |
| `bApplyBOXFOV` | `False` | Enable a manual sub-volume (box field of view) instead of the automatic one. |
| `FOVDiameter` | `200.0` | Manual FOV diameter (mm); used when `bApplyBOXFOV` is `True`. |
| `FOVLength` | `400.0` | Manual FOV length (mm); used when `bApplyBOXFOV` is `True`. |
| `bExtractAirRegions` | `True` | Create air-region masks (CT/PETRA/ZTE). Recommended to model far-field standing waves. |
| `bDisableCTMedianFilter` | `False` | Disable the median filter applied to the CT during processing. |

---

## Pseudo-CT mapping — CT / ZTE / PETRA (Step 1)

| `bb.Config` key | Default | Description |
| --- | --- | --- |
| `bInvertZTE` | `False` | Set `True` for inverted ZTE images (e.g. GE `oZTEo`), otherwise processing errors. |
| `PetraNPeaks` | `2` | Number of histogram bins used in PETRA bone selection (matches UCL's *petra-to-ct*). |
| `PetraMRIPeakDistance` | `50` | Minimal distance between PETRA histogram bins. |
| `bGeneratePETRAHistogram` | `False` | Also generate the PETRA histogram (as in *petra-to-ct*). |
| `ZTESlope` | `-2085.0` | Slope of the linear ZTE-signal → pseudo-CT (HU) conversion. |
| `ZTEOffset` | `2329.0` | Offset of the linear ZTE → pseudo-CT conversion. |
| `PETRASlope` | `-2080.0` | Slope of the linear PETRA-signal → pseudo-CT conversion. |
| `PETRAOffset` | `2133.2` | Offset of the linear PETRA → pseudo-CT conversion. |

Adjust the slope/offset only if you have re-fitted the ZTE/PETRA → pseudo-CT
relationship for your site.

---

## Simulation & thermal

| `bb.Config` key | Default | Affects | Description |
| --- | --- | --- | --- |
| `BaselineTemperature` | `37.0` | Step 3 | Baseline (body) temperature in °C for the thermal simulation. |
| `LimitBHTEIterationsPerProcess` | `100` | Step 3 | Cap on bio-heat-transfer-equation iterations per process (memory/performance tuning). |
| `bForceNoAbsorptionSkullScalp` | `False` | Step 2/3 | Force zero acoustic absorption in skull and scalp (verification/benchmarking). |

---

## Homogeneous medium (verification / benchmarking)

Replace the segmented head with a single homogeneous medium — useful for
verification and benchmarking against analytical or reference solutions.

| `bb.Config` key | Default | Description |
| --- | --- | --- |
| `bForceHomogenousMedium` | `False` | Enable the homogeneous medium. When `False`, the values below are ignored. |

When enabled, the medium properties come from the nested
`bb.Config['HomogenousMediumValues']` dictionary:

| `HomogenousMediumValues[...]` | Default | Units | Description |
| --- | --- | --- | --- |
| `'Density'` | `1000.0` | kg/m³ | Mass density. |
| `'LongSoS'` | `1500.0` | m/s | Longitudinal speed of sound. |
| `'LongAtt'` | `5.0` | Np/m/MHz | Longitudinal attenuation. |
| `'ShearSoS'` | `0.0` | m/s | Shear speed of sound. |
| `'ShearAtt'` | `0.0` | Np/m/MHz | Shear attenuation. |
| `'ThermalConductivity'` | `0.5` | W/m/°C | Thermal conductivity. |
| `'SpecificHeat'` | `3583.0` | J/kg/°C | Specific heat capacity. |
| `'Perfusion'` | `555.0` | ml/min/kg | Blood perfusion. |
| `'Absorption'` | `0.85` | – | Absorption fraction (of attenuation converted to heat). |
| `'InitTemperature'` | `37.0` | °C | Initial temperature. |

```python
bb.Config['bForceHomogenousMedium'] = True
hm = bb.Config['HomogenousMediumValues']
hm['Density'] = 1000.0
hm['LongSoS'] = 1500.0
hm['LongAtt'] = 5.0
```

---

## Output / debugging

| `bb.Config` key | Default | Description |
| --- | --- | --- |
| `bSaveStress` | `False` | Also save the acoustic **stress** field from Step 2. |
| `bSaveDisplacement` | `False` | Also save the acoustic **displacement** field from Step 2. |

---

## External-tool roots

These are folder paths used by optional integrations. The *operations* that use
them (PlanTUS, brain segmentation, calibration) are not scriptable here, but the
paths are ordinary `bb.Config` entries.

| `bb.Config` key | Default | Description |
| --- | --- | --- |
| `SimbNINBSRoot` | `'...'` | SimNIBS installation root, required when `bSegmentBrainTissue` is `True`. |
| `PlanTUSRoot` | `'...'` | PlanTUS installation root (PlanTUS operation not scriptable). |
| `ConnectomeRoot` | `'...'` | Connectome Workbench root (used by PlanTUS). |
| `TxOptimizedWeights` | `{tx: '' , …}` | Per-transducer path to a calibration `.h5` file. Dictionary keyed by transducer name; see [Tx calibration](../Advanced/TransducerCalibration.md). |
