"""
Example BabelBrain integration script.

Run against the source tree:
    python BabelBrain/BabelBrain.py --execute BabelBrain/scripting_example.py             # GUI shown, closes at end
    python BabelBrain/BabelBrain.py --execute BabelBrain/scripting_example.py --keep-open  # GUI stays open/interactive
    python BabelBrain/BabelBrain.py --execute BabelBrain/scripting_example.py --headless   # no window (CI)

Run against the frozen distribution:
    Windows:  BabelBrain.exe --execute scripting_example.py
    macOS:    /Applications/BabelBrain.app/Contents/MacOS/BabelBrain --execute scripting_example.py

By default the run closes BabelBrain when the script finishes and exits with
code 0 on success / non-zero on error — so it drops straight into CI. Add
--keep-open to leave the window live afterwards (dev_driver-style); the process
then exits when you close the window, still returning the script's exit code.

Injected names (see scripting.make_namespace): launch, launch_from_last_selection,
wait_until, wait, auto_answer_dialogs, restore_dialogs, check_no_error, fail,
QMessageBox.  Edit the paths below to point at your own data.
"""

# By default, launch() is seeded from your last GUI selection (lastselection.yaml)
# and the explicit inputs below override it. Pass --do-not-use-last-selection to
# require every input to be specified here instead.


# --- Option A: lean on the last GUI selection (the default) ---
#   launch() starts from lastselection.yaml; pass only what you override:
# for example
# bb = launch(frequency_khz=500)
# otherwise just call launch alone

bb = launch()

# # --- Option B: specify every input explicitly (portable, CI-friendly) ---
# bb = launch(
#     t1w="/path/to/T1W.nii.gz",
#     simbnibs="/path/to/m2m_subject/",
#     simbnibs_type="charm",
#     trajectory="/path/to/trajectory.txt",
#     trajectory_type="brainsight",
#     thermal_profile="/path/to/thermal_profile.yaml",
#     transducer="CTX_500",
#     ct_type="CT",                 # NONE | CT | ZTE | PETRA
#     ct="/path/to/ct.nii.gz",
#     coreg_ct=True,
#     frequency_khz=500,
#     ppw=6,
#     hu_threshold=300,
#     # gpu="...", backend="Metal",  # else SelFiles' auto-selected GPU is used
#     output_path="/path/to/output_folder/",
# )

# Cached results would otherwise raise a "recalculate?" prompt — answer Yes so
# the run is deterministic and never blocks. Select No to reload results by default.
auto_answer_dialogs(question=QMessageBox.No)

# --- Step 1: domain generation ---
print("Step 1: CalculatePlanningMask")
bb.testing_error = False
bb.Widget.CalculatePlanningMask.click()
wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
check_no_error(bb)

# --- Step 2: acoustic field on every trajectory tab ---
bb.Widget.tabWidget.setCurrentIndex(1)
n_tabs = bb.AcSim._txTabs.count()
for i in range(n_tabs):
    print(f"Step 2: CalculateAcField (tab {i})")
    bb.AcSim._txTabs.setCurrentIndex(i)
    bb.AcSim.Widget.CalculateAcField.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3_600_000)
    check_no_error(bb)

# --- Step 3: thermal ---
bb.Widget.tabWidget.setCurrentIndex(2)
for i in range(n_tabs):
    print(f"Step 3: CalculateThermal (tab {i})")
    bb.ThermalSim._txTabs.setCurrentIndex(i)
    bb.ThermalSim.Widget.CalculateThermal.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
    check_no_error(bb)

print("Integration run OK.")   # falling off the end -> exit code 0
