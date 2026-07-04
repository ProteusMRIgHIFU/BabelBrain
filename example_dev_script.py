"""bb is the live BabelBrain widget. Add/remove/reorder freely."""
bb.testing_error = False

# --- Step 1: Domain generation ---
print("[dev] Step 1: CalculatePlanningMask")
bb.Widget.CalculatePlanningMask.click()
wait_until(bb.Widget.tabWidget.isEnabled)   # re-enabled when the worker finishes
wait(1000)                                   # let plots draw

# --- Step 2: Acoustic field on every trajectory tab ---
bb.Widget.tabWidget.setCurrentIndex(1)
n_tabs = bb.AcSim._txTabs.count()
for i in range(n_tabs):
    print(f"[dev] Step 2: CalculateAcField (tab {i}/{n_tabs - 1})")
    bb.AcSim._txTabs.setCurrentIndex(i)   # repoints bb.AcSim.Widget to this tab
    bb.AcSim.Widget.CalculateAcField.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3600000)
    wait(500)

# --- CombineTrajectories (only present/enabled once all fields are done) ---
if hasattr(bb.AcSim.Widget, 'CombineTrajectories') \
        and bb.AcSim.Widget.CombineTrajectories.isEnabled():
    print("[dev] Step 2: CombineTrajectories")
    bb.AcSim.Widget.CombineTrajectories.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3600000)
    wait(500)

bb.Widget.tabWidget.setCurrentIndex(2)
for i in range(n_tabs):
    print(f"[dev] Step 3: CalculateThermal (tab {i}/{n_tabs - 1})")
    bb.ThermalSim._txTabs.setCurrentIndex(i)   # repoints bb.ThermalSim.Widget to this tab
    bb.ThermalSim.Widget.CalculateThermal.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3600000)
    wait(500)

if hasattr(bb.ThermalSim.Widget, 'CombineTrajectories') \
        and bb.ThermalSim.Widget.CombineTrajectories.isEnabled():
    print("[dev] Step 3: CombineTrajectories")
    bb.ThermalSim.Widget.CombineTrajectories.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=3600000)
    wait(500)

print("[dev] Script done. Window is yours — interact away.")