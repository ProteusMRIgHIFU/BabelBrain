"""
Development driver for BabelBrain.

Runs the app, drives a scripted chain of UI interactions up to a point, then
hands the *live* window back to you for manual interaction. No pytest.

Run from the repo root:

    python BabelBrain/dev_driver.py

Edit `script()` below to change the chain. That's the only part you touch.
"""
import os
import sys

# sys.path[0] is this file's dir (BabelBrain/) so the app's internal imports
# (SelFiles.SelFiles, etc.) resolve exactly as they do for BabelBrain.py.
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from BabelBrain import (
    BabelBrain,
    GetLatestSelection,
    _apply_color_scheme,
    resource_path,
)
from SelFiles.SelFiles import SelFiles


# --------------------------------------------------------------------------
# Helpers (the pytest-qt replacements, built on Qt's own QTest)
# --------------------------------------------------------------------------
def wait_until(cond, timeout_ms=900000, interval_ms=100):
    """Pump the event loop until cond() is true. Worker-thread signals still
    fire while we wait; the GUI stays responsive. Mirrors qtbot.waitUntil."""
    elapsed = 0
    while not cond():
        QTest.qWait(interval_ms)
        elapsed += interval_ms
        if elapsed >= timeout_ms:
            raise TimeoutError("wait_until timed out")


def wait(ms):
    """Pump the event loop for a fixed time (e.g. let plots draw)."""
    QTest.qWait(ms)


def auto_answer_dialogs(question=QMessageBox.No,
                        warning=QMessageBox.Ok,
                        critical=QMessageBox.Ok):
    """Make blocking QMessageBox.* static calls return canned answers so the
    script never stalls on a modal. Flip `question` to QMessageBox.Yes to
    force recalculation instead of reload. Call again mid-script to change."""
    QMessageBox.question = staticmethod(lambda *a, **k: question)
    QMessageBox.warning = staticmethod(lambda *a, **k: warning)
    QMessageBox.critical = staticmethod(lambda *a, **k: critical)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)


# --------------------------------------------------------------------------
# EDIT ME: the chain of interactions to reach the point you care about.
# --------------------------------------------------------------------------
def script(bb):
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


# --------------------------------------------------------------------------
# Boot: reproduces BabelBrain.main() closely, minus arg parsing / sys.exit.
# --------------------------------------------------------------------------
def _prefill_selfiles(selwidget):
    """Best-effort prefill from your last real GUI session so you don't
    re-pick files every run. Anything missing you set in the shown dialog."""
    cfg = GetLatestSelection()
    if not cfg:
        return
    ui = selwidget.ui
    for key, setter in [
        ('simbnibs_path', ui.SimbNIBSlineEdit.setText),
        ('T1W',           ui.T1WlineEdit.setText),
        ('Mat4Trajectory', ui.TrajectorylineEdit.setText),
        ('ThermalProfile', ui.ThermalProfilelineEdit.setText),
        ('CT_or_ZTE_input', ui.CTlineEdit.setText),
    ]:
        if cfg.get(key):
            setter(cfg[key])
    if 'CTType' in cfg:
        ui.CTTypecomboBox.setCurrentIndex(cfg['CTType'])
    if cfg.get('TxSystem'):
        selwidget.SelectTxSystem(cfg['TxSystem'])


def main():
    if os.getenv('FSLDIR') is None:
        os.environ['FSLDIR'] = '/usr/local/fsl'
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        os.environ['PATH'] = os.environ['PATH'] + ':' + '/usr/local/fsl/bin'

    app = QApplication([])
    _apply_color_scheme(app)
    app.setWindowIcon(QIcon(os.path.join(resource_path(), 'Proteus-Alciato-logo.png')))

    # Pick/confirm input files once per session via the normal dialog.
    selwidget = SelFiles()
    _prefill_selfiles(selwidget)
    if selwidget.exec() == -1:
        return

    widget = BabelBrain(selwidget, AltOutputFilesPath=None)
    widget.show()

    auto_answer_dialogs(question=QMessageBox.No)  # "Mask exists? -> reload"

    # Kick off the script once the loop is running; the loop keeps running
    # afterwards so the window stays interactive.
    QTimer.singleShot(500, lambda: script(widget))
    app.exec()


if __name__ == '__main__':
    main()
