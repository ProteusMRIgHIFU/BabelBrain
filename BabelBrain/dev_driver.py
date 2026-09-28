"""
Development driver for BabelBrain.

Runs the app, drives a scripted chain of UI interactions up to a point, then
hands the *live* window back to you for manual interaction. No pytest.

Run from the repo root:

    python BabelBrain/dev_driver.py                 # runs built-in script() below
    python BabelBrain/dev_driver.py my_run.py       # runs an external snippet

An external snippet is plain Python containing ONLY the body you'd put in
script() — no def, no indentation. These names are pre-bound in its scope:
    bb, wait_until, wait, auto_answer_dialogs, restore_dialogs, QMessageBox
so you can edit/throw away snippets without touching (or committing) this
driver. Errors in a snippet print a traceback but leave the window live.
"""
import argparse
import os
import traceback

# sys.path[0] is this file's dir (BabelBrain/) so the app's internal imports
# (SelFiles.SelFiles, etc.) resolve exactly as they do for BabelBrain.py.
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from BabelBrain import (
    BabelBrain,
    GetLatestSelection,
    _apply_color_scheme,
)
from SelFiles.SelFiles import SelFiles
# The event-loop / dialog helpers are shared with the frozen scripting engine
# (BabelBrain --execute) so both behave identically.
from scripting import wait_until, wait, auto_answer_dialogs, restore_dialogs
from Utils.paths import resource_path


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

    print("[dev] Script done. Window is yours — interact away.")


def run_script_file(bb, path):
    """Exec an external snippet with the helper names pre-bound. Compiling with
    the real path means tracebacks point at the snippet's own line numbers."""
    with open(path) as f:
        code = f.read()
    ns = {
        'bb': bb,
        'wait_until': wait_until,
        'wait': wait,
        'auto_answer_dialogs': auto_answer_dialogs,
        'restore_dialogs': restore_dialogs,
        'QMessageBox': QMessageBox,
    }
    try:
        exec(compile(code, path, 'exec'), ns)
        print("[dev] Snippet done. Window is yours — interact away.")
    except Exception:
        traceback.print_exc()
        print(f"[dev] Snippet '{path}' errored — window left live for debugging.")


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
    parser = argparse.ArgumentParser(description="BabelBrain dev driver")
    parser.add_argument('script_file', nargs='?', default=None,
                        help="Optional Python snippet to run instead of built-in script()")
    args = parser.parse_args()

    if os.getenv('FSLDIR') is None:
        os.environ['FSLDIR'] = '/usr/local/fsl'
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        os.environ['PATH'] = os.environ['PATH'] + ':' + '/usr/local/fsl/bin'

    app = QApplication([])
    _apply_color_scheme(app)
    app.setWindowIcon(QIcon(os.path.join(resource_path(__file__), 'Proteus-Alciato-logo.png')))

    # Pick/confirm input files once per session via the normal dialog.
    selwidget = SelFiles()
    _prefill_selfiles(selwidget)
    if selwidget.exec() == -1:
        return

    widget = BabelBrain(selwidget, AltOutputFilesPath=None)
    widget.show()

    auto_answer_dialogs(question=QMessageBox.No)  # "Mask exists? -> reload"

    # Kick off the chain once the loop is running; the loop keeps running
    # afterwards so the window stays interactive. Always restore real dialog
    # behaviour when the chain ends so handed-back modals prompt you normally.
    def runner():
        try:
            if args.script_file:
                run_script_file(widget, args.script_file)
            else:
                script(widget)
        finally:
            restore_dialogs()
            print("[dev] Dialogs restored — modals will prompt you normally now.")

    QTimer.singleShot(500, runner)
    app.exec()


if __name__ == '__main__':
    main()
