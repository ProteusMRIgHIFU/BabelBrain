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
import sys
import traceback

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


# Pristine QMessageBox statics, captured before any patching so we can put the
# real (user-prompting) behaviour back once the automated chain is done.
_ORIG_DIALOGS = {name: getattr(QMessageBox, name)
                 for name in ('question', 'warning', 'critical', 'information')}


def auto_answer_dialogs(question=QMessageBox.No,
                        warning=QMessageBox.Ok,
                        critical=QMessageBox.Ok):
    """Make blocking QMessageBox.* static calls return canned answers so the
    script never stalls on a modal. Flip `question` to QMessageBox.Yes to
    force recalculation instead of reload. Call again mid-script to change.
    Undone by restore_dialogs() after the chain finishes."""
    QMessageBox.question = staticmethod(lambda *a, **k: question)
    QMessageBox.warning = staticmethod(lambda *a, **k: warning)
    QMessageBox.critical = staticmethod(lambda *a, **k: critical)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)


def restore_dialogs():
    """Put the real QMessageBox.* behaviour back so modals prompt the user
    again. Call inside a snippet if you want dialogs live before it ends."""
    for name, orig in _ORIG_DIALOGS.items():
        setattr(QMessageBox, name, staticmethod(orig))


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
    app.setWindowIcon(QIcon(os.path.join(resource_path(), 'Proteus-Alciato-logo.png')))

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
