"""
Scripting engine for BabelBrain — drive the app from a Python script, headless
or windowed, in source *or* in the frozen PyInstaller distribution.

This backs three entry points that share one implementation:

  * the frozen/binary CLI:  BabelBrain --execute script.py   (see BabelBrain.main)
  * the dev launcher:       dev_driver.py                     (imports the helpers)
  * ad-hoc automation / 3rd-party integration tests

A script is plain Python. These names are injected into its global scope
(see `make_namespace`):

    launch, launch_from_last_selection   -> build a ready BabelBrain widget
    wait_until, wait                     -> pump the Qt loop while work runs
    auto_answer_dialogs, restore_dialogs -> canned answers for modal QMessageBox
    check_no_error, fail                 -> integration-test helpers
    QMessageBox

Typical integration script::

    bb = launch(t1w="scan.nii.gz", simbnibs="m2m_sub/", trajectory="traj.txt",
                thermal_profile="profile.yaml", transducer="CTX_500",
                ct_type="CT", ct="ct.nii.gz", frequency_khz=500)
    auto_answer_dialogs(question=QMessageBox.Yes)   # recalculate if cached
    bb.Widget.CalculatePlanningMask.click()
    wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=900_000)
    check_no_error(bb)
    # ...run steps 2 & 3, then fall off the end -> exit code 0

Uncaught exceptions (including failed asserts / check_no_error / fail) end the
run with a non-zero exit code, so it slots straight into CI.
"""
import os
import sys
import traceback

from PySide6.QtCore import QElapsedTimer, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox


# ── Event-loop helpers (Qt-native replacements for pytest-qt's qtbot) ────────
# These reenter the *real* event loop via a nested QEventLoop rather than
# busy-spinning QTest.qWait. That matters for live GUI feedback: a busy loop
# never goes idle, and macOS only flushes intermediate frames (clock dialog,
# plots, progress) to the screen when the loop idles — so a scripted run would
# otherwise show no progress. A nested loop idles between timer ticks, so each
# update composites and you watch the run advance.
def wait(ms):
    """Run the event loop for a fixed time (e.g. let plots draw / GUI refresh)."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_until(cond, timeout_ms=900000, interval_ms=50, settle_ms=150):
    """Run the event loop until cond() is true. Worker-thread signals fire and
    the GUI updates live while we wait. Raises TimeoutError on timeout.

    After the condition is met we keep the loop running for `settle_ms`. A step
    finishing typically enables the tab (the condition) *and* schedules the
    result plot via matplotlib's draw_idle(), which defers the real render to a
    0-ms timer. Without this settle window the script would race into the next
    step (or app.quit()) before that redraw ever fired, so the canvases would
    never visibly update even though everything else did."""
    if not cond():
        loop = QEventLoop()
        elapsed = QElapsedTimer()
        elapsed.start()
        timed_out = {'v': False}
        timer = QTimer()
        timer.setInterval(interval_ms)

        def check():
            if cond():
                loop.quit()
            elif elapsed.elapsed() >= timeout_ms:
                timed_out['v'] = True
                loop.quit()

        timer.timeout.connect(check)
        timer.start()
        try:
            loop.exec()
        finally:
            timer.stop()
        if timed_out['v']:
            raise TimeoutError("wait_until timed out after %d ms" % timeout_ms)
    if settle_ms:
        wait(settle_ms)   # let deferred draw_idle() redraws fire and composite


# ── Modal-dialog control ─────────────────────────────────────────────────────
# Pristine QMessageBox statics, captured before any patching so the real
# (user-prompting) behaviour can be restored when a run ends.
_ORIG_DIALOGS = {name: getattr(QMessageBox, name)
                 for name in ('question', 'warning', 'critical', 'information')}


def auto_answer_dialogs(question=QMessageBox.No,
                        warning=QMessageBox.Ok,
                        critical=QMessageBox.Ok):
    """Make blocking QMessageBox.* static calls return canned answers so a
    script never stalls on a modal. Flip `question` to QMessageBox.Yes to,
    e.g., force recalculation instead of reload. Undone by restore_dialogs()."""
    QMessageBox.question = staticmethod(lambda *a, **k: question)
    QMessageBox.warning = staticmethod(lambda *a, **k: warning)
    QMessageBox.critical = staticmethod(lambda *a, **k: critical)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)


def restore_dialogs():
    """Put the real QMessageBox.* behaviour back so modals prompt the user."""
    for name, orig in _ORIG_DIALOGS.items():
        setattr(QMessageBox, name, staticmethod(orig))


# ── Integration-test helpers ─────────────────────────────────────────────────
class ScriptError(RuntimeError):
    pass


def fail(msg):
    """Abort the script with a non-zero exit code and a clear message."""
    raise ScriptError(msg)


def check_no_error(bb):
    """Fail the run if a BabelBrain worker flagged an error (bb.testing_error).
    Call after each wait_until in an integration test."""
    if getattr(bb, 'testing_error', False):
        raise ScriptError("BabelBrain reported testing_error=True")


def reset_advanced_config(bb):
    """Reset every Advanced-Options parameter in bb.Config to its default value.

    Gives a deterministic starting state (independent of any previously-saved
    session) — useful before an automated/regression run. Mirrors the dialog's
    "Reset to defaults"; call it right after launch() and before Step 1, then
    apply any specific overrides you want. Returns bb."""
    from Options.Options import ApplyAdvancedConfig
    ApplyAdvancedConfig(bb.Config, bb._AllTransducers, force=True)
    return bb


# ── Input-configuration mappings (see SelFiles combo boxes) ──────────────────
_TRAJECTORY_TYPE = {'brainsight': 0, 'slicer': 1,'localite':2}
_SIMBNIBS_TYPE = {'charm': 0, 'headreco': 1}
_CT_TYPE = {'none': 0, 'no': 0, 'ct': 1, 'real ct': 1, 'zte': 2, 'petra': 3}
_COREG = {'no': 0, 'yes': 1, False: 0, True: 1}


def _combo_index(value, mapping, what):
    if isinstance(value, bool):
        return mapping[value]
    if isinstance(value, int):
        return value
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError("Unknown %s %r (expected one of %s or an int index)"
                         % (what, value, sorted(k for k in mapping if isinstance(k, str))))
    return mapping[key]


def _apply_inputs(sf, inputs):
    """Set the given SelFiles fields. Only keys present are touched, so this
    also works to override a base (last-selection) configuration."""
    ui = sf.ui

    def has(k):
        return k in inputs and inputs[k] is not None

    if has('trajectory_type'):
        ui.TrajectoryTypecomboBox.setCurrentIndex(
            _combo_index(inputs['trajectory_type'], _TRAJECTORY_TYPE, 'trajectory_type'))
    if has('trajectory'):
        ui.TrajectorylineEdit.setText(str(inputs['trajectory']))
    if has('simbnibs_type'):
        ui.SimbNIBSTypecomboBox.setCurrentIndex(
            _combo_index(inputs['simbnibs_type'], _SIMBNIBS_TYPE, 'simbnibs_type'))
    if has('simbnibs'):
        ui.SimbNIBSlineEdit.setText(str(inputs['simbnibs']))
    if has('t1w'):
        ui.T1WlineEdit.setText(str(inputs['t1w']))
    if has('ct_type'):
        ui.CTTypecomboBox.setCurrentIndex(
            _combo_index(inputs['ct_type'], _CT_TYPE, 'ct_type'))
    if has('ct'):
        ui.CTlineEdit.setText(str(inputs['ct']))
    if has('coreg_ct'):
        ui.CoregCTcomboBox.setCurrentIndex(
            _combo_index(inputs['coreg_ct'], _COREG, 'coreg_ct'))
    if has('thermal_profile'):
        ui.ThermalProfilelineEdit.setText(str(inputs['thermal_profile']))
    if has('transducer'):
        tx = inputs['transducer']
        if isinstance(tx, int):
            ui.TransducerTypecomboBox.setCurrentIndex(tx)
        else:
            sf.SelectTxSystem(str(tx))
    # Computing engine: explicit (gpu, backend), or a substring match on the
    # combo text, else leave SelFiles' auto-selected default.
    if has('gpu') or has('backend'):
        sf.SelectComputingEngine(GPU=str(inputs.get('gpu', 'CPU')),
                                 Backend=str(inputs.get('backend', '')))
    elif has('computing'):
        from PySide6.QtCore import Qt
        idx = ui.ComputingEnginecomboBox.findText(str(inputs['computing']), Qt.MatchContains)
        if idx >= 0:
            ui.ComputingEnginecomboBox.setCurrentIndex(idx)
    if has('multipoint_type'):
        ui.MultiPointTypecomboBox.setCurrentIndex(int(inputs['multipoint_type']))
    if has('multipoint'):
        ui.MultiPointlineEdit.setText(str(inputs['multipoint']))
    if has('ct_mapping'):
        # value is the (row) name tuple used by SelFiles._dfCTParams
        try:
            ui.CTMappingcomboBox.setCurrentIndex(
                sf._dfCTParams.index.get_loc(tuple(inputs['ct_mapping'])))
        except Exception as e:
            raise ValueError("Invalid ct_mapping %r: %s" % (inputs['ct_mapping'], e))


def _validate(sf):
    """Mirror SelFiles.Continue's checks without popping its modal, raising a
    single clear error if inputs are incomplete/invalid."""
    sf.msgDetails = ""
    checks = [
        (sf.ValidTrajectory(), "trajectory"),
        (sf.ValidSimNIBS(), "SimbNIBS folder"),
        (sf.ValidThermalProfile(), "thermal profile"),
        (sf.ValidateMultiPointProfile(), "multi-point profile"),
        (os.path.isfile(sf.ui.T1WlineEdit.text()), "T1W file"),
    ]
    if sf.ui.CTTypecomboBox.currentIndex() > 0:
        checks.append((os.path.isfile(sf.ui.CTlineEdit.text()), "CT/ZTE/PETRA file"))
    bad = [what for ok, what in checks if not ok]
    if bad:
        detail = (" — " + sf.msgDetails) if getattr(sf, 'msgDetails', '') else ""
        raise ValueError("Invalid input(s): %s%s" % (", ".join(bad), detail))


def build_selfiles(inputs, base_config=None):
    """Create a SelFiles widget, optionally seed it from a saved selection
    (base_config), apply `inputs` on top, validate, and return it."""
    from SelFiles.SelFiles import SelFiles
    sf = SelFiles()
    if base_config:
        _apply_prev_config(sf, base_config)
    _apply_inputs(sf, inputs)
    _validate(sf)
    return sf


def _post_construct(bb, inputs):
    """Apply the simulation parameters that live on the main widget rather than
    on SelFiles (frequency, points-per-wavelength, HU threshold)."""
    w = bb.Widget
    if inputs.get('frequency_khz') is not None:
        freq = str(int(inputs['frequency_khz']))
        idx = w.USMaskkHzDropDown.findText(freq)
        if idx >= 0:
            w.USMaskkHzDropDown.setCurrentIndex(idx)
    if inputs.get('ppw') is not None:
        w.USPPWSpinBox.setProperty('UserData', int(inputs['ppw']))
    if inputs.get('hu_threshold') is not None and hasattr(w, 'HUThresholdSpinBox'):
        w.HUThresholdSpinBox.setValue(int(inputs['hu_threshold']))


def launch(base_config=None, **inputs):
    """Build a ready-to-drive BabelBrain widget from explicit inputs.

    Recognised keys: t1w, trajectory, trajectory_type, simbnibs, simbnibs_type,
    ct_type, ct, coreg_ct, ct_mapping, thermal_profile, transducer, gpu, backend,
    computing, multipoint_type, multipoint, frequency_khz, ppw, hu_threshold,
    output_path. `base_config` (a saved-selection dict) seeds any field not
    given in **inputs. Returns the shown BabelBrain widget."""
    sf = build_selfiles(inputs, base_config=base_config)
    bb = _babel_main().BabelBrain(sf, AltOutputFilesPath=inputs.get('output_path'))
    bb.show()
    _post_construct(bb, inputs)
    # Force the window on-screen *now*. The script runs in one event-loop
    # callback, and macOS defers a window's first composite until the loop goes
    # idle — which a synchronous script (qWait spinning) only reaches when it
    # ends. Without this the GUI would appear only at the very end (and never at
    # all when the run closes on completion). Harmless under --headless/offscreen.
    _activate_macos_app()   # foreground the app so macOS composites live updates
    bb.raise_()
    bb.activateWindow()
    wait(150)  # let the window composite/appear before the script proceeds
    return bb


def launch_from_last_selection(**overrides):
    """Like launch(), but seed every field from the last GUI selection
    (lastselection.yaml); **overrides win over the saved values."""
    cfg = _babel_main().GetLatestSelection() or {}
    return launch(base_config=cfg, **overrides)


def _apply_prev_config(sf, cfg):
    """Seed a SelFiles widget from a saved-selection dict (the same shape
    GetLatestSelection returns / BabelBrain writes). Best-effort: unknown or
    missing keys are skipped."""
    ui = sf.ui
    if cfg.get('simbnibs_path'):
        ui.SimbNIBSlineEdit.setText(cfg['simbnibs_path'])
    if cfg.get('T1W'):
        ui.T1WlineEdit.setText(cfg['T1W'])
    if cfg.get('Mat4Trajectory'):
        ui.TrajectorylineEdit.setText(cfg['Mat4Trajectory'])
    if cfg.get('ThermalProfile'):
        ui.ThermalProfilelineEdit.setText(cfg['ThermalProfile'])
    if cfg.get('CT_or_ZTE_input'):
        ui.CTlineEdit.setText(cfg['CT_or_ZTE_input'])
    if 'CTType' in cfg:
        ui.CTTypecomboBox.setCurrentIndex(cfg['CTType'])
    if 'CTMapCombo' in cfg:
        try:
            ui.CTMappingcomboBox.setCurrentIndex(
                sf._dfCTParams.index.get_loc(tuple(cfg['CTMapCombo'])))
        except Exception:
            pass
    if 'SimbNIBSType' in cfg:
        ui.SimbNIBSTypecomboBox.setCurrentIndex(0 if cfg['SimbNIBSType'] == 'charm' else 1)
    if 'TrajectoryType' in cfg:
        indexTypes=['brainsight','slicer','localite']
        ui.TrajectoryTypecomboBox.setCurrentIndex(indexTypes.index(cfg['TrajectoryType']))
    if 'CoregCT_MRI' in cfg:
        ui.CoregCTcomboBox.setCurrentIndex(cfg['CoregCT_MRI'])
    if 'ComputingBackend' in cfg and cfg['ComputingBackend'] != 0:
        backend = {1: 'CUDA', 2: 'OpenCL', 3: 'Metal', 4: 'MLX'}.get(cfg['ComputingBackend'], '')
        if backend:
            sf.SelectComputingEngine(GPU=cfg.get('ComputingDevice', ''), Backend=backend)
    if cfg.get('TxSystem'):
        sf.SelectTxSystem(cfg['TxSystem'])
    if cfg.get('EnableMultiPoint'):
        ui.MultiPointTypecomboBox.setCurrentIndex(1)
    if cfg.get('MultiPoint', '').strip() if isinstance(cfg.get('MultiPoint'), str) else False:
        ui.MultiPointlineEdit.setText(cfg['MultiPoint'])


# ── Script runner ────────────────────────────────────────────────────────────
def make_namespace(use_last_selection=False):
    """Globals injected into an executed script."""
    return {
        'launch': launch_from_last_selection if use_last_selection else launch,
        'launch_from_last_selection': launch_from_last_selection,
        'wait_until': wait_until,
        'wait': wait,
        'auto_answer_dialogs': auto_answer_dialogs,
        'restore_dialogs': restore_dialogs,
        'check_no_error': check_no_error,
        'reset_advanced_config': reset_advanced_config,
        'fail': fail,
        'QMessageBox': QMessageBox,
    }


def _babel_main():
    """Return the module that defines the BabelBrain widget / GetLatestSelection.

    Those live in the entry script BabelBrain.py, which is '__main__' at runtime
    in BOTH source and the frozen app (including a frozen 'spawn' child, as long
    as BabelBrain.py defers freeze_support() to its __main__ guard so the module
    finishes defining before the child is dispatched). We must NOT plain
    `import BabelBrain`: a 2-byte BabelBrain/__init__.py makes 'BabelBrain' a
    *package*, and in the frozen bundle that empty package shadows the entry
    module. Prefer __main__; fall back to importing the module for non-entry
    contexts (e.g. dev_driver, where __main__ is dev_driver.py)."""
    import importlib
    m = sys.modules.get('__main__')
    if m is not None and hasattr(m, 'GetLatestSelection') and hasattr(m, 'BabelBrain'):
        return m
    # Fallback for non-entry contexts. Validate it actually has the symbols — the
    # frozen empty 'BabelBrain' package would otherwise be returned and blow up
    # later with a confusing AttributeError at the call site.
    mod = importlib.import_module('BabelBrain')
    if not (hasattr(mod, 'GetLatestSelection') and hasattr(mod, 'BabelBrain')):
        raise RuntimeError(
            "Could not locate the BabelBrain entry module (GetLatestSelection / "
            "BabelBrain widget). __main__ is incomplete and 'import BabelBrain' "
            "resolved to %r, which lacks those symbols. In a frozen build this "
            "means a spawn child was dispatched before BabelBrain.py finished "
            "defining — ensure freeze_support() is called from its __main__ guard."
            % getattr(mod, '__file__', mod))
    return mod


def _has_visible_window(app):
    return any(w.isVisible() for w in app.topLevelWidgets())


def _activate_macos_app():
    """Bring the process to the foreground on macOS.

    A binary launched from a terminal with no user interaction (unlike
    dev_driver, whose modal file dialog you click through) stays a *background*
    app, and macOS then won't composite live window updates — so a scripted run
    shows a static window and no progress. `[NSApp activateIgnoringOtherApps:]`
    fixes that. Done via the Obj-C runtime through ctypes so there's no pyobjc
    dependency (works in the frozen bundle too). Best-effort / no-op elsewhere."""
    if sys.platform != 'darwin':
        return
    try:
        import ctypes
        objc = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.dylib')
        ctypes.cdll.LoadLibrary('/System/Library/Frameworks/AppKit.framework/AppKit')
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls = objc.objc_getClass(b'NSApplication')
        shared = objc.objc_msgSend(cls, objc.sel_registerName(b'sharedApplication'))
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        objc.objc_msgSend(shared, objc.sel_registerName(b'activateIgnoringOtherApps:'), True)
    except Exception:
        pass


def run_script(app, path=None, code=None, use_last_selection=False, keep_open=False):
    """Execute a script/snippet against a running Qt loop and return an exit
    code (0 = clean, non-zero = uncaught exception). The script is kicked off
    once the loop is live so wait_until can pump events.

    On completion the loop is quit and the process exits — unless keep_open is
    set AND a window is visible, in which case BabelBrain is left interactive
    (like dev_driver); Qt then exits when the user closes the window, still
    propagating the script's exit code."""
    result = {'code': 0}

    def runner():
        ns = make_namespace(use_last_selection)
        try:
            if code is not None:
                exec(compile(code, '<--code>', 'exec'), ns)
            else:
                with open(path) as f:
                    src = f.read()
                exec(compile(src, path, 'exec'), ns)
            print("[scripting] script completed with no error.")
        except SystemExit as e:
            result['code'] = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:
            traceback.print_exc()
            src_name = '<--code>' if code is not None else path
            print("[scripting] script '%s' failed (see traceback above)." % src_name,
                  file=sys.stderr)
            result['code'] = 1
        finally:
            restore_dialogs()
            if keep_open and _has_visible_window(app):
                print("[scripting] script finished — BabelBrain left open. "
                      "Close the window to exit (exit code %d)." % result['code'])
            else:
                app.quit()

    # Small delay (not 0) so app.exec() runs the native loop first: the app
    # foregrounds and the platform warms up before the (blocking) script starts,
    # matching dev_driver's delayed kickoff.
    QTimer.singleShot(300, runner)
    app.exec()
    return result['code']


def run_scripting(app, args):
    """Entry point used by BabelBrain.main() for --execute / --code. Reuses the
    established BABEL_PYTEST bypass so file-save dialogs etc. don't block an
    unattended run. Returns the process exit code."""
    os.environ.setdefault('BABEL_PYTEST', '1')
    headless = getattr(args, 'headless', False)
    keep_open = getattr(args, 'keep_open', False)
    if keep_open and headless:
        print("[scripting] --keep-open ignored under --headless (no visible window).")
        keep_open = False
    return run_script(app,
                      path=getattr(args, 'execute', None),
                      code=getattr(args, 'code', None),
                      use_last_selection=getattr(args, 'use_last_selection', False),
                      keep_open=keep_open)
