"""
Client-side offload of the BabelBrain pipeline to a remote job server.

When the user selects a remote server as the "computing engine"
(Config['ComputingBackend'] == 5, Config['RemoteServer'] set), the three local
workers (RunMaskGeneration / RunSimulation / RunThermalSim) hand their step to
this worker instead of spawning a local GPU subprocess. It:

  1. opens a (temp) workspace on the server and uploads the inputs,
  2. submits the step as a job carrying the client's advanced configuration,
  3. follows progress (printed to the log window),
  4. downloads the produced artifacts to the SAME local paths the GUI expects
     (server and client derive identical filenames from the trajectory ID +
     transducer + frequency + PPW, so a basename copy lands each file where the
     local Step-N display code looks for it), and
  5. cleans the workspace up.

RunServerCalculation mirrors RunMaskGeneration's Qt signals (finished / endError
/ logTelemetry) and its run() slot, so the existing QThread plumbing in
ExecuteTrajectory drives it unchanged.

Introduced incrementally. THIS increment wires Step 1 (planning) for the common
single-trajectory case; Steps 2/3 and multi-trajectory / persistent sessions
follow. Reuses client_functions.py (HTTP helpers) and RemoteServers.py.
"""
import os
import time
import traceback
from glob import glob

from PySide6.QtCore import QObject, Signal

import client_functions as cf
import RemoteServers

# Map the GUI step to the server's `steps` key (see server.py _STEP).
STEP_PLANNING = 'planning'
STEP_ACOUSTIC = 'acoustic'
STEP_THERMAL = 'thermal'

# Advanced-config-dict features a server must advertise for offload to work.
_REQUIRED_FEATURES = {'workspaces', 'uploads', 'artifact_download', 'persistent_session'}


class RemoteNotReady(Exception):
    """Raised when the configured server can't be reached / used."""


class RunServerCalculation(QObject):
    """Prepares and drives one pipeline step on a remote server."""

    finished = Signal(object)
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self, mainApp, step, trajectory=None):
        super().__init__()
        self._mainApp = mainApp
        self._step = step
        self._trajectory = trajectory      # active trajectory index for Steps 2/3
        self._server = (mainApp.Config or {}).get('RemoteServer')
        self._errorText = None
        # Capture the trajectory's user parameters NOW, on the GUI thread (the
        # worker runs off-thread and must not read live widgets). Empty for Step 1
        # (its controls are handled by _planning_steps).
        self._param_actions = self._capture_params()

    # ── server addressing / preflight ────────────────────────────────────
    def _bind(self):
        """Point the shared client helpers at this job's server."""
        cf.BASE = RemoteServers.base_url(self._server)
        cf.TOKEN = self._server.get('token')

    def preflight(self):
        """Verify the server is reachable and capable, or raise RemoteNotReady."""
        if not self._server:
            raise RemoteNotReady("No remote server is configured for this session.")
        ok, info = RemoteServers.test_connection(self._server)
        if not ok:
            raise RemoteNotReady("Remote server '%s' is not available:\n%s"
                                 % (self._server.get('name', '?'), info))
        caps = info.get('capabilities', {})
        missing = _REQUIRED_FEATURES - set(caps.get('features', []))
        if missing:
            raise RemoteNotReady("Server '%s' is missing required features: %s"
                                 % (self._server.get('name', '?'), ", ".join(sorted(missing))))
        return caps

    # ── advanced configuration ───────────────────────────────────────────
    def advanced_config_payload(self):
        """The advanced-configuration dict to send with the job: the server's own
        default advanced config as the key set, overlaid with the client's current
        AdvancedOptions values. Fetching the server default guarantees the key set
        matches what the server expects; the overlay carries the local user's
        choices (client-owned config)."""
        self._bind()
        default = cf._req("GET", "/defaultconfig")
        cfg = dict(self._mainApp.Config or {})
        return {k: (cfg[k] if k in cfg else dv) for k, dv in default.items()}

    # ── input staging ────────────────────────────────────────────────────
    def _upload_inputs(self, ws_id):
        """Upload the minimal input set into the workspace; return the server-side
        paths to reference in the JobSpec. Only the files Step 1 actually reads
        are sent (not the whole m2m_* folder)."""
        cfg = self._mainApp.Config
        paths = {}
        paths['t1w'] = cf.upload(ws_id, cfg['T1W'], os.path.basename(cfg['T1W']))

        # SimNIBS folder: Step 1 only needs the segmentation (+ the charm log for
        # the SimNIBS version). Staged under m2m/ so `simbnibs` points at a dir.
        m2m = cfg['simbnibs_path']
        seg = os.path.join(m2m, 'final_tissues.nii.gz')

        if not os.path.isfile(seg):
            raise RemoteNotReady("Expected segmentation not found:\n%s\n(remote offload "
                                 "currently supports SimNIBS 'charm' output.)" % seg)
        seg_server = cf.upload(ws_id, seg, 'm2m/final_tissues.nii.gz')

        if cfg.get('bSegmentBrainTissue'):
            mshfile = glob(os.path.join(m2m,'*.msh'))
            if len(mshfile)!=1:
                raise RuntimeError("There should be one (and only one) .msh file at " + m2m)
            mshfile=mshfile[0]
            cf.upload(ws_id, mshfile, 'm2m/' +os.path.split(mshfile)[1])


        paths['simbnibs'] = os.path.dirname(seg_server)
        charm_log = os.path.join(m2m, 'charm_log.html')
        if os.path.isfile(charm_log):
            cf.upload(ws_id, charm_log, 'm2m/charm_log.html')

        paths['trajectory'] = cf.upload(ws_id, cfg['Mat4Trajectory'],
                                        os.path.basename(cfg['Mat4Trajectory']))
        if cfg.get('ThermalProfile') and os.path.isfile(cfg['ThermalProfile']):
            paths['thermal_profile'] = cf.upload(ws_id, cfg['ThermalProfile'],
                                                 os.path.basename(cfg['ThermalProfile']))
        if cfg.get('bUseCT') and cfg.get('CT_or_ZTE_input'):
            ct = cfg['CT_or_ZTE_input']
            paths['ct'] = cf.upload(ws_id, ct, os.path.basename(ct))
        return paths

    def _planning_steps(self):
        """The ordered planning actions: Step-1 controls that live on the main
        widget (frequency / PPW / HU or density threshold), then run."""
        m = self._mainApp
        acts = [{'action': 'set', 'control': 'USMaskkHzDropDown',
                 'value': str(int(m._Frequency / 1e3))},
                {'action': 'set', 'control': 'USPPWSpinBox', 'value': int(m._BasePPW)}]
        if m.Config.get('bUseCT'):
            acts.append({'action': 'set', 'control': 'HUThresholdSpinBox',
                         'value': float(m.Widget.HUTreshold.value())})
            if m.Config.get('CTType') in [2, 3]:
                acts.append({'action': 'set', 'control': 'ZTERangeSlider',
                             'value': m.Widget.ZTERangeSlider.value()})
        acts.append({'action': 'run'})
        return acts

    # ── per-trajectory parameter capture (Steps 2/3) ─────────────────────
    def _capture_params(self):
        """Read every user-settable control on the active trajectory's Step-2/3
        widget as `set` actions, so the remote run reproduces the client's
        parameters 1:1. Runs on the GUI thread (from __init__). Future-proof: any
        control the server can set (see _settable_controls in server.py) is picked
        up automatically — no per-parameter maintenance."""
        if self._step not in (STEP_ACOUSTIC, STEP_THERMAL):
            return []
        m = self._mainApp
        traj = int(self._trajectory or 0)
        try:
            sim = m.AcSim if self._step == STEP_ACOUSTIC else m.ThermalSim
            widget = sim._Widgets[traj]
        except Exception:
            return []
        return self._scan_settable(widget)

    @staticmethod
    def _scan_settable(widget):
        """Mirror server.py's _settable_controls scan, but read current values.
        Combos/checkboxes are emitted before numeric values so that a combo change
        that resets a spin box can't clobber the value we set afterwards."""
        choices, values = [], []
        for name in dir(widget):
            if name.startswith('_'):
                continue
            try:
                w = getattr(widget, name)
            except Exception:
                continue
            try:
                if hasattr(w, 'setCurrentText') and hasattr(w, 'currentText'):
                    choices.append({'action': 'set', 'control': name,
                                    'value': w.currentText()})
                elif (hasattr(w, 'setChecked') and hasattr(w, 'isCheckable')
                      and w.isCheckable()):
                    choices.append({'action': 'set', 'control': name,
                                    'value': bool(w.isChecked())})
                elif hasattr(w, 'setValue') and hasattr(w, 'value'):
                    values.append({'action': 'set', 'control': name,
                                   'value': w.value()})
            except Exception:
                continue
        return choices + values

    def _build_spec(self, ws_id, server_paths):
        cfg = self._mainApp.Config
        spec = {
            'workspace': ws_id,                 # outputs collected here
            'recalculate': True,                # this path only runs when recalculating
            'keep_alive': True,                 # keep the session so Steps 2/3 reuse it
            'config': self.advanced_config_payload(),
            't1w': server_paths['t1w'],
            'simbnibs': server_paths['simbnibs'],
            'simbnibs_type': cfg.get('SimbNIBSType'),
            'trajectory': server_paths['trajectory'],
            'trajectory_type': cfg.get('TrajectoryType'),
            'transducer': cfg.get('TxSystem'),
            'ct_type': cfg.get('CTType'),       # server _combo_index accepts the int
            'coreg_ct': cfg.get('CoregCT_MRI'),
            'steps': {STEP_PLANNING: self._planning_steps()},
        }
        if 'thermal_profile' in server_paths:
            spec['thermal_profile'] = server_paths['thermal_profile']
        if cfg.get('bUseCT'):
            spec['ct'] = server_paths['ct']
            if cfg.get('CTMapCombo') is not None:
                spec['ct_mapping'] = list(cfg['CTMapCombo'])
        return spec

    # ── run (Qt worker slot) ──────────────────────────────────────────────
    def run(self):
        try:
            if self._step == STEP_PLANNING:
                self._run_planning()
            elif self._step in (STEP_ACOUSTIC, STEP_THERMAL):
                self._run_reuse_step()
            else:
                raise RemoteNotReady("Remote step %r is not wired yet." % self._step)
        except RemoteNotReady as e:
            self._fail(str(e))
        except Exception:
            self._fail(traceback.format_exc())

    def _run_planning(self):
        """Step 1: open a fresh persistent session + workspace, upload inputs, run
        planning (all trajectories in one server run), download, and KEEP the
        session/workspace so Steps 2/3 reuse them."""
        m = self._mainApp
        self.preflight()
        self._bind()
        out_dir = m.Config['OutputFilesPath']
        os.makedirs(out_dir, exist_ok=True)

        cleanup_session(m)                     # drop any previous session first

        print("*" * 40)
        print("*" * 5 + " Remote Step 1 on %s — BE PATIENT..." % RemoteServers.base_url(self._server))
        print("*" * 40)
        ws = cf.create_workspace(mode='temp')
        ws_id = ws['workspace_id']
        T0 = time.time()
        print('[remote] uploading inputs…')
        server_paths = self._upload_inputs(ws_id)
        spec = self._build_spec(ws_id, server_paths)
        self.logTelemetry.emit("CTS:L3:S1: Frequency=%d PPW=%d (remote)"
                               % (m._Frequency, m._BasePPW))
        job_id = cf.submit(spec)
        print('[remote] submitted planning job', job_id)
        result = self._follow(job_id)
        if result['state'] != 'SUCCEEDED':
            try:
                cf.delete_workspace(ws_id)
                cf._req("DELETE", "/session")
            except Exception:
                pass
            raise RemoteNotReady("Remote planning failed:\n%s"
                                 % (result.get('error') or result['state']))
        output_files = self._download(result, out_dir)
        # Persist the live session so Steps 2/3 reuse the server bb + workspace.
        m._RemoteSession = {'workspace': ws_id, 'server_paths': server_paths}

        dt = time.time() - T0
        print("*" * 40)
        print("*" * 5 + " DONE (remote) in %.1fs" % dt)
        print("*" * 40)
        self.finished.emit(output_files)

    def _run_reuse_step(self):
        """Steps 2/3: reuse the persistent session opened by Step 1. Submit the
        step for the active trajectory (select_trajectory -> run), download its
        artifacts. The GUI's finished slot reloads them from disk."""
        m = self._mainApp
        sess = getattr(m, '_RemoteSession', None)
        if not sess:
            raise RemoteNotReady("Run Step 1 (Domain) on the remote server first — "
                                 "there is no active remote session to reuse.")
        self.preflight()
        self._bind()
        out_dir = m.Config['OutputFilesPath']
        traj = int(self._trajectory or 0)
        print("[remote] Step %s trajectory %d on %s (%d parameter(s))"
              % (self._step, traj, RemoteServers.base_url(self._server),
                 len(self._param_actions)))
        acts = ([{'action': 'select_trajectory', 'index': traj}]
                + self._param_actions
                + [{'action': 'run'}])
        spec = {
            'workspace': sess['workspace'],
            'recalculate': True,
            'keep_alive': True,                 # keep the session for further steps
            'config': self.advanced_config_payload(),
            'steps': {self._step: acts},
        }
        job_id = cf.submit(spec)
        print('[remote] submitted %s job %s' % (self._step, job_id))
        result = self._follow(job_id)
        if result['state'] != 'SUCCEEDED':
            raise RemoteNotReady("Remote %s failed:\n%s"
                                 % (self._step, result.get('error') or result['state']))
        self._download(result, out_dir)
        self.finished.emit({})

    def _download(self, result, out_dir):
        """Download a job's artifacts to out_dir; return the minimal output_files
        dict Step 1 needs (harmless for Steps 2/3)."""
        os.makedirs(out_dir, exist_ok=True)
        print('[remote] downloading %d artifact(s) to %s'
              % (len(result['artifacts']), out_dir))
        downloaded = cf.download_all(result['job_id'], result['artifacts'], out_dir)
        for lp, n in downloaded:
            print('   %10d B  %s' % (n, os.path.basename(lp)))
        return self._reconstruct_output_files([lp for lp, _ in downloaded])

    def _follow(self, job_id, poll=1.0):
        seen = 0
        while True:
            for ev in cf._req("GET", "/jobs/%s/events?since=%d" % (job_id, seen))["events"]:
                print("[remote %3d%%] %-9s %s" % (ev["percent"], ev["phase"], ev["message"]))
                if 'CTS:' in ev["message"]:
                    self.logTelemetry.emit(ev["message"])
                seen += 1
            st = cf._req("GET", "/jobs/%s" % job_id)
            if st["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return st
            time.sleep(poll)

    def _reconstruct_output_files(self, local_paths):
        """The minimal output_files dict Step 1's VerifyResults needs. Only
        'pCTfname' is consulted (ZTE/PETRA pseudo-CT confirmation); everything
        else the GUI reloads from the deterministic prefix paths."""
        out = {}
        for p in local_paths:
            if p.endswith('_pCT.nii.gz'):
                out['pCTfname'] = p
        return out

    def _fail(self, msg):
        self._errorText = msg
        self._mainApp._remoteErrorText = msg
        print('[remote] ERROR:\n' + msg)
        self.logTelemetry.emit("CTS:L2: remote error")
        self.endError.emit()


def cleanup_session(mainApp):
    """Tear down the persistent remote session (server bb + temp workspace) held
    by this app, if any. Safe to call when there is no session, on any thread —
    it only does HTTP. Called before a fresh Step 1 and when the app closes."""
    sess = getattr(mainApp, '_RemoteSession', None)
    if not sess:
        return
    srv = (getattr(mainApp, 'Config', None) or {}).get('RemoteServer')
    if srv:
        cf.BASE = RemoteServers.base_url(srv)
        cf.TOKEN = srv.get('token')
        try:
            cf.delete_workspace(sess['workspace'])
        except Exception:
            pass
        try:
            cf._req("DELETE", "/session")
        except Exception:
            pass
    mainApp._RemoteSession = None
