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
import urllib.error
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


# ── Carry-over files ─────────────────────────────────────────────────────────
# When a prior step was RELOADED locally (user answered "No" to recalculate), the
# server session has no in-memory state for it. To run the next step remotely we
# upload the minimal set of that step's outputs and reload them on the server.
# These lists are the files the server's reload path reads (UpdateMask for Step 1;
# UpdateAcResults/_LoadAcResultData for Step 2). Paths are built from the
# per-trajectory prefix path (Config['OutputFilesPath'] + prefix).
#
# FIRST PASS = single-focus. For multifocus / phased arrays add the extra
# per-focus files here (e.g. the additional DataForSim/field files) — the rest of
# the machinery (upload + reload) needs no change.
def planning_carryover_files(cfg, prefix_path):
    files = [prefix_path + 'BabelViscoInput.nii.gz',
             prefix_path + 'T1W_Resampled.nii.gz']
    if cfg.get('bUseCT'):
        files += [prefix_path + 'CT.nii.gz', prefix_path + 'CT-cal.npz']
        if cfg.get('bExtractAirRegions'):
            files.append(prefix_path + 'AirRegions.nii.gz')
    return files


def acoustic_carryover_files(cfg, prefix_path):
    return [prefix_path + 'DataForSim.h5',
            prefix_path + 'Water_DataForSim.h5',
            prefix_path + 'FullElasticSolution_Sub_NORM.nii.gz',
            prefix_path + 'Water_FullElasticSolution_Sub_NORM.nii.gz']


def thermal_carryover_files(cfg, prefix_path):
    """Step-3 outputs the server needs to RELOAD a trajectory's thermal result (so
    a thermal COMBINE can run) when it was reloaded locally. The exact filename
    depends on the Isppa/DC/PRF/Duration combination, so glob the ThermalField
    results for this trajectory's prefix."""
    return sorted(glob(prefix_path + '*ThermalField*.h5'))


def merged_acoustic_carryover_files(cfg, merged_prefix_path):
    """The merged-acoustic outputs the server needs to RELOAD the combined Step-2
    result (so a thermal combine can run on it) when it was made locally."""
    return [merged_prefix_path + 'Merged_DataForSim.h5',
            merged_prefix_path + 'Water_Merged_DataForSim.h5',
            merged_prefix_path + 'Merged_NORM.nii.gz']


def _new_session(workspace_id, server_paths):
    """Per-trajectory session state tracking what the server has loaded, so we
    prime only what's missing (and can drive multi-trajectory combines)."""
    return {'workspace': workspace_id, 'server_paths': server_paths,
            'launched': False,
            'session_id': None,         # server-assigned session (per-session worker)
            'planning': False,          # planning (all trajectories) loaded
            'acoustic': set(),          # trajectory indices with acoustic loaded
            'thermal': set(),           # trajectory indices with thermal loaded
            'merged_acoustic': False}   # combined Step-2 field present on server


class RemoteNotReady(Exception):
    """Raised when the configured server can't be reached / used."""


class RunServerCalculation(QObject):
    """Prepares and drives one pipeline step on a remote server."""

    finished = Signal(object)
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self, mainApp, step, trajectory=None, combine=False):
        super().__init__()
        self._mainApp = mainApp
        self._step = step
        self._trajectory = trajectory      # active trajectory index for Steps 2/3
        self._combine = combine            # CombineTrajectories (merged) operation
        self._server = (mainApp.Config or {}).get('RemoteServer')
        self._errorText = None
        # Capture the trajectory's user parameters NOW, on the GUI thread (the
        # worker runs off-thread and must not read live widgets). Empty for Step 1
        # and for a combine (which just clicks the CombineTrajectories button).
        self._param_actions = [] if combine else self._capture_params()
        # A thermal combine scales the merge by EACH trajectory's Isppa (see
        # Babel_Thermal.CombineTrajectories -> MergedPressureRatio), so capture
        # every trajectory's Step-3 GUI inputs to sync them on the server just
        # before combining. (The acoustic combine merges from files, no GUI input.)
        self._combine_sync_actions = (self._capture_all_traj_params(STEP_THERMAL)
                                      if combine and step == STEP_THERMAL else [])
        # Thermal runs must use the profile currently selected on the client (the
        # user may have picked a new one via Babel_Thermal.SelectProfile). Capture
        # its path now; the worker uploads it and tells the server to reload it.
        self._thermal_profile = ((mainApp.Config or {}).get('ThermalProfile')
                                 if step == STEP_THERMAL else None)

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

    def _capture_all_traj_params(self, step):
        """Capture EVERY trajectory's Step-2/3 settable controls as
        (select_trajectory + set…) actions, so a combine syncs each trajectory's
        GUI inputs (e.g. IsppaSpinBox) to the server before merging. GUI thread."""
        m = self._mainApp
        try:
            sim = m.AcSim if step == STEP_ACOUSTIC else m.ThermalSim
            ntraj = len(m.Config['ID'])
        except Exception:
            return []
        actions = []
        for traj in range(ntraj):
            try:
                sets = self._scan_settable(sim._Widgets[traj])
            except Exception:
                continue
            if sets:
                actions.append({'action': 'select_trajectory', 'index': traj})
                actions += sets
        return actions

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

    def _input_fields(self, server_paths):
        """SelFiles-level input-selection fields (uploaded server paths + types)
        needed to LAUNCH the server's BabelBrain. Sent only on the first job of a
        session; later jobs reuse the live session and ignore inputs."""
        cfg = self._mainApp.Config
        fields = {
            't1w': server_paths['t1w'],
            'simbnibs': server_paths['simbnibs'],
            'simbnibs_type': cfg.get('SimbNIBSType'),
            'trajectory': server_paths['trajectory'],
            'trajectory_type': cfg.get('TrajectoryType'),
            'transducer': cfg.get('TxSystem'),
            'ct_type': cfg.get('CTType'),       # server _combo_index accepts the int
            'coreg_ct': cfg.get('CoregCT_MRI'),
        }
        if 'thermal_profile' in server_paths:
            fields['thermal_profile'] = server_paths['thermal_profile']
        if cfg.get('bUseCT'):
            fields['ct'] = server_paths['ct']
            if cfg.get('CTMapCombo') is not None:
                fields['ct_mapping'] = list(cfg['CTMapCombo'])
        return fields

    # ── session lifecycle / job submission ────────────────────────────────
    def _ensure_session(self):
        """Return the live remote session, creating a fresh one (temp workspace +
        uploaded inputs) if none exists. A session may already exist from a remote
        Step 1, or be missing because Step 1 was reloaded locally."""
        m = self._mainApp
        sess = getattr(m, '_RemoteSession', None)
        if sess:
            return sess
        ws = cf.create_workspace(mode='temp')
        print('[remote] no session — uploading inputs to prime a new one')
        server_paths = self._upload_inputs(ws['workspace_id'])
        sess = _new_session(ws['workspace_id'], server_paths)
        m._RemoteSession = sess
        return sess

    def _submit_step(self, sess, step_key, acts, recalculate):
        """Submit one step as a job and follow it. Inputs are attached only on the
        first job of a session (so the server launches its BabelBrain); later jobs
        reuse the live session."""
        spec = {
            'workspace': sess['workspace'],
            'recalculate': recalculate,
            'keep_alive': True,
            'config': self.advanced_config_payload(),
            'steps': {step_key: acts},
        }
        # Route every step of this session to the SAME server worker so Step-1
        # state is retained (the server assigns the session on the first job).
        if sess.get('session_id'):
            spec['session_id'] = sess['session_id']
        if not sess.get('launched'):
            spec.update(self._input_fields(sess['server_paths']))
        job_id = cf.submit(spec)
        if not sess.get('session_id'):
            sess['session_id'] = cf.LAST_SESSION_ID
        print('[remote] submitted %s job %s (recalc=%s, session=%s)'
              % (step_key, job_id, recalculate, sess.get('session_id')))
        result = self._follow(job_id)
        if result['state'] != 'SUCCEEDED':
            raise RemoteNotReady("Remote %s failed:\n%s"
                                 % (step_key, result.get('error') or result['state']))
        sess['launched'] = True
        return result

    def _thermal_profile_actions(self, sess):
        """Upload the client's current thermal profile YAML and return the action
        that makes the server (re)load it, so a profile the user changed in Step 3
        (Babel_Thermal.SelectProfile) is honoured. Empty if not a thermal step."""
        if not self._thermal_profile or not os.path.isfile(self._thermal_profile):
            return []
        server_path = cf.upload(sess['workspace'], self._thermal_profile,
                                os.path.basename(self._thermal_profile))
        print('[remote] using thermal profile:', os.path.basename(self._thermal_profile))
        return [{'action': 'set_thermal_profile', 'value': server_path}]

    def _upload_existing(self, ws_id, local_path):
        if os.path.isfile(local_path):
            cf.upload(ws_id, local_path, os.path.basename(local_path))
            return True
        print('[remote] WARNING carry-over file missing (skipped):', local_path)
        return False

    def _reload(self, sess, step_key, acts, files):
        """Upload carry-over files and run a step in RELOAD mode (recalculate=
        False) so the server loads that state into its live session instead of
        recomputing it."""
        for f in files:
            self._upload_existing(sess['workspace'], f)
        self._submit_step(sess, step_key, acts, recalculate=False)

    # ── ensure prior server state (prime only what's missing) ─────────────
    def _ensure_planning(self, sess):
        if sess['planning']:
            return
        m, cfg = self._mainApp, self._mainApp.Config
        files = []
        for traj in range(len(cfg['ID'])):     # planning reload chains all trajectories
            files += planning_carryover_files(cfg, m._prefix_path[traj])
        print('[remote] priming planning (reload all trajectories from carry-over)')
        self._reload(sess, 'planning', self._planning_steps(), files)
        sess['planning'] = True

    def _ensure_acoustic(self, sess, traj):
        self._ensure_planning(sess)
        if traj in sess['acoustic']:
            return
        m, cfg = self._mainApp, self._mainApp.Config
        files = acoustic_carryover_files(cfg, m._prefix_path[traj])
        print('[remote] priming acoustic trajectory %d (reload from carry-over)' % traj)
        self._reload(sess, 'acoustic',
                     [{'action': 'select_trajectory', 'index': traj},
                      {'action': 'run'}], files)
        sess['acoustic'].add(traj)

    def _ensure_merged_acoustic(self, sess):
        for traj in range(len(self._mainApp.Config['ID'])):
            self._ensure_acoustic(sess, traj)
        if sess['merged_acoustic']:
            return
        m, cfg = self._mainApp, self._mainApp.Config
        files = merged_acoustic_carryover_files(cfg, m._merged_prefix_path)
        print('[remote] priming merged acoustic (reload from carry-over)')
        self._reload(sess, 'acoustic',
                     [{'action': 'select_trajectory', 'index': 0},
                      {'action': 'click', 'control': 'CombineTrajectories'}], files)
        sess['merged_acoustic'] = True

    def _ensure_thermal(self, sess, traj):
        self._ensure_acoustic(sess, traj)
        if traj in sess['thermal']:
            return
        m, cfg = self._mainApp, self._mainApp.Config
        files = thermal_carryover_files(cfg, m._prefix_path[traj])
        print('[remote] priming thermal trajectory %d (reload from carry-over)' % traj)
        self._reload(sess, 'thermal',
                     [{'action': 'select_trajectory', 'index': traj},
                      {'action': 'run'}], files)
        sess['thermal'].add(traj)

    # ── run (Qt worker slot) ──────────────────────────────────────────────
    def run(self):
        try:
            if self._combine:
                self._run_combine()
            elif self._step == STEP_PLANNING:
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
        sess = _new_session(ws['workspace_id'], None)
        m._RemoteSession = sess
        T0 = time.time()
        try:
            print('[remote] uploading inputs…')
            sess['server_paths'] = self._upload_inputs(ws['workspace_id'])
            self.logTelemetry.emit("CTS:L3:S1: Frequency=%d PPW=%d (remote)"
                                   % (m._Frequency, m._BasePPW))
            result = self._submit_step(sess, STEP_PLANNING, self._planning_steps(),
                                       recalculate=True)
            output_files = self._download(result, out_dir)
            sess['planning'] = True
        except BaseException:
            cleanup_session(m)                 # don't leak the workspace/session
            raise

        dt = time.time() - T0
        print("*" * 40)
        print("*" * 5 + " DONE (remote) in %.1fs" % dt)
        print("*" * 40)
        self.finished.emit(output_files)

    def _run_reuse_step(self):
        """Steps 2/3: run on the persistent session. Any prior step that was
        reloaded locally (so the server lacks its state) is primed first from
        uploaded carry-over files. Then submit this step for the active
        trajectory; the GUI's finished slot reloads the downloaded results."""
        m = self._mainApp
        self.preflight()
        self._bind()
        out_dir = m.Config['OutputFilesPath']
        sess = self._ensure_session()
        traj = int(self._trajectory or 0)
        # Ensure the server has the prior-step state this trajectory needs.
        if self._step == STEP_ACOUSTIC:
            self._ensure_planning(sess)
        else:  # thermal needs this trajectory's acoustic (and planning)
            self._ensure_acoustic(sess, traj)
        print("[remote] Step %s trajectory %d (%d parameter(s))"
              % (self._step, traj, len(self._param_actions)))
        acts = (self._thermal_profile_actions(sess)          # no-op unless thermal
                + [{'action': 'select_trajectory', 'index': traj}]
                + self._param_actions
                + [{'action': 'run'}])
        result = self._submit_step(sess, self._step, acts, recalculate=True)
        self._download(result, out_dir)
        sess[self._step].add(traj)
        self.finished.emit({})

    def _run_combine(self):
        """CombineTrajectories (merged Step-2 or Step-3). Ensures every trajectory
        the merge needs is loaded on the server (priming from carry-over files any
        that were reloaded locally), then clicks CombineTrajectories and downloads
        the merged artifacts. The GUI's finished slot reloads them from disk."""
        m = self._mainApp
        self.preflight()
        self._bind()
        out_dir = m.Config['OutputFilesPath']
        sess = self._ensure_session()
        ntraj = len(m.Config['ID'])
        if self._step == STEP_ACOUSTIC:
            for traj in range(ntraj):
                self._ensure_acoustic(sess, traj)
            print('[remote] acoustic CombineTrajectories over %d trajectories' % ntraj)
            result = self._submit_step(sess, 'acoustic',
                        [{'action': 'select_trajectory', 'index': 0},
                         {'action': 'click', 'control': 'CombineTrajectories'}],
                        recalculate=True)
            sess['merged_acoustic'] = True
        else:  # thermal combine needs the merged acoustic + every trajectory's thermal
            self._ensure_merged_acoustic(sess)
            for traj in range(ntraj):
                self._ensure_thermal(sess, traj)
            # Sync each trajectory's Isppa (and other Step-3 inputs) to the server,
            # then combine — the merge is scaled per trajectory by these values.
            acts = (self._thermal_profile_actions(sess)
                    + self._combine_sync_actions
                    + [{'action': 'select_trajectory', 'index': 0},
                       {'action': 'click', 'control': 'CombineTrajectories'}])
            print('[remote] thermal CombineTrajectories over %d trajectories (%d sync action(s))'
                  % (ntraj, len(self._combine_sync_actions)))
            result = self._submit_step(sess, 'thermal', acts, recalculate=True)
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

    def _follow(self, job_id, poll=1.0, max_stalls=120):
        """Poll events + status until the job is terminal. A heavy step can hold
        the server's GIL for long stretches (e.g. a merged-thermal compute or a
        big display refresh), during which polls time out — the job is still
        running, so we keep waiting rather than failing. Only give up after
        max_stalls consecutive unreachable polls (a genuinely dead server)."""
        seen = 0
        stalls = 0
        while True:
            try:
                evs = cf._req("GET", "/jobs/%s/events?since=%d" % (job_id, seen),
                              timeout=30, retries=1)["events"]
                for ev in evs:
                    if ev.get("type") == "log":
                        # Server-side calculation output, streamed verbatim so it
                        # shows in the client's output window as a local run would.
                        print(ev["message"])
                    else:
                        print("[remote %3d%%] %-9s %s" % (ev["percent"], ev["phase"], ev["message"]))
                    if 'CTS:' in ev["message"]:
                        self.logTelemetry.emit(ev["message"])
                    seen += 1
                st = cf._req("GET", "/jobs/%s" % job_id, timeout=30, retries=1)
                stalls = 0
                if st["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    return st
            except (TimeoutError, urllib.error.URLError, OSError):
                stalls += 1
                if stalls >= max_stalls:
                    raise
                print("[remote] server busy during a heavy step; still waiting… (%d)" % stalls)
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
            sid = sess.get('session_id')
            if sid:
                cf._req("DELETE", "/sessions/%s" % sid)
        except Exception:
            pass
    mainApp._RemoteSession = None
