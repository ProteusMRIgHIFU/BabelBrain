"""
BabelBrain job server (v0).

Exposes the BabelBrain pipeline over a small HTTP/JSON API so language-agnostic
clients (Python, C#, C++, curl, …) can submit jobs, follow progress, and collect
results — a network front-end over the same engine used by `--execute`
(see scripting.py).

Design constraints that shape this:
  * All GUI/pipeline work must happen on the Qt main thread. The HTTP server
    runs on background threads and NEVER touches Qt directly; it hands work to a
    QtJobExecutor that runs on the main thread (pumped by a QTimer).
  * One BabelBrain process = one GPU pipeline. Jobs are therefore serialized:
    a single-worker queue, one job in flight at a time.
  * Jobs are long-running and produce files, so the API is job-oriented:
    submit -> job_id (immediately) -> poll status / stream events -> artifacts.

Transport is intentionally thin (stdlib http.server, no extra dependencies) so
it bundles into the frozen app as-is; the JobManager/executor core is
transport-agnostic and can be re-fronted with FastAPI or gRPC later.

Run it via the BabelBrain CLI:
    python BabelBrain/BabelBrain.py --serve                 # localhost:8760, GUI shown
    python BabelBrain/BabelBrain.py --serve --headless      # no window (server box)
    python BabelBrain/BabelBrain.py --serve --serve-port 9000 --serve-token SECRET

HTTP API:
    GET  /healthz                      -> {status, busy, queued}
    GET  /capabilities                 -> {transducers: [...], features: [...]}
    GET  /defaultconfig                -> BabelBrain's default advanced config
    GET  /currentconfig                -> the server's current advanced config
    POST /jobs            (JobSpec)     -> {job_id}   (config field mandatory)
    GET  /jobs                         -> [job summaries]
    GET  /jobs/{id}                    -> job status (+ artifacts when done)
    GET  /jobs/{id}/events?since=N     -> events since index N (poll)
    GET  /jobs/{id}/stream             -> Server-Sent Events (push)
    GET  /jobs/{id}/artifacts/{n}      -> download artifact #n (raw bytes)
    POST /jobs/{id}/cancel             -> {cancelled: bool}
    POST /workspaces      ({mode,path?})-> {workspace_id, root, mode}
    GET  /workspaces                   -> [workspace summaries]
    GET  /workspaces/{id}              -> workspace info (+ staged files)
    POST /workspaces/{id}/files?name=R -> stage an upload (raw body) -> {path,size}
    DELETE /workspaces/{id}            -> close (temp workspaces are deleted)
    GET  /session                      -> {alive, idle_seconds, timeout}
    DELETE /session                    -> end the persistent session (close bb)

A JobSpec carries: input-selection fields (see _SERVER_INPUT_KEYS), a mandatory
`config` object (advanced configuration; the client owns it), `steps` — an object
mapping any of 'planning'/'acoustic'/'thermal' to an ordered list of actions
({"action": "set"|"click"|"run"|"select_trajectory", ...}) mirroring the scripting
operations — and an optional `keep_alive` flag (see below).

Filesystem sharing is OPTIONAL. A client that does not share the server's disk
can: create a workspace (mode "temp" -> server-managed tempdir, deleted on close;
mode "persistent" -> a real directory), upload its inputs into it (each upload
returns the absolute server path to reference in the JobSpec), pass the
workspace id on the job (outputs default into the workspace), and afterwards pull
results back via /jobs/{id}/artifacts/{n}. A client that DOES share the disk can
keep sending real server paths and ignore workspaces entirely.

Persistent sessions: by default each job opens a fresh BabelBrain widget and
tears it down when the job ends. Set `keep_alive: true` on a job to KEEP that
widget alive afterwards; the next job (or jobs) then REUSES it — so a client can
run the pipeline piecewise (e.g. planning first, inspect the result, then decide
whether to re-run it or move on to acoustic) with Step-1 state retained. The
session stays up across jobs until the client sends DELETE /session (a job with
keep_alive omitted/false also reuses then closes it), or, if configured, an idle
timeout elapses (--serve-session-timeout / BABEL_SERVER_SESSION_TIMEOUT, seconds;
0 = never). The server is single-worker, so there is at most one live session.

All endpoints except /healthz require `Authorization: Bearer <token>` when a
token is configured (--serve-token or BABEL_SERVER_TOKEN).
"""
import json
import os
import queue
import shutil
import signal
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from PySide6.QtCore import QObject, QTimer


# ── Verbose server logging ───────────────────────────────────────────────────
def _log(msg):
    """Timestamped server-side log line. During a running job this goes through
    the stdout tee, so it is ALSO streamed to the client (see _StreamTee)."""
    print("[server %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


class _StreamTee:
    """Wraps the real stdout while a job runs: mirrors every write to the console
    AND turns each completed line into a 'log' event on the job, so the client
    sees the calculation output live — as if it were running locally. Thread-safe
    because calculation output is printed from worker QThreads."""

    def __init__(self, job, manager, real):
        self._job = job
        self._manager = manager
        self._real = real
        self._buf = ''
        self._lock = threading.Lock()

    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        lines = []
        with self._lock:
            self._buf += s
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                lines.append(line)
        for line in lines:
            # phase=current (don't disturb progress); type='log' -> client prints raw
            self._manager.add_event(self._job, self._job.phase, line, etype='log')

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self):
        return False


# ── Job model ────────────────────────────────────────────────────────────────
class JobState:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobCancelled(Exception):
    pass


# Input-selection keys that map onto scripting.launch() — the SelFiles-level
# choices. Step-1 controls (frequency / PPW / HU threshold) are deliberately NOT
# here: the client sets them via 'planning' actions so it controls both the
# values and their order, exactly like the GUI / scripting.
_SERVER_INPUT_KEYS = (
    't1w', 'simbnibs', 'simbnibs_type', 'trajectory', 'trajectory_type',
    'thermal_profile', 'transducer', 'ct_type', 'ct', 'coreg_ct', 'ct_mapping',
    'gpu', 'backend', 'computing', 'multipoint_type', 'multipoint', 'output_path',
)


class Job:
    __slots__ = ('id', 'spec', 'state', 'phase', 'percent', 'error',
                 'artifacts', 'events', 'cancel_requested', 'created',
                 'started', 'finished', 'exit_code', '_bb')

    def __init__(self, job_id, spec):
        self.id = job_id
        self.spec = spec
        self.state = JobState.QUEUED
        self.phase = "queued"
        self.percent = 0
        self.error = None
        self.artifacts = []
        self.events = []
        self.cancel_requested = False
        self.created = time.time()
        self.started = None
        self.finished = None
        self.exit_code = None
        self._bb = None            # live widget, not serialized

    def to_dict(self):
        return {
            'job_id': self.id,
            'state': self.state,
            'phase': self.phase,
            'percent': self.percent,
            'error': self.error,
            'artifacts': self.artifacts,
            'exit_code': self.exit_code,
            'created': self.created,
            'started': self.started,
            'finished': self.finished,
            'event_count': len(self.events),
        }


class JobManager:
    """Thread-safe registry + FIFO queue. The HTTP threads call submit/get/list/
    cancel; the Qt-main-thread executor calls mark_*/add_event."""

    def __init__(self):
        self._jobs = {}
        self._order = []
        self._queue = queue.Queue()
        self._lock = threading.Lock()

    def submit(self, spec):
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, spec)
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        self.add_event(job, "queued", "Job accepted", 0)
        steps = spec.get('steps') or {}
        _log("job %s accepted: steps=%s recalc=%s keep_alive=%s workspace=%s"
             % (job_id, ",".join(steps.keys()) or "(none)", spec.get('recalculate', True),
                bool(spec.get('keep_alive')), spec.get('workspace')))
        self._queue.put(job_id)
        return job_id

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list_summaries(self):
        with self._lock:
            return [self._jobs[j].to_dict() for j in self._order]

    def next_queued(self):
        """Pop the next job id, skipping any cancelled while queued. Returns the
        Job or None. Called only on the main thread."""
        while True:
            try:
                job_id = self._queue.get_nowait()
            except queue.Empty:
                return None
            job = self.get(job_id)
            if job is None:
                continue
            if job.state == JobState.CANCELLED:      # cancelled while queued
                continue
            return job

    def busy_and_queued(self):
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.state == JobState.RUNNING)
            queued = sum(1 for j in self._jobs.values() if j.state == JobState.QUEUED)
        return running > 0, queued

    def cancel(self, job_id):
        job = self.get(job_id)
        if job is None:
            return None
        with self._lock:
            if job.state == JobState.QUEUED:
                job.state = JobState.CANCELLED
                job.finished = time.time()
                accepted = True
            elif job.state == JobState.RUNNING:
                job.cancel_requested = True          # honored between steps
                accepted = True
            else:
                accepted = False
        if accepted:
            self.add_event(job, "cancel", "Cancellation requested", job.percent)
        return accepted

    # -- mutations from the executor (main thread) --
    def add_event(self, job, phase, message, percent=None, etype="phase"):
        with self._lock:
            if percent is not None:
                job.percent = percent
            job.phase = phase
            ev = {'seq': len(job.events), 'ts': time.time(), 'type': etype,
                  'phase': phase, 'message': message, 'percent': job.percent,
                  'state': job.state}
            job.events.append(ev)

    def mark_running(self, job):
        with self._lock:
            job.state = JobState.RUNNING
            job.started = time.time()
        _log("job %s started" % job.id)
        self.add_event(job, "running", "Job started", 1)

    def mark_finished(self, job, state, exit_code, error=None):
        with self._lock:
            job.state = state
            job.exit_code = exit_code
            job.error = error
            job.finished = time.time()
            if state == JobState.SUCCEEDED:
                job.percent = 100
        dt = (job.finished - job.started) if job.started else 0.0
        _log("job %s %s in %.1fs (%d artifact(s))"
             % (job.id, state, dt, len(job.artifacts)))
        if error:
            _log("job %s error:\n%s" % (job.id, error.rstrip()))
        self.add_event(job, state.lower(), error or "Job %s" % state.lower(), etype="final")

    def events_since(self, job, index):
        with self._lock:
            return list(job.events[index:])


# ── Workspaces (server-side input staging / output collection) ───────────────
def _safe_relpath(name):
    """Turn a client-supplied upload name into a safe path relative to a
    workspace root. Rejects absolute paths and any '..' component so an upload
    can never escape its workspace. Sub-directories are allowed (so a whole
    m2m_* folder can be staged with structure preserved)."""
    norm = str(name).replace('\\', '/').strip('/')
    parts = []
    for p in norm.split('/'):
        if p in ('', '.'):
            continue
        if p == '..':
            raise ValueError("'..' is not allowed in an upload name")
        parts.append(p)
    if not parts:
        raise ValueError("an upload 'name' is required")
    return os.path.join(*parts)


class Workspace:
    __slots__ = ('id', 'mode', 'root', 'created', 'files')

    def __init__(self, ws_id, mode, root):
        self.id = ws_id
        self.mode = mode           # 'temp' | 'persistent'
        self.root = root
        self.created = time.time()
        self.files = []            # staged relative paths

    def to_dict(self):
        return {'workspace_id': self.id, 'mode': self.mode, 'root': self.root,
                'created': self.created, 'files': list(self.files)}


class WorkspaceManager:
    """Thread-safe registry of staging areas. 'temp' workspaces live in a
    server-managed tempdir and are removed on DELETE or at server shutdown;
    'persistent' workspaces point at a real directory the server keeps."""

    def __init__(self, temp_base=None):
        self._ws = {}
        self._lock = threading.Lock()
        self._temp_base = temp_base            # None -> system temp

    def create(self, mode, path=None):
        if mode not in ('temp', 'persistent'):
            raise ValueError("workspace 'mode' must be 'temp' or 'persistent'")
        if mode == 'temp':
            root = tempfile.mkdtemp(prefix='babel-ws-', dir=self._temp_base)
        else:
            if not path:
                raise ValueError("'path' is required for a persistent workspace")
            root = os.path.abspath(os.path.expanduser(path))
            os.makedirs(root, exist_ok=True)
        ws = Workspace(uuid.uuid4().hex[:12], mode, root)
        with self._lock:
            self._ws[ws.id] = ws
        _log("workspace %s created (%s): %s" % (ws.id, mode, root))
        return ws

    def get(self, ws_id):
        with self._lock:
            return self._ws.get(ws_id)

    def list_summaries(self):
        with self._lock:
            return [w.to_dict() for w in self._ws.values()]

    def dest_for(self, ws, name):
        """Resolve+create the parent dir for an upload; return (abs_path, rel)."""
        rel = _safe_relpath(name)
        dest = os.path.join(ws.root, rel)
        os.makedirs(os.path.dirname(dest) or ws.root, exist_ok=True)
        return dest, rel

    def note_file(self, ws, rel):
        with self._lock:
            if rel not in ws.files:
                ws.files.append(rel)

    def delete(self, ws_id):
        with self._lock:
            ws = self._ws.pop(ws_id, None)
        if ws is None:
            return None
        if ws.mode == 'temp':
            shutil.rmtree(ws.root, ignore_errors=True)
            _log("workspace %s deleted (temp dir removed): %s" % (ws.id, ws.root))
        else:
            _log("workspace %s closed (persistent dir kept): %s" % (ws.id, ws.root))
        return ws

    def cleanup_all(self):
        with self._lock:
            all_ws = list(self._ws.values())
            self._ws.clear()
        for ws in all_ws:
            if ws.mode == 'temp':
                shutil.rmtree(ws.root, ignore_errors=True)


# ── Pipeline runner (reuses the scripting engine) ────────────────────────────
def _apply_advanced(config, key, value):
    """Apply one advanced-config override to bb.Config. A dict value is merged
    into an existing dict entry (e.g. HomogenousMediumValues); otherwise set."""
    if isinstance(value, dict) and isinstance(config.get(key), dict):
        config[key].update(value)
    else:
        config[key] = value


# Per-step wiring: which main tab, the canonical "run" button, the default wait
# timeout, and how to reach the step's control widget (re-fetched each action,
# because selecting a trajectory tab repoints AcSim.Widget / ThermalSim.Widget).
_STEP = {
    'planning': dict(tab=0, run='CalculatePlanningMask', timeout=3_600_000,
                     widget=lambda bb: bb.Widget, tabs=None, base_pct=10),
    'acoustic': dict(tab=1, run='CalculateAcField', timeout=3_600_000,
                     widget=lambda bb: bb.AcSim.Widget,
                     tabs=lambda bb: bb.AcSim._txTabs, base_pct=35),
    'thermal':  dict(tab=2, run='CalculateThermal', timeout=3_600_000,
                     widget=lambda bb: bb.ThermalSim.Widget,
                     tabs=lambda bb: bb.ThermalSim._txTabs, base_pct=75),
}


def _settable_controls(widget):
    """Names of the value-settable controls on a step widget (for error hints /
    discovery). Control names differ per transducer family — e.g. the CTX/ring
    'Z Steering' field is TPODistanceSpinBox, not ZSteeringSpinBox."""
    out = []
    for name in dir(widget):
        if name.startswith('_'):
            continue
        try:
            obj = getattr(widget, name)
        except Exception:
            continue
        if any(hasattr(obj, s) for s in ('setValue', 'setChecked', 'setCurrentText')):
            out.append(name)
    return sorted(set(out))


def _clickable_controls(widget):
    from PySide6.QtWidgets import QAbstractButton
    out = []
    for name in dir(widget):
        if name.startswith('_'):
            continue
        try:
            obj = getattr(widget, name)
        except Exception:
            continue
        if isinstance(obj, QAbstractButton):
            out.append(name)
    return sorted(set(out))


def _set_control(widget, control, value):
    """Set one GUI control on a step's widget, auto-detecting the widget type."""
    w = getattr(widget, control, None)
    if w is None:
        raise ValueError("no settable control %r on this step. Available: %s"
                         % (control, ", ".join(_settable_controls(widget))))
    if control == 'USPPWSpinBox':
        w.setProperty('UserData', int(value))       # PPW is read from UserData
    elif hasattr(w, 'setCurrentText'):              # combo / dropdown
        if isinstance(value, int):
            w.setCurrentIndex(value)
        else:
            idx = w.findText(str(value))
            w.setCurrentIndex(idx) if idx >= 0 else w.setCurrentText(str(value))
    elif hasattr(w, 'setChecked'):                  # check box
        w.setChecked(bool(value))
    elif hasattr(w, 'setValue'):                    # spin box / slider / scrollbar
        w.setValue(value)
    else:
        raise ValueError("don't know how to set control %r" % control)


def _click(bb, widget, control, wait, timeout):
    from scripting import wait_until, check_no_error
    btn = getattr(widget, control, None)
    if btn is None:
        raise ValueError("no clickable control %r on this step. Available: %s"
                         % (control, ", ".join(_clickable_controls(widget))))
    if not btn.isEnabled():
        raise ValueError("control %r is disabled" % control)
    btn.click()
    if wait:
        wait_until(bb.Widget.tabWidget.isEnabled, timeout_ms=timeout)
        check_no_error(bb)


def _describe(act):
    a = act.get('action')
    if a == 'set':
        return "set %s = %r" % (act.get('control'), act.get('value'))
    if a == 'click':
        return "click %s%s" % (act.get('control'),
                               "" if act.get('wait', True) else " (no wait)")
    if a == 'run':
        return "run"
    if a == 'select_trajectory':
        return "select trajectory %s" % act.get('index')
    if a == 'set_thermal_profile':
        return "set thermal profile -> %s" % os.path.basename(str(act.get('value')))
    return str(a)


def _run_step(bb, name, actions, emit, check_cancel):
    """Execute an ordered list of client actions for one pipeline step."""
    meta = _STEP[name]
    bb.Widget.tabWidget.setCurrentIndex(meta['tab'])
    for act in actions:
        check_cancel()
        if not isinstance(act, dict) or 'action' not in act:
            raise ValueError("each %s action must be an object with an 'action' key" % name)
        a = act['action']
        emit(name, "%s: %s" % (name, _describe(act)))
        if a == 'select_trajectory':
            if meta['tabs'] is None:
                raise ValueError("select_trajectory is not valid in 'planning'")
            meta['tabs'](bb).setCurrentIndex(int(act['index']))
        elif a == 'set_thermal_profile':
            # Point the server at the client's (re)selected thermal profile and
            # reload it, exactly as Babel_Thermal.SelectProfile does locally
            # (without the file dialog). Re-reads AllDC_PRF_Duration/BaseIsppa/etc.
            bb.Config['ThermalProfile'] = act['value']
            bb.ThermalSim.DefaultConfig()
        elif a == 'set':
            _set_control(meta['widget'](bb), act['control'], act['value'])
        elif a == 'click':
            _click(bb, meta['widget'](bb), act['control'],
                   act.get('wait', True), act.get('timeout_ms', meta['timeout']))
        elif a == 'run':
            _log("========== %s: running calculation (%s) ==========" % (name, meta['run']))
            _click(bb, meta['widget'](bb), meta['run'], True,
                   act.get('timeout_ms', meta['timeout']))
            _log("========== %s: calculation finished ==========" % name)
        else:
            raise ValueError("unknown action %r in step %s" % (a, name))


def run_pipeline(job, manager, headless, existing_bb=None):
    """Execute one job on the Qt main thread: open BabelBrain with the client's
    inputs (or reuse a persistent session's widget), apply the client-supplied
    advanced config, then run each requested step's ordered action list. Emits
    progress; raises on error/cancellation.

    When `existing_bb` is given the launch is skipped and that live widget is
    reused — this is how a client runs the pipeline piecewise across several jobs
    (e.g. planning first, inspect, then acoustic) while Step-1 state is retained."""
    from PySide6.QtWidgets import QMessageBox
    from scripting import (launch, launch_from_last_selection,
                           auto_answer_dialogs, restore_dialogs, reset_advanced_config)

    spec = job.spec

    def emit(phase, message, percent=None):
        manager.add_event(job, phase, message, percent)

    def check_cancel():
        if job.cancel_requested:
            raise JobCancelled()

    inputs = {k: spec[k] for k in _SERVER_INPUT_KEYS if spec.get(k) is not None}
    steps = spec.get('steps') or {}
    recalc = spec.get('recalculate', True)

    _log("job %s: executing step(s) %s%s" % (
        job.id, ",".join(k for k in ('planning', 'acoustic', 'thermal') if k in steps),
        "" if existing_bb is None else " (reusing live session)"))

    if existing_bb is not None:
        emit("launch", "Reusing the persistent BabelBrain session", 5)
        _log("job %s: reusing the persistent BabelBrain session" % job.id)
        bb = existing_bb
        job._bb = bb
    else:
        emit("launch", "Opening BabelBrain with the requested inputs", 5)
        _log("job %s: launching BabelBrain (transducer=%s, output=%s)"
             % (job.id, inputs.get('transducer'), inputs.get('output_path')))
        if spec.get('use_last_selection'):
            bb = launch_from_last_selection(**inputs)
        else:
            bb = launch(**inputs)
        job._bb = bb

    # Record artifacts at save time — across the step subprocesses — into a
    # per-job temporary ledger (env var inherited by children; deleted at end).
    import ArtifactIO
    artlog = ArtifactIO.begin_run()

    # The client owns the advanced configuration: start from a deterministic
    # default baseline, then apply the mandatory client-supplied config dict.
    reset_advanced_config(bb)
    for k, v in spec['config'].items():
        _apply_advanced(bb.Config, k, v)
    _log("job %s: applied %d advanced-config override(s); recalculate=%s"
         % (job.id, len(spec['config']), recalc))

    # Stream the calculation's stdout to the client as 'log' events, so the user
    # sees the same progress/error prints they would locally. The step's heavy
    # work runs on a worker QThread, but sys.stdout is process-global, so the tee
    # (installed here on the main thread) captures those prints too.
    import sys as _sys
    _saved_out = _sys.stdout
    _sys.stdout = _StreamTee(job, manager, _saved_out)
    auto_answer_dialogs(question=QMessageBox.Yes if recalc else QMessageBox.No)
    try:
        bb.testing_error = False
        for name in ('planning', 'acoustic', 'thermal'):
            if name in steps:
                ArtifactIO.set_step({'planning': 1, 'acoustic': 2, 'thermal': 3}[name])
                acts = steps[name] or []
                emit(name, "Step '%s': %d action(s)" % (name, len(acts)), _STEP[name]['base_pct'])
                _log("---------- step '%s': %d action(s) ----------" % (name, len(acts)))
                _run_step(bb, name, acts, emit, check_cancel)
    finally:
        _sys.stdout = _saved_out
        restore_dialogs()

    # Ground-truth artifacts from the record-at-save sidecar. During the Phase-2
    # transition we cross-check the count against the Phase-1 predicted manifest.
    # Artifacts are the recorded ledger (record-at-save, cross-process),
    # filtered to files still present. Complete for fresh and cached/reload runs.
    try:
        recorded = ArtifactIO.read_manifest(artlog, existing_only=True)
        job.artifacts = [{'path': e['path'], 'fmt': e.get('fmt'),
                          'step': e.get('step'), 'role': e.get('role', 'output')}
                         for e in recorded]
    except Exception as e:
        job.artifacts = []
        emit("collect", "artifact recording failed: %s" % e)
    ArtifactIO.end_run()
    _log("job %s: collected %d artifact(s):" % (job.id, len(job.artifacts)))
    for a in job.artifacts:
        _log("    %s" % os.path.basename(a['path']))
    emit("collect", "Collected %d artifact(s)" % len(job.artifacts), 98)


# ── Qt-main-thread executor ──────────────────────────────────────────────────
class QtJobExecutor(QObject):
    """Runs on the Qt main thread. A QTimer polls the queue; one job runs at a
    time. The _busy guard is essential: wait_until reenters the event loop via a
    nested QEventLoop, which can re-fire this timer — the guard makes those
    re-entrant ticks no-ops instead of starting a second job."""

    def __init__(self, manager, headless=False, session_timeout=0, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._headless = headless
        self._busy = False
        # Persistent-session state: one live widget kept across jobs when a job
        # opts in with keep_alive. All access is on the Qt main thread except the
        # close/alive flags, which HTTP threads only *set*/read (atomic bools).
        self._live_bb = None
        self._live_bb_last = 0.0
        self._close_requested = False
        self._session_timeout = session_timeout        # seconds; 0 = never reap
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -- session control (called from HTTP threads) --
    def request_close_session(self):
        """Ask for the persistent session to be torn down. The actual close runs
        on the next main-thread tick. Returns whether a session was alive."""
        alive = self._live_bb is not None
        self._close_requested = True
        return alive

    def session_info(self):
        alive = self._live_bb is not None
        return {'alive': alive,
                'idle_seconds': (time.time() - self._live_bb_last) if alive else None,
                'timeout': self._session_timeout or None}

    def _idle_expired(self):
        return (self._session_timeout and self._live_bb is not None
                and (time.time() - self._live_bb_last) > self._session_timeout)

    def _tick(self):
        if self._busy:
            return
        # Deferred teardown / idle reap of the persistent session (main thread,
        # so it is safe to touch Qt here — never from an HTTP thread). Handle a
        # pending close even when no session is live: DELETE /session on an
        # already-closed session still sets the flag, and it must be cleared here
        # or it would sabotage the NEXT session's keep_alive (stale close).
        if self._close_requested or self._idle_expired():
            self._teardown_live()
        job = self._manager.next_queued()
        if job is None:
            return
        self._busy = True
        try:
            self._run(job)
        finally:
            self._busy = False

    def _run(self, job):
        self._manager.mark_running(job)
        try:
            run_pipeline(job, self._manager, self._headless,
                         existing_bb=self._live_bb)
            self._manager.mark_finished(job, JobState.SUCCEEDED, exit_code=0)
        except JobCancelled:
            self._manager.mark_finished(job, JobState.CANCELLED, exit_code=2,
                                        error="Cancelled")
        except BaseException:
            self._manager.mark_finished(job, JobState.FAILED, exit_code=1,
                                        error=traceback.format_exc())
        finally:
            self._finalize_bb(job)

    def _finalize_bb(self, job):
        """After a job: keep the widget alive as the persistent session if the
        job asked for it (keep_alive), otherwise close it. A reused session with
        keep_alive=False is the client's signal to end the session -> close."""
        bb = job._bb
        job._bb = None
        if bb is None:
            return
        if job.spec.get('keep_alive') and not self._close_requested:
            self._live_bb = bb
            self._live_bb_last = time.time()
            _log("session kept alive after job %s (ready to reuse)" % job.id)
        else:
            if bb is self._live_bb:
                self._live_bb = None
            self._close_requested = False
            _log("closing BabelBrain session after job %s" % job.id)
            self._destroy(bb)

    def _teardown_live(self):
        bb = self._live_bb
        self._live_bb = None
        self._close_requested = False
        if bb is not None:
            _log("tearing down persistent session")
        self._destroy(bb)

    def _destroy(self, bb):
        if bb is not None:
            try:
                bb.close()
                bb.deleteLater()
            except Exception:
                pass


# ── HTTP surface (stdlib) ────────────────────────────────────────────────────
def _validate_spec(spec):
    """Reject a malformed JobSpec at submit time. The advanced-configuration
    dict is MANDATORY: every job must carry its own config (fetch a baseline
    from GET /defaultconfig or /currentconfig, edit, and send it back)."""
    if not isinstance(spec, dict):
        return "request body must be a JSON object"
    if not isinstance(spec.get('config'), dict):
        return ("'config' (advanced-configuration object) is required on every "
                "job; get a baseline from GET /defaultconfig or /currentconfig")
    if 'steps' in spec and not isinstance(spec['steps'], dict):
        return ("'steps' must be an object mapping any of 'planning'/'acoustic'/"
                "'thermal' to an ordered list of actions")
    if 'keep_alive' in spec and not isinstance(spec['keep_alive'], bool):
        return "'keep_alive' must be a boolean"
    return None


def _make_handler(manager, token, capabilities, default_config, current_config,
                  wsman, executor):

    class Handler(BaseHTTPRequestHandler):
        server_version = "BabelBrain/0"

        def log_message(self, *a):          # quieter default logging
            pass

        # -- helpers --
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream_body_to(self, dest):
            """Stream the raw request body to a file in chunks (uploads can be
            large NIfTI/CT volumes; don't buffer them whole in memory)."""
            remaining = int(self.headers.get("Content-Length", 0))
            written = 0
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
                    written += len(chunk)
            return written

        def _send_file(self, path):
            """Stream a file back to the client as raw bytes (artifact download)."""
            if not os.path.isfile(path):
                return self._send(404, {"error": "artifact file no longer exists"})
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition",
                             'attachment; filename="%s"' % os.path.basename(path))
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1 << 16)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def _authorized(self):
            if not token:
                return True
            hdr = self.headers.get("Authorization", "")
            return hdr == "Bearer " + token

        def _guard(self):
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return False
            return True

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode() or "{}")

        # -- routing --
        def do_GET(self):
            url = urlparse(self.path)
            path = url.path.rstrip("/") or "/"
            try:
                if path == "/healthz":
                    busy, queued = manager.busy_and_queued()
                    return self._send(200, {"status": "ok", "busy": busy, "queued": queued})
                if not self._guard():
                    return
                if path == "/capabilities":
                    return self._send(200, capabilities)
                if path == "/defaultconfig":
                    return self._send(200, default_config)
                if path == "/currentconfig":
                    return self._send(200, current_config)
                if path == "/jobs":
                    return self._send(200, {"jobs": manager.list_summaries()})
                if path == "/workspaces":
                    return self._send(200, {"workspaces": wsman.list_summaries()})
                if path == "/session":
                    return self._send(200, executor.session_info())
                parts = path.strip("/").split("/")
                if parts[0] == "workspaces" and len(parts) == 2:
                    ws = wsman.get(parts[1])
                    if ws is None:
                        return self._send(404, {"error": "unknown workspace"})
                    return self._send(200, ws.to_dict())
                if parts[0] == "jobs" and len(parts) >= 2:
                    job = manager.get(parts[1])
                    if job is None:
                        return self._send(404, {"error": "unknown job"})
                    if len(parts) == 2:
                        return self._send(200, job.to_dict())
                    if len(parts) == 3 and parts[2] == "events":
                        since = int(parse_qs(url.query).get("since", ["0"])[0])
                        return self._send(200, {"events": manager.events_since(job, since)})
                    if len(parts) == 3 and parts[2] == "stream":
                        return self._stream(job)
                    if len(parts) == 4 and parts[2] == "artifacts":
                        try:
                            idx = int(parts[3])
                            entry = job.artifacts[idx]
                        except (ValueError, IndexError):
                            return self._send(404, {"error": "unknown artifact"})
                        _log("serving artifact #%d to client: %s"
                             % (idx, os.path.basename(entry['path'])))
                        return self._send_file(entry['path'])
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        def do_POST(self):
            url = urlparse(self.path)
            path = url.path.rstrip("/") or "/"
            try:
                if not self._guard():
                    return
                if path == "/jobs":
                    spec = self._body()
                    err = _validate_spec(spec)
                    if err:
                        return self._send(400, {"error": err})
                    # If the job names a workspace, default its output into that
                    # workspace so results are collected where the client can
                    # reach them (and, for a temp workspace, cleaned up on close).
                    ws_id = spec.get('workspace')
                    if ws_id:
                        ws = wsman.get(ws_id)
                        if ws is None:
                            return self._send(400, {"error": "unknown workspace %r" % ws_id})
                        spec.setdefault('output_path', ws.root)
                    job_id = manager.submit(spec)
                    return self._send(202, {"job_id": job_id})
                if path == "/workspaces":
                    body = self._body()
                    try:
                        ws = wsman.create(body.get('mode', 'temp'), body.get('path'))
                    except ValueError as e:
                        return self._send(400, {"error": str(e)})
                    return self._send(201, ws.to_dict())
                parts = path.strip("/").split("/")
                if parts[0] == "jobs" and len(parts) == 3 and parts[2] == "cancel":
                    accepted = manager.cancel(parts[1])
                    if accepted is None:
                        return self._send(404, {"error": "unknown job"})
                    return self._send(200, {"cancelled": bool(accepted)})
                if (parts[0] == "workspaces" and len(parts) == 3
                        and parts[2] == "files"):
                    ws = wsman.get(parts[1])
                    if ws is None:
                        return self._send(404, {"error": "unknown workspace"})
                    name = parse_qs(url.query).get("name", [None])[0]
                    try:
                        dest, rel = wsman.dest_for(ws, name)
                    except (ValueError, TypeError):
                        return self._send(400, {"error": "a valid ?name= is required"})
                    size = self._stream_body_to(dest)
                    wsman.note_file(ws, rel)
                    _log("received upload: %s (%d bytes) into workspace %s"
                         % (rel, size, ws.id))
                    return self._send(201, {"path": dest, "relpath": rel, "size": size})
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        def do_DELETE(self):
            url = urlparse(self.path)
            path = url.path.rstrip("/") or "/"
            try:
                if not self._guard():
                    return
                parts = path.strip("/").split("/")
                if path == "/session":
                    alive = executor.request_close_session()
                    _log("client requested session close (was alive: %s)" % bool(alive))
                    return self._send(200, {"closing": bool(alive)})
                if parts[0] == "workspaces" and len(parts) == 2:
                    ws = wsman.delete(parts[1])
                    if ws is None:
                        return self._send(404, {"error": "unknown workspace"})
                    return self._send(200, {"deleted": True, "mode": ws.mode})
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        def _stream(self, job):
            """Server-Sent Events: push each new event, then a terminal event.
            Runs in this request's own thread (ThreadingHTTPServer)."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            sent = 0
            try:
                while True:
                    evs = manager.events_since(job, sent)
                    for ev in evs:
                        self.wfile.write(("data: %s\n\n" % json.dumps(ev)).encode())
                        self.wfile.flush()
                    sent += len(evs)
                    if job.state in TERMINAL and sent >= len(job.events):
                        break
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def _discover_transducers():
    try:
        from SelFiles.SelFiles import SelFiles
        sf = SelFiles()
        txs = sf.GetAllTransducers()
        sf.deleteLater()
        return list(txs)
    except Exception:
        return []


def _default_config_dict(transducers):
    """BabelBrain's pristine advanced-config defaults (for GET /defaultconfig)."""
    from Options.Options import DefaultAdvancedConfig
    return DefaultAdvancedConfig(transducers)


def _current_config_dict(transducers):
    """The server's current advanced config (for GET /currentconfig): the saved
    lastselection.yaml values (read-only) over the defaults. Memory-only — jobs
    never mutate it, and in server mode it is never written back."""
    defaults = _default_config_dict(transducers)
    try:
        from scripting import _babel_main
        saved = _babel_main().GetLatestSelection() or {}
    except Exception:
        saved = {}
    return {k: (saved[k] if k in saved else dv) for k, dv in defaults.items()}


# ── Entry point (called from BabelBrain.main) ────────────────────────────────
def run_server(app, args):
    """Start the HTTP server (background threads) and run the Qt loop (main
    thread) until interrupted. Returns a process exit code."""
    host = getattr(args, 'serve_host', '127.0.0.1')
    port = int(getattr(args, 'serve_port', 8760))
    token = getattr(args, 'serve_token', None) or os.environ.get('BABEL_SERVER_TOKEN')
    headless = getattr(args, 'headless', False)

    # Server mode: configuration is memory-only (SaveLatestSelection is disabled)
    # and dialogs are bypassed as in scripting mode.
    os.environ['BABEL_SERVER_MODE'] = '1'
    os.environ.setdefault('BABEL_PYTEST', '1')

    # A server must NOT exit when a job's window (or a transient dialog such as
    # the progress clock) closes — otherwise the whole process quits between/after
    # steps. Only our SIGINT handler ends the loop.
    app.setQuitOnLastWindowClosed(False)

    session_timeout = int(getattr(args, 'serve_session_timeout', 0)
                          or os.environ.get('BABEL_SERVER_SESSION_TIMEOUT', 0) or 0)

    manager = JobManager()
    wsman = WorkspaceManager(temp_base=getattr(args, 'serve_workspace_root', None)
                             or os.environ.get('BABEL_SERVER_WORKSPACE_ROOT'))
    executor = QtJobExecutor(manager, headless=headless,
                             session_timeout=session_timeout)
    transducers = _discover_transducers()
    capabilities = {"transducers": transducers,
                    "server_version": "v0", "single_worker": True,
                    "features": ["workspaces", "uploads", "artifact_download",
                                 "persistent_session"]}
    default_config = _default_config_dict(transducers)
    current_config = _current_config_dict(transducers)

    httpd = ThreadingHTTPServer(
        (host, port),
        _make_handler(manager, token, capabilities, default_config,
                      current_config, wsman, executor))
    server_thread = threading.Thread(target=httpd.serve_forever, name="babel-http",
                                     daemon=True)
    server_thread.start()
    print("[server] BabelBrain job server on http://%s:%d  (auth: %s, headless: %s)"
          % (host, port, "token" if token else "disabled", headless))

    _shutting = {'v': False}

    def _shutdown(*_a):
        if _shutting['v']:
            # Second Ctrl-C — or a job is mid-step and the main thread can't
            # yield — so force-exit immediately.
            os._exit(130)
        _shutting['v'] = True
        print("\n[server] shutting down… (press Ctrl-C again to force-quit)", flush=True)
        # Stop the HTTP server off the signal handler so it returns fast.
        threading.Thread(target=httpd.shutdown, daemon=True).start()
        app.quit()

    # Keep the interpreter ticking so the Python SIGINT handler runs promptly
    # even while Qt's C++ event loop is otherwise idle.
    wake = QTimer()                       # noqa: F841 (kept alive for the loop's lifetime)
    wake.timeout.connect(lambda: None)
    wake.start(100)

    try:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except Exception:
        pass

    app.exec()
    try:
        httpd.shutdown()
    except Exception:
        pass
    executor._teardown_live()    # close a lingering persistent session (main thread)
    wsman.cleanup_all()          # remove any lingering temp workspaces
    return 0
