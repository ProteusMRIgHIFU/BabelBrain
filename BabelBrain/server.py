"""
BabelBrain job server (v0).

Exposes the BabelBrain pipeline over a small HTTP/JSON API so language-agnostic
clients (Python, C#, C++, curl, …) can submit jobs, follow progress, and collect
results — a network front-end over the same engine used by `--execute`
(see scripting.py).

Design constraints that shape this:
  * All GUI/pipeline work must happen on the Qt main thread. So each SESSION runs
    in its own worker PROCESS (a headless BabelBrain with its own Qt loop); the
    controller here holds only the HTTP surface, job records, and the GPU pool.
  * GPUs are a shared pool. A worker binds a GPU only for the duration of a 'run'
    action's compute, then returns it — so N GPUs serve up to N concurrent runs,
    and a persistent session never pins a GPU. If all are busy a run waits and
    the client is told "waiting for GPU to be available".
  * Jobs are long-running and produce files, so the API is job-oriented:
    submit -> {job_id, session_id} -> poll status / stream events -> artifacts.

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
    POST /jobs            (JobSpec)     -> {job_id, session_id}  (config mandatory)
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
    GET  /sessions                     -> [session summaries]
    GET  /sessions/{id}                -> session info (alive, idle_seconds, jobs)
    DELETE /sessions/{id}              -> end that session (close its worker)

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

Sessions and concurrency: each session is an independent worker process, so many
clients run concurrently (bounded only by the shared GPU pool at 'run' time). The
first job with `keep_alive: true` and no `session_id` creates a session; POST
/jobs returns its `session_id`, which the client passes back on later jobs to
REUSE the same worker — so it can run the pipeline piecewise (planning, inspect,
then acoustic/thermal) with Step-1 state retained. A job WITHOUT keep_alive runs
in a transient session that is closed once it finishes. Sessions end on
DELETE /sessions/{id}, on a keep_alive-less job completing, or, if configured, an
idle timeout (--serve-session-timeout / BABEL_SERVER_SESSION_TIMEOUT, seconds;
0 = never). The client-supplied `gpu`/`backend` in a spec are ignored — the
server assigns a GPU from its pool (--use-GPUs) per run.

All endpoints except /healthz require `Authorization: Bearer <token>` when a
token is configured (--serve-token or BABEL_SERVER_TOKEN). The token is compared
in constant time.

Transport security (TLS): pass --serve-certfile/--serve-keyfile (or
BABEL_SERVER_CERTFILE/BABEL_SERVER_KEYFILE) to serve over HTTPS instead of plain
HTTP. Add --serve-cafile (BABEL_SERVER_CAFILE) to REQUIRE and verify client
certificates (mutual TLS) — the strongest option for a known set of users. When
the server binds a non-loopback address it refuses to start without a token, and
warns loudly if TLS is not enabled; set BABEL_SERVER_ALLOW_INSECURE=1 to override
(e.g. when a reverse proxy terminates TLS in front). The recommended production
setup is still a TLS-terminating reverse proxy (nginx/Caddy) with BabelBrain
bound to localhost.
"""
import faulthandler
import hmac
import json
import multiprocessing
import os
import queue
import shutil
import signal
import ssl
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


# ── Diagnostics (for chasing intermittent hangs) ─────────────────────────────
_DEBUG = os.environ.get('BABEL_SERVER_DEBUG', '') not in ('', '0', 'false', 'False')


def _dbg(msg):
    """Verbose trace line, enabled with BABEL_SERVER_DEBUG=1. Prefixed with the
    pid so controller vs. per-session-worker lines are distinguishable."""
    if _DEBUG:
        print("[dbg %s pid%d] %s"
              % (time.strftime("%H:%M:%S"), os.getpid(), msg), flush=True)


# What the current process is doing right now — updated at each pipeline step so a
# watchdog / stack dump can say where a stuck worker is parked.
_ACTIVITY = {'what': 'idle', 'since': time.time(), 'job': None, 'seq': 0}


def _activity(what, job_id=None):
    _ACTIVITY['what'] = what
    _ACTIVITY['since'] = time.time()
    _ACTIVITY['seq'] += 1
    if job_id is not None:
        _ACTIVITY['job'] = job_id
    _dbg("activity: %s" % what)


def _install_stack_dumper(tag):
    """Register SIGUSR1 to dump every thread's stack (faulthandler). When a
    session hangs, find its worker pid in the log ('session X started (worker pid
    N)') and run  kill -USR1 N  — the full stack dump lands on the server console,
    pinpointing where it is stuck (e.g. wait_until, a queue put, matplotlib).
    Best-effort; SIGUSR1 does not exist on Windows."""
    try:
        faulthandler.enable()
        if hasattr(signal, 'SIGUSR1'):
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
            _log("%s: SIGUSR1 -> thread-stack dump enabled (pid %d)"
                 % (tag, os.getpid()))
    except Exception as e:
        _log("stack dumper not installed: %s" % e)


def _watchdog_loop(stop, threshold):
    """Optional (BABEL_SERVER_WATCHDOG=<seconds>): if a single activity runs longer
    than `threshold`, dump all thread stacks automatically. Off by default because
    a legitimate compute is one long activity; set it above your longest run."""
    while not stop.wait(5.0):
        a = _ACTIVITY
        if a['what'] != 'idle' and (time.time() - a['since']) > threshold:
            print("[watchdog pid%d] activity '%s' (job %s) has run %.0fs — "
                  "dumping thread stacks:"
                  % (os.getpid(), a['what'], a['job'], time.time() - a['since']),
                  flush=True)
            try:
                faulthandler.dump_traceback()
            except Exception:
                pass
            a['since'] = time.time()      # re-arm; report again after another window


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


# ── GPU pool ─────────────────────────────────────────────────────────────────
class GPUPool:
    """Shared pool of GPU devices, arbitrated by a multiprocessing.Queue.

    A device is a (name, backend) descriptor. Session workers acquire a device
    ONLY for the duration of a 'run' action's compute and return it immediately
    afterwards, so GPUs stay commodities rather than being pinned to a session.

    The queue IS the arbiter — no lock needed: acquire() pops a free device (or
    blocks until one is returned, giving the natural "waiting for GPU" behaviour),
    release() puts it back. The pool (holding the mp.Queue) is passed to each
    worker at spawn so every worker draws from the same set.
    """

    def __init__(self, devices):
        self._devices = [tuple(d) for d in devices]
        self._size = len(self._devices)
        self._q = multiprocessing.Queue()
        for d in self._devices:
            self._q.put(d)

    @property
    def size(self):
        return self._size

    @property
    def devices(self):
        return list(self._devices)

    def acquire(self, on_wait=None):
        """Return the next free (name, backend). If none is free, call on_wait()
        once (to notify the client) and then block until one is released."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            if on_wait is not None:
                on_wait()
            return self._q.get()

    def release(self, device):
        self._q.put(tuple(device))


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


# backend string -> Config['ComputingBackend'] code (canonical map; scripting.py:332)
_BACKEND_CODE = {'CUDA': 1, 'OpenCL': 2, 'Metal': 3, 'MLX': 4}


def _assign_gpu(bb, device):
    """Point the session's bb at an assigned GPU for the next compute. The
    calculate handlers read Config['ComputingDevice']/['ComputingBackend'] when
    they spawn the compute subprocess (BabelBrain.py:1804-1805), so setting them
    here — just before the run's click — is what binds this compute to `device`."""
    name, backend = device
    bb.Config['ComputingDevice'] = name
    bb.Config['ComputingBackend'] = _BACKEND_CODE.get(backend, 0)


def _run_calculation(bb, name, meta, act, emit, gpu_pool):
    """The 'run' action. Acquire a GPU from the shared pool for the duration of
    THIS compute only, then release it — GPUs are never held across actions, so a
    persistent session ties up a GPU only while actually computing. If the pool is
    momentarily empty, emit 'waiting for GPU to be available' and block until one
    frees. gpu_pool=None keeps the legacy behaviour (device comes from the spec)."""
    device = None
    if gpu_pool is not None:
        _activity("%s: acquiring GPU" % name)
        device = gpu_pool.acquire(
            on_wait=lambda: emit(name, "waiting for GPU to be available"))
        _assign_gpu(bb, device)
        emit(name, "running on GPU %s [%s]" % (device[0], device[1]))
    try:
        _log("========== %s: running calculation (%s) ==========" % (name, meta['run']))
        # If a hang shows up "after DONE ...", the activity stays on this line
        # (wait_until is still blocking for the tab to re-enable) — SIGUSR1 the
        # worker to see exactly where. The next marker proves wait_until returned.
        _activity("%s: compute issued, waiting for completion (%s)" % (name, meta['run']))
        t0 = time.time()
        _click(bb, meta['widget'](bb), meta['run'], True,
               act.get('timeout_ms', meta['timeout']))
        _activity("%s: compute completed (wait_until returned)" % name)
        _log("========== %s: calculation finished in %.1fs =========="
             % (name, time.time() - t0))
    finally:
        if device is not None:
            _dbg("%s: releasing GPU %s" % (name, device))
            gpu_pool.release(device)


def _run_step(bb, name, actions, emit, check_cancel, gpu_pool=None):
    """Execute an ordered list of client actions for one pipeline step."""
    meta = _STEP[name]
    bb.Widget.tabWidget.setCurrentIndex(meta['tab'])
    for act in actions:
        check_cancel()
        if not isinstance(act, dict) or 'action' not in act:
            raise ValueError("each %s action must be an object with an 'action' key" % name)
        a = act['action']
        _activity("%s action: %s" % (name, _describe(act)))
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
            _run_calculation(bb, name, meta, act, emit, gpu_pool)
        else:
            raise ValueError("unknown action %r in step %s" % (a, name))


def run_pipeline(job, manager, headless, existing_bb=None, gpu_pool=None):
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

    # The server owns GPU selection: ignore any client-supplied gpu/backend here
    # (a device is assigned from the pool per 'run' — see _run_calculation).
    inputs = {k: spec[k] for k in _SERVER_INPUT_KEYS
              if k not in ('gpu', 'backend') and spec.get(k) is not None}
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
    t_job = time.time()
    try:
        bb.testing_error = False
        for name in ('planning', 'acoustic', 'thermal'):
            if name in steps:
                ArtifactIO.set_step({'planning': 1, 'acoustic': 2, 'thermal': 3}[name])
                acts = steps[name] or []
                _activity("step '%s' (%d actions)" % (name, len(acts)), job.id)
                emit(name, "Step '%s': %d action(s)" % (name, len(acts)), _STEP[name]['base_pct'])
                _log("---------- step '%s': %d action(s) ----------" % (name, len(acts)))
                t_step = time.time()
                _run_step(bb, name, acts, emit, check_cancel, gpu_pool)
                # Elapsed at each step boundary: a slow step still prints this
                # (with a large number); a HUNG step never reaches it — so the
                # last "step ... done" vs "step ... : N action(s)" pins the stall.
                _log("---------- step '%s' done in %.1fs (job elapsed %.1fs) ----------"
                     % (name, time.time() - t_step, time.time() - t_job))
                emit(name, "step '%s' done in %.1fs" % (name, time.time() - t_step))
    finally:
        _sys.stdout = _saved_out
        restore_dialogs()

    _activity("collecting artifacts", job.id)
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

    def __init__(self, manager, headless=False, session_timeout=0, parent=None,
                 gpu_pool=None, always_persist=False):
        super().__init__(parent)
        self._manager = manager
        self._headless = headless
        self._busy = False
        # Persistent-session state: one live widget kept across jobs when a job
        # opts in with keep_alive. All access is on the Qt main thread except the
        # close/alive flags, which other threads only *set*/read (atomic bools).
        self._live_bb = None
        self._live_bb_last = 0.0
        self._close_requested = False
        self._session_timeout = session_timeout        # seconds; 0 = never reap
        # Multi-GPU server mode: the shared GPU pool (bound only during a 'run'),
        # and always_persist — in a per-session worker the widget is kept alive
        # across ALL the session's jobs (the controller, not per-job keep_alive,
        # decides when the session ends), so Step-1 state is retained.
        self._gpu_pool = gpu_pool
        self._always_persist = always_persist
        self._shutdown = False
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def request_shutdown(self):
        """Ask the executor to tear down its session and stop the Qt loop. Safe to
        call from another thread (only sets a flag; the work happens on _tick)."""
        self._shutdown = True

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
        # Worker shutdown (controller closed this session): tear down the live
        # widget and stop the Qt loop. Done here on the main thread — the drain
        # thread only sets the flag via request_shutdown().
        if self._shutdown:
            self._teardown_live()
            from PySide6.QtWidgets import QApplication
            appq = QApplication.instance()
            if appq is not None:
                appq.quit()
            return
        # Deferred teardown / idle reap of the persistent session (main thread,
        # so it is safe to touch Qt here — never from an HTTP thread). Handle a
        # pending close even when no session is live: a close on an already-closed
        # session still sets the flag, and it must be cleared here or it would
        # sabotage the NEXT session's keep_alive (stale close).
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
        _activity("starting job %s" % job.id, job.id)
        self._manager.mark_running(job)
        try:
            run_pipeline(job, self._manager, self._headless,
                         existing_bb=self._live_bb, gpu_pool=self._gpu_pool)
            self._manager.mark_finished(job, JobState.SUCCEEDED, exit_code=0)
        except JobCancelled:
            self._manager.mark_finished(job, JobState.CANCELLED, exit_code=2,
                                        error="Cancelled")
        except BaseException:
            self._manager.mark_finished(job, JobState.FAILED, exit_code=1,
                                        error=traceback.format_exc())
        finally:
            _activity("finalizing job %s" % job.id, job.id)
            self._finalize_bb(job)
            _activity("idle")

    def _finalize_bb(self, job):
        """After a job: keep the widget alive as the persistent session if the
        job asked for it (keep_alive), otherwise close it. A reused session with
        keep_alive=False is the client's signal to end the session -> close."""
        bb = job._bb
        job._bb = None
        if bb is None:
            return
        # In a per-session worker (always_persist) the widget is ALWAYS kept for
        # the next job; the controller ends the session out-of-band. Otherwise the
        # legacy single-session rule applies (keep only if the job opted in).
        keep = self._always_persist or (job.spec.get('keep_alive')
                                        and not self._close_requested)
        if keep:
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


# ── Per-session worker process (multi-GPU concurrency) ───────────────────────
class _WorkerManager:
    """The 'manager' handed to run_pipeline/QtJobExecutor INSIDE a session worker
    process. It implements the JobManager methods they call, but instead of
    keeping an event stream it FORWARDS every event/lifecycle change to the
    controller over out_q; jobs arrive via ingest() (fed from the controller)."""

    def __init__(self, out_q):
        self._out = out_q
        self._jobs = {}
        self._queue = queue.Queue()

    def ingest(self, job_id, spec):
        self._jobs[job_id] = Job(job_id, spec)
        self._queue.put(job_id)

    def cancel_local(self, job_id):
        job = self._jobs.get(job_id)
        if job is not None:
            job.cancel_requested = True

    def get(self, job_id):
        return self._jobs.get(job_id)

    def next_queued(self):
        while True:
            try:
                jid = self._queue.get_nowait()
            except queue.Empty:
                return None
            job = self._jobs.get(jid)
            if job is None or job.state == JobState.CANCELLED:
                continue
            return job

    def add_event(self, job, phase, message, percent=None, etype="phase"):
        if percent is not None:
            job.percent = percent
        job.phase = phase
        self._out.put({'kind': 'event', 'job_id': job.id, 'phase': phase,
                       'message': message, 'percent': job.percent, 'etype': etype})

    def mark_running(self, job):
        job.state = JobState.RUNNING
        job.started = time.time()
        self._out.put({'kind': 'running', 'job_id': job.id})

    def mark_finished(self, job, state, exit_code, error=None):
        job.state = state
        job.exit_code = exit_code
        job.error = error
        job.finished = time.time()
        _dbg("worker emitting FINAL for job %s -> %s (%d artifact(s))"
             % (job.id, state, len(job.artifacts)))
        self._out.put({'kind': 'final', 'job_id': job.id, 'state': state,
                       'exit_code': exit_code, 'error': error,
                       'artifacts': list(job.artifacts)})


def _session_worker_main(in_q, out_q, gpu_pool, headless):
    """Entry point of a session-worker subprocess: a headless BabelBrain that runs
    ONE session's jobs against a persistent widget, drawing a GPU from the shared
    pool per 'run'. Job/cancel/close messages arrive on in_q; events/finals go out
    on out_q (drained by the controller into its JobManager)."""
    os.environ['BABEL_SERVER_MODE'] = '1'
    os.environ.setdefault('BABEL_PYTEST', '1')
    if headless:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ.setdefault('BABEL_NO_PLOTS', '1')

    _install_stack_dumper("session-worker")     # kill -USR1 <pid> -> stack dump
    _dbg("session worker up (pid %d)" % os.getpid())

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    manager = _WorkerManager(out_q)
    executor = QtJobExecutor(manager, headless=headless, session_timeout=0,
                             gpu_pool=gpu_pool, always_persist=True)

    # Optional auto stack-dump if any single activity runs longer than N seconds.
    wd_threshold = float(os.environ.get('BABEL_SERVER_WATCHDOG', 0) or 0)
    wd_stop = threading.Event()
    if wd_threshold > 0:
        threading.Thread(target=_watchdog_loop, args=(wd_stop, wd_threshold),
                         name="worker-watchdog", daemon=True).start()

    stop = threading.Event()

    def _drain_in():
        # Deliver jobs and control messages to the executor. Runs off the Qt
        # thread so a cancel/close is seen even while a run occupies the event
        # loop; the executor honours cancel between steps and shuts down on _tick.
        while not stop.is_set():
            try:
                msg = in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get('kind')
            if kind == 'job':
                manager.ingest(msg['job_id'], msg['spec'])
            elif kind == 'cancel':
                manager.cancel_local(msg['job_id'])
            elif kind == 'close':
                executor.request_shutdown()
                stop.set()
                return

    threading.Thread(target=_drain_in, name="worker-in", daemon=True).start()
    app.exec()
    stop.set()
    wd_stop.set()
    executor._teardown_live()


# ── Controller-side session registry ─────────────────────────────────────────
class _Session:
    __slots__ = ('id', 'proc', 'in_q', 'out_q', 'created', 'last_used', 'jobs')

    def __init__(self, sid, proc, in_q, out_q):
        self.id = sid
        self.proc = proc
        self.in_q = in_q
        self.out_q = out_q
        self.created = time.time()
        self.last_used = time.time()
        self.jobs = set()          # job ids routed here (for info + crash cleanup)

    def to_dict(self):
        return {'session_id': self.id, 'alive': self.proc.is_alive(),
                'created': self.created,
                'idle_seconds': time.time() - self.last_used,
                'jobs': list(self.jobs)}


class SessionManager:
    """Controller-side registry of per-session worker processes. Spawns a worker
    on demand, routes jobs to it, and drains its events back into the controller's
    JobManager so /jobs, /jobs/{id}/events and /stream keep working unchanged.
    Concurrency = number of live sessions; GPU concurrency is bounded by the pool."""

    def __init__(self, controller_manager, gpu_pool, headless=False, session_timeout=0,
                 worker_target=_session_worker_main):
        self._cm = controller_manager
        self._pool = gpu_pool
        self._headless = headless
        self._timeout = session_timeout
        self._worker_target = worker_target    # injectable for tests (fake compute)
        self._sessions = {}
        self._job_session = {}     # job_id -> session_id (to route cancels)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._reap_loop, name="session-reaper",
                         daemon=True).start()

    def create(self):
        sid = uuid.uuid4().hex[:12]
        in_q, out_q = multiprocessing.Queue(), multiprocessing.Queue()
        # NOT daemon: the worker itself spawns the compute subprocess
        # (CalculateMaskProcess), and daemonic processes can't have children.
        # We tear workers down explicitly (close -> join -> terminate) in
        # _reap()/shutdown_all(), so they don't linger.
        proc = multiprocessing.Process(
            target=self._worker_target, name="babel-session-%s" % sid,
            args=(in_q, out_q, self._pool, self._headless), daemon=False)
        proc.start()
        sess = _Session(sid, proc, in_q, out_q)
        with self._lock:
            self._sessions[sid] = sess
        threading.Thread(target=self._drain, args=(sess,), name="drain-%s" % sid,
                         daemon=True).start()
        _log("session %s started (worker pid %s)" % (sid, proc.pid))
        return sid

    def get(self, sid):
        with self._lock:
            return self._sessions.get(sid)

    def list_summaries(self):
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def submit_to(self, sid, job_id, spec):
        sess = self.get(sid)
        if sess is None:
            return False
        sess.jobs.add(job_id)
        sess.last_used = time.time()
        with self._lock:
            self._job_session[job_id] = sid
        sess.in_q.put({'kind': 'job', 'job_id': job_id, 'spec': spec})
        return True

    def cancel(self, sid, job_id):
        sess = self.get(sid)
        if sess is not None:
            sess.in_q.put({'kind': 'cancel', 'job_id': job_id})

    def cancel_job(self, job_id):
        """Forward a cancel to whichever session's worker is running the job."""
        with self._lock:
            sid = self._job_session.get(job_id)
        if sid:
            self.cancel(sid, job_id)

    def close(self, sid):
        sess = self.get(sid)
        if sess is None:
            return False
        try:
            sess.in_q.put({'kind': 'close'})
        except Exception:
            pass
        return True

    def _drain(self, sess):
        """Apply a worker's out_q to the controller JobManager. Ends when the
        worker process exits (clean close or crash), then reaps the session."""
        cm = self._cm
        while True:
            try:
                msg = sess.out_q.get(timeout=0.5)
            except queue.Empty:
                if not sess.proc.is_alive():
                    break
                continue
            if not isinstance(msg, dict):
                continue
            # NEVER let one bad message kill this thread: if the drain stops, the
            # worker's events pile up undelivered and a finished job looks hung.
            try:
                job = cm.get(msg.get('job_id'))
                if job is None:
                    continue
                kind = msg.get('kind')
                if kind == 'event':
                    cm.add_event(job, msg['phase'], msg['message'],
                                 msg.get('percent'), msg.get('etype', 'phase'))
                elif kind == 'running':
                    cm.mark_running(job)
                elif kind == 'final':
                    job.artifacts = msg.get('artifacts') or []
                    cm.mark_finished(job, msg['state'], msg['exit_code'],
                                     msg.get('error'))
                    sess.jobs.discard(job.id)
                    _dbg("drain %s: FINAL job %s -> %s" % (sess.id, job.id, msg['state']))
                    if not job.spec.get('keep_alive'):
                        self.close(sess.id)      # one-shot / transient -> end session
            except Exception:
                _log("drain error (session %s, msg kind=%r) — continuing:\n%s"
                     % (sess.id, msg.get('kind'), traceback.format_exc()))
        _dbg("drain %s: worker exited, reaping" % sess.id)
        self._reap(sess)

    def _reap(self, sess):
        with self._lock:
            self._sessions.pop(sess.id, None)
        # Any job routed here that never reached a terminal state (worker crashed,
        # or the session was closed with jobs still queued) is marked failed.
        for jid in list(sess.jobs):
            job = self._cm.get(jid)
            if job is not None and job.state not in TERMINAL:
                self._cm.mark_finished(job, JobState.FAILED, 1,
                                       error="session ended before job completed")
        try:
            sess.proc.join(timeout=2)
        except Exception:
            pass
        if sess.proc.is_alive():
            sess.proc.terminate()
            sess.proc.join(timeout=2)
        _log("session %s ended" % sess.id)

    def _reap_loop(self):
        while not self._stop.wait(2.0):
            if not self._timeout:
                continue
            for s in self.list_summaries():
                if s['alive'] and s['idle_seconds'] > self._timeout:
                    _log("session %s idle > %ds -> closing"
                         % (s['session_id'], self._timeout))
                    self.close(s['session_id'])

    def shutdown_all(self):
        self._stop.set()
        with self._lock:
            sessions = list(self._sessions.values())
        for sess in sessions:
            self.close(sess.id)
        for sess in sessions:
            try:
                sess.proc.join(timeout=3)
            except Exception:
                pass
            if sess.proc.is_alive():
                sess.proc.terminate()
                sess.proc.join(timeout=2)


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
                  wsman, sessionmgr):

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
            # Constant-time compare so a caller can't learn the token byte-by-byte
            # from response timing. compare_digest short-circuits on length only.
            return hmac.compare_digest(hdr, "Bearer " + token)

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
                if path == "/sessions":
                    return self._send(200, {"sessions": sessionmgr.list_summaries()})
                parts = path.strip("/").split("/")
                if parts[0] == "sessions" and len(parts) == 2:
                    sess = sessionmgr.get(parts[1])
                    if sess is None:
                        return self._send(404, {"error": "unknown session"})
                    return self._send(200, sess.to_dict())
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
                    # Resolve the session (per-session worker). Reuse the client's
                    # session_id when given; otherwise spin up a fresh session. A
                    # job without keep_alive runs in a transient session that the
                    # controller closes once the job finishes.
                    sid = spec.get('session_id')
                    if sid:
                        if sessionmgr.get(sid) is None:
                            return self._send(404, {"error": "unknown session %r" % sid})
                    else:
                        sid = sessionmgr.create()
                    job_id = manager.submit(spec)
                    if not sessionmgr.submit_to(sid, job_id, spec):
                        return self._send(404, {"error": "unknown session %r" % sid})
                    return self._send(202, {"job_id": job_id, "session_id": sid})
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
                    sessionmgr.cancel_job(parts[1])   # forward to the worker running it
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
                if parts[0] == "sessions" and len(parts) == 2:
                    if sessionmgr.get(parts[1]) is None:
                        return self._send(404, {"error": "unknown session"})
                    sessionmgr.close(parts[1])
                    _log("client requested close of session %s" % parts[1])
                    return self._send(200, {"closing": True})
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


def discover_gpus():
    """Enumerate available GPUs as [name, backend] pairs, reusing SelFiles'
    per-platform logic (Metal on macOS; CUDA/OpenCL elsewhere). Constructing
    SelFiles already runs GetAvailableGPUs() into sf._GPUs (SelFiles.py:197), so
    we just read it back. Requires a QApplication to already exist."""
    try:
        from SelFiles.SelFiles import SelFiles
        sf = SelFiles()
        gpus = [list(g) for g in sf._GPUs]
        #we do a clean up to avoid reporting the same GPUs appearing with CUDA/Metal and OpenCL
        CUDAMetalgpus=[]
        for (name,backend) in gpus:
            if backend in ['CUDA','Metal']:
                CUDAMetalgpus.append(name)
        finalgpus=[]
        for (name,backend) in gpus:
            if (backend=='OpenCL' and name not in CUDAMetalgpus) or backend  != 'OpenCL':
                finalgpus.append((name,backend))
        sf.deleteLater()
        return finalgpus
    except Exception as e:
        _log("GPU discovery failed: %s" % e)
        return []


def print_available_gpus():
    """Back the --print-GPUs-available flag: list index/name/backend, then done."""
    gpus = discover_gpus()
    if not gpus:
        print("No GPUs detected (check drivers / BabelViscoFDTD install).")
        return
    print("Available GPUs (index: name [backend]):")
    for i, (name, backend) in enumerate(gpus):
        print("  %d: %s [%s]" % (i, name, backend))


def select_gpus(all_gpus, use_gpus_arg):
    """Resolve the --use-GPUs selection (comma-separated indices into `all_gpus`,
    or empty/None for all) into the concrete device list for the pool."""
    if use_gpus_arg is None or str(use_gpus_arg).strip() == "":
        return list(all_gpus)
    selected = []
    for tok in str(use_gpus_arg).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            idx = int(tok)
        except ValueError:
            raise ValueError("--use-GPUs expects comma-separated indices; got %r" % tok)
        if idx < 0 or idx >= len(all_gpus):
            raise ValueError("--use-GPUs index %d out of range (0..%d); "
                             "run --print-GPUs-available to list them"
                             % (idx, len(all_gpus) - 1))
        selected.append(all_gpus[idx])
    return selected


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


def _is_loopback(host):
    """True if `host` binds only the local machine (no remote reach). An empty
    host or 0.0.0.0/:: means "all interfaces" — i.e. remotely reachable."""
    return str(host).strip().lower() in ('127.0.0.1', 'localhost', '::1')


def _build_tls_context(certfile, keyfile, cafile):
    """Build the server SSLContext for HTTPS. With `cafile` also require and
    verify client certificates (mutual TLS). Returns None if no cert is given."""
    if not certfile:
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if cafile:
        ctx.load_verify_locations(cafile)
        ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


# ── Entry point (called from BabelBrain.main) ────────────────────────────────
def run_server(app, args):
    """Start the controller: HTTP server + session/GPU scheduling. Per-session
    worker processes do the actual pipeline work (each on a GPU drawn from the
    shared pool per run), so the controller itself runs no Qt loop. Blocks until
    interrupted; returns a process exit code."""
    host = getattr(args, 'serve_host', '127.0.0.1')
    port = int(getattr(args, 'serve_port', 8760))
    token = getattr(args, 'serve_token', None) or os.environ.get('BABEL_SERVER_TOKEN')
    headless = getattr(args, 'headless', False)

    # ── Transport security ───────────────────────────────────────────────────
    certfile = getattr(args, 'serve_certfile', None) or os.environ.get('BABEL_SERVER_CERTFILE')
    keyfile = getattr(args, 'serve_keyfile', None) or os.environ.get('BABEL_SERVER_KEYFILE')
    cafile = getattr(args, 'serve_cafile', None) or os.environ.get('BABEL_SERVER_CAFILE')
    allow_insecure = os.environ.get('BABEL_SERVER_ALLOW_INSECURE', '') not in ('', '0', 'false', 'False')
    try:
        tls_ctx = _build_tls_context(certfile, keyfile, cafile)
    except (ssl.SSLError, OSError, ValueError) as e:
        print("[server] cannot load TLS cert/key (%s): %s" % (certfile, e), flush=True)
        return 2
    scheme = "https" if tls_ctx is not None else "http"

    # Guard the exposed-and-open foot-gun. When bound to a remotely reachable
    # address, a missing token means anyone who can reach the port can read/write
    # files through the API — refuse unless explicitly overridden. Plain HTTP on
    # a public bind (token crosses the wire in cleartext) is warned, not blocked,
    # since a reverse proxy may terminate TLS in front (BABEL_SERVER_ALLOW_INSECURE).
    if not _is_loopback(host) and not allow_insecure:
        if not token:
            print("[server] REFUSING to start: bound to a non-loopback address (%s) "
                  "with no token. Set --serve-token/BABEL_SERVER_TOKEN, bind "
                  "127.0.0.1, or set BABEL_SERVER_ALLOW_INSECURE=1 to override."
                  % host, flush=True)
            return 2
        if tls_ctx is None:
            print("[server] WARNING: serving plain HTTP on a non-loopback address "
                  "(%s) — the bearer token and all data cross the network in "
                  "cleartext. Enable TLS (--serve-certfile/--serve-keyfile) or put "
                  "a TLS-terminating reverse proxy in front." % host, flush=True)

    # Server mode: configuration is memory-only (SaveLatestSelection is disabled)
    # and dialogs are bypassed as in scripting mode.
    os.environ['BABEL_SERVER_MODE'] = '1'
    os.environ.setdefault('BABEL_PYTEST', '1')
    app.setQuitOnLastWindowClosed(False)
    _install_stack_dumper("controller")     # kill -USR1 <controller pid> -> dump

    session_timeout = int(getattr(args, 'serve_session_timeout', 0)
                          or os.environ.get('BABEL_SERVER_SESSION_TIMEOUT', 0) or 0)
    use_gpus_arg = getattr(args, 'use_GPUs', None) or os.environ.get('BABEL_SERVER_USE_GPUS')

    manager = JobManager()
    wsman = WorkspaceManager(temp_base=getattr(args, 'serve_workspace_root', None)
                             or os.environ.get('BABEL_SERVER_WORKSPACE_ROOT'))
    transducers = _discover_transducers()

    # Build the GPU pool the sessions draw from. Discovery + config probing use
    # the controller's QApplication (already created) but not its event loop.
    all_gpus = discover_gpus()
    try:
        selected_gpus = select_gpus(all_gpus, use_gpus_arg)
    except ValueError as e:
        print("[server] %s" % e, flush=True)
        return 2
    gpu_pool = GPUPool(selected_gpus)
    if gpu_pool.size == 0:
        print("[server] WARNING: no GPUs in the pool — runs will wait indefinitely "
              "for a GPU. Check drivers or --use-GPUs / --print-GPUs-available.",
              flush=True)
    else:
        print("[server] GPU pool (%d): %s"
              % (gpu_pool.size, ", ".join("%s [%s]" % (n, b)
                                          for n, b in gpu_pool.devices)), flush=True)

    sessionmgr = SessionManager(manager, gpu_pool, headless=headless,
                                session_timeout=session_timeout)

    capabilities = {"transducers": transducers, "server_version": "v1-multigpu",
                    "gpus": [list(d) for d in gpu_pool.devices],
                    "gpu_pool_size": gpu_pool.size,
                    "max_concurrent_gpu_jobs": gpu_pool.size,
                    "tls": tls_ctx is not None,
                    "mutual_tls": bool(tls_ctx is not None and cafile),
                    "features": ["workspaces", "uploads", "artifact_download",
                                 # 'persistent_session' kept for older clients
                                 # (RunServerCalculation._REQUIRED_FEATURES); the
                                 # new per-session workers provide it.
                                 "persistent_session", "sessions", "multi_gpu"]}
    default_config = _default_config_dict(transducers)
    current_config = _current_config_dict(transducers)

    httpd = ThreadingHTTPServer(
        (host, port),
        _make_handler(manager, token, capabilities, default_config,
                      current_config, wsman, sessionmgr))
    if tls_ctx is not None:
        # Wrap the listening socket: ThreadingHTTPServer then does the TLS
        # handshake per connection in its worker threads, transparently.
        httpd.socket = tls_ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, name="babel-http",
                     daemon=True).start()
    tls_desc = ("mutual-TLS" if (tls_ctx is not None and cafile)
                else "TLS" if tls_ctx is not None else "no-TLS")
    print("[server] BabelBrain multi-GPU job server on %s://%s:%d  "
          "(auth: %s, %s, headless: %s)"
          % (scheme, host, port, "token" if token else "disabled",
             tls_desc, headless), flush=True)

    stop_event = threading.Event()
    _shutting = {'v': False}

    def _shutdown(*_a):
        if _shutting['v']:
            os._exit(130)                 # second Ctrl-C — force-quit
        _shutting['v'] = True
        print("\n[server] shutting down… (press Ctrl-C again to force-quit)", flush=True)
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except Exception:
        pass

    # No Qt loop in the controller — just wait (waking periodically so SIGINT is
    # handled promptly on every platform) until asked to stop.
    while not stop_event.wait(0.5):
        pass

    try:
        httpd.shutdown()
    except Exception:
        pass
    sessionmgr.shutdown_all()    # close every session worker
    wsman.cleanup_all()          # remove any lingering temp workspaces
    return 0
