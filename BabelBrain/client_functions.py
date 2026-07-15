"""
Shared helpers for BabelBrain job-server clients (pure stdlib, no dependencies).

Every reference client (server_client_example.py, server_client_remote.py, …)
imports from here so the HTTP plumbing lives in one place. Point it at a server
once by setting the module-level BASE / TOKEN, then use the helpers:

    import client_functions as cf
    cf.BASE = "http://gpu-box:8760"
    cf.TOKEN = "secret"                       # only if the server requires it

    config = cf._req("GET", "/defaultconfig")
    job_id = cf.submit(spec)
    result = cf.follow(job_id)

For clients that do NOT share the server's filesystem there are also
create_workspace / upload / upload_dir / download(_all) / delete_workspace.
"""
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error

# Configure these once per client before calling anything below.
BASE = "http://127.0.0.1:8760"
TOKEN = None          # set to the --serve-token value if the server requires it


def _headers(extra=None):
    """Base headers for every request, adding bearer auth when a TOKEN is set."""
    h = dict(extra or {})
    if TOKEN:
        h["Authorization"] = "Bearer " + TOKEN
    return h


def _req(method, path, body=None, timeout=60, retries=8):
    """JSON HTTP request with retry on transient timeouts. While a heavy step runs
    on the server's Qt main thread it can briefly starve the HTTP thread (GIL), so
    a poll may time out even though the job is progressing fine — just retry. Only
    idempotent GETs are retried; POST is sent once to avoid double-submitting."""
    data = json.dumps(body).encode() if body is not None else None
    headers = _headers({"Content-Type": "application/json"} if data else {})
    attempts = retries if method == "GET" else 1
    for attempt in range(attempts):
        try:
            r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError:
            raise                                   # real 4xx/5xx — don't retry
        except (TimeoutError, urllib.error.URLError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(2)                           # server busy; back off and retry


# Set by submit() to the session the server assigned/reused for the last job.
# Pass it back as spec['session_id'] to route later jobs to the SAME persistent
# worker (so Step-1 state is retained across a planning -> acoustic -> thermal
# sequence). On a multi-GPU server each session is an independent worker; a GPU
# is bound only while a 'run' actually computes.
LAST_SESSION_ID = None


# ── Jobs ─────────────────────────────────────────────────────────────────────
def submit(spec):
    """Submit a JobSpec; returns the new job_id and records the server-assigned
    session_id in the module-level LAST_SESSION_ID."""
    global LAST_SESSION_ID
    resp = _req("POST", "/jobs", spec)
    LAST_SESSION_ID = resp.get("session_id")
    return resp["job_id"]


def follow(job_id, poll=1.0, on_event=None):
    """Poll status + new events until the job reaches a terminal state. Prints
    each event (or calls on_event(ev) if provided). 'log' events carry the
    server-side calculation stdout and are printed verbatim, so a remote run reads
    like a local one; other events are printed as progress. Returns the final
    status."""
    seen = 0
    while True:
        for ev in _req("GET", "/jobs/%s/events?since=%d" % (job_id, seen))["events"]:
            if on_event:
                on_event(ev)
            elif ev.get("type") == "log":
                print(ev["message"])
            else:
                print("  [%3d%%] %-9s %s" % (ev["percent"], ev["phase"], ev["message"]))
            seen += 1
        status = _req("GET", "/jobs/%s" % job_id)
        if status["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return status
        time.sleep(poll)


# ── Workspaces / uploads / downloads (no-shared-filesystem clients) ───────────
def create_workspace(mode="temp", path=None):
    """Create a server-side staging area. mode 'temp' -> server tempdir (removed
    on delete/shutdown); 'persistent' -> a real directory (`path` required)."""
    body = {"mode": mode}
    if path:
        body["path"] = path
    return _req("POST", "/workspaces", body)


def delete_workspace(ws_id):
    """Close a workspace; a 'temp' workspace and everything in it is removed."""
    return _req("DELETE", "/workspaces/%s" % ws_id)


def upload(ws_id, local_path, name, timeout=300):
    """Stream one local file into a workspace under relative `name`; returns the
    absolute server path to reference in a JobSpec."""
    with open(local_path, "rb") as f:
        data = f.read()
    r = urllib.request.Request(
        "%s/workspaces/%s/files?name=%s" % (BASE, ws_id, urllib.parse.quote(name)),
        data=data, method="POST",
        headers=_headers({"Content-Type": "application/octet-stream"}))
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())["path"]


def upload_dir(ws_id, local_dir, prefix, timeout=300):
    """Stage an entire directory (e.g. an m2m_* SimNIBS folder) preserving
    structure under `prefix`. Returns the server path of the staged dir root."""
    server_root = None
    for root, _dirs, files in os.walk(local_dir):
        for fn in files:
            lp = os.path.join(root, fn)
            rel = os.path.relpath(lp, local_dir).replace(os.sep, "/")
            sp = upload(ws_id, lp, "%s/%s" % (prefix, rel), timeout=timeout)
            if server_root is None:                 # infer the staged dir root
                marker = "/" + prefix + "/"
                server_root = sp[: sp.rfind(marker) + len("/" + prefix)]
    return server_root


def download(job_id, index, out_path, timeout=300):
    """Download artifact #index to the local file `out_path`; returns byte count."""
    r = urllib.request.Request("%s/jobs/%s/artifacts/%d" % (BASE, job_id, index),
                               headers=_headers())
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        blob = resp.read()
    with open(out_path, "wb") as f:
        f.write(blob)
    return len(blob)


def download_all(job_id, artifacts, out_dir, timeout=300):
    """Download every artifact into out_dir (named by basename). Returns a list of
    (local_path, size) tuples."""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for i, a in enumerate(artifacts):
        local = os.path.join(out_dir, os.path.basename(a["path"]))
        out.append((local, download(job_id, i, local, timeout=timeout)))
    return out
