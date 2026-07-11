"""
Artifact recording — log every file the pipeline writes, at the moment it's
written, so the artifact list is *observed* rather than reverse-engineered.

Why a file sidecar (not an in-memory list): the heavy steps run in separate
processes (Step 1 CalculateMaskProcess, Step 2 CalculateFieldProcess, Step 3
CalculateThermalProcess), so a registry in the main/server process would never
see their saves. Instead every save appends one JSON line to a per-job temporary
ledger named by the BABEL_ARTIFACT_LOG env var. That env var is inherited by
spawned children, so main process, worker threads and subprocesses all append to
the same file. The server reads it at the end of the job, then deletes it.

The ledger is ephemeral and per-job (a tempfile): it captures only what that job
writes, so the artifact list matches which steps actually ran and never reports
stale files from an earlier config. Nothing accumulates in the output directory.
A job that only reloads cached results (saves nothing) therefore reports no
freshly-produced artifacts — which is the intended per-job semantics.

Recording is a no-op when BABEL_ARTIFACT_LOG is unset (a normal GUI session), so
this is invisible outside server/scripted runs.

Dependencies are stdlib-only on purpose, so this imports cheaply anywhere,
including inside a freshly-spawned subprocess.
"""
import json
import os
import tempfile
import time

_ENV_LOG = 'BABEL_ARTIFACT_LOG'
_ENV_STEP = 'BABEL_STEP'

_EXT_FMT = {
    '.nii.gz': 'nifti', '.nii': 'nifti', '.h5': 'h5', '.hdf5': 'h5',
    '.npy': 'npy', '.npz': 'npz', '.mat': 'mat', '.csv': 'csv',
    '.stl': 'stl', '.yml': 'yaml', '.yaml': 'yaml', '.txt': 'text',
}


def _fmt_from_path(path):
    pl = str(path).lower()
    for ext, fmt in _EXT_FMT.items():
        if pl.endswith(ext):
            return fmt
    return 'other'


def set_step(step):
    """Tag subsequent records with a pipeline step (1/2/3). Set via env so the
    value is inherited by subprocesses spawned afterwards. Call None to clear."""
    if step is None:
        os.environ.pop(_ENV_STEP, None)
    else:
        os.environ[_ENV_STEP] = str(step)


def _current_step():
    v = os.environ.get(_ENV_STEP)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return v


def record(path, step=None, role='output', fmt=None):
    """Log that `path` was written. Appends to the run's sidecar (no-op if
    BABEL_ARTIFACT_LOG is unset). Returns `path`, so it can wrap a save call."""
    log = os.environ.get(_ENV_LOG)
    if not log or not path:
        return path
    try:
        entry = {
            'path': os.path.abspath(str(path)),
            'fmt': fmt or _fmt_from_path(path),
            'step': step if step is not None else _current_step(),
            'role': role,
            'ts': time.time(),
        }
        with open(log, 'a') as f:                       # O_APPEND: safe across processes
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass                                            # never let logging break a save
    return path


# ── typed save helpers (for the direct call-site sweep) ──────────────────────
def save_nifti(img, path, **kw):
    img.to_filename(str(path))
    return record(path, **kw)


def save_h5(data, path, **kw):
    from BabelViscoFDTD.H5pySimple import SaveToH5py
    SaveToH5py(data, str(path))
    return record(path, **kw)


def save_sitk(img, path, **kw):
    import SimpleITK as sitk
    sitk.WriteImage(img, str(path))
    return record(path, **kw)


def save_mat(path, mdict, **kw):
    from scipy.io import savemat
    savemat(str(path), mdict)
    return record(path, **kw)


def save_npy(path, arr, **kw):
    import numpy as np
    np.save(str(path), arr)
    return record(path, **kw)


# ── run lifecycle + reading ──────────────────────────────────────────────────
def begin_run():
    """Create a fresh, per-job temporary ledger and point BABEL_ARTIFACT_LOG at
    it (the path is inherited by spawned children). Ephemeral by design: the
    artifact list reflects only what THIS job writes — so it naturally tracks
    which steps ran and never reports stale outputs from earlier configs — and
    end_run() deletes it, leaving nothing behind in the output directory.
    Returns the ledger path."""
    fd, log = tempfile.mkstemp(prefix='babel-artifacts-', suffix='.jsonl')
    os.close(fd)
    os.environ[_ENV_LOG] = log
    return log


def end_run():
    """Clear the recording env and delete the per-job temporary ledger."""
    log = os.environ.pop(_ENV_LOG, None)
    os.environ.pop(_ENV_STEP, None)
    if log:
        try:
            os.remove(log)
        except Exception:
            pass


def read_manifest(log_path=None, existing_only=False):
    """Read the sidecar and return a list of records (deduped by path, last
    write wins). With existing_only, drop entries whose file is gone."""
    log = log_path or os.environ.get(_ENV_LOG)
    seen = {}
    if log and os.path.isfile(log):
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                seen[e['path']] = e
    recs = list(seen.values())
    if existing_only:
        recs = [e for e in recs if os.path.isfile(e['path'])]
    return recs
