"""
Persistent-session BabelBrain client (pure stdlib, shared filesystem).

server_client_example.py runs the whole pipeline in ONE job. In practice a client
often wants to run part of it, look at the result, then decide what to do next.
This example keeps ONE BabelBrain session alive across THREE separate jobs:

    job 1: planning              (keep_alive=True)   -> Step-1 domain
    job 2: acoustic trajectory 0 (keep_alive=True)   -> reuses the live session
    job 3: thermal  trajectory 0 (keep_alive=False)  -> reuses, then closes it

Because the session persists, job 2/3 do NOT relaunch or recompute Step 1 — they
continue on the worker job 1 left behind. The server returns a session_id on the
first job (cf.LAST_SESSION_ID); pass it on the next jobs to reuse that worker.
Set keep_alive=False on the last call (or send DELETE /sessions/{id}) to release
the session. On a multi-GPU server a GPU is bound only while a 'run' computes, so
a persistent session does not tie up a GPU between steps. Shared HTTP helpers
live in client_functions.py.

Start the server first:
    python BabelBrain/BabelBrain.py --serve --headless
"""
import client_functions as cf

cf.BASE = "http://127.0.0.1:8760"
cf.TOKEN = None          # set to the --serve-token value if the server requires it

LOCAL = "/Users/spichardo/Documents/TempForSim/SDR_0p55"


def run(label, steps, keep_alive, session_id=None, extra=None):
    """Submit one job (config is mandatory on every job), follow it, return the
    final status. `extra` carries the input-selection fields (only needed on the
    first job — later jobs reuse the live session and ignore inputs). Passing
    `session_id` routes the job to the SAME worker so Step-1 state is retained."""
    spec = {"config": cf._req("GET", "/defaultconfig"),
            "recalculate": True, "keep_alive": keep_alive, "steps": steps}
    if session_id:
        spec["session_id"] = session_id
    spec.update(extra or {})
    print("\n=== %s (keep_alive=%s) ===" % (label, keep_alive))
    job_id = cf.submit(spec)
    result = cf.follow(job_id)
    print("  ->", result["state"], "| session:", cf.LAST_SESSION_ID)
    return result


if __name__ == "__main__":
    print("features:", cf._req("GET", "/capabilities").get("features"))

    inputs = {
        "t1w": LOCAL + "/m2m_SDR_0p55/T1.nii.gz",
        "simbnibs": LOCAL + "/m2m_SDR_0p55",
        "trajectory": LOCAL + "/T1.txt",
        "thermal_profile": "/Users/spichardo/Documents/GitHub/BabelBrain/Profiles/Thermal_Profile_1.yaml",
        "transducer": "CTX_250",
        "ct_type": "CT",
        "ct": LOCAL + "/CT.nii.gz",
        "output_path": LOCAL + "/m2m_SDR_0p55",
    }

    # job 1 — planning; open the session and keep it alive. The server returns a
    # session_id (recorded in cf.LAST_SESSION_ID) to route the next jobs here.
    result = run("planning", {"planning": [
        {"action": "set", "control": "USMaskkHzDropDown", "value": "250"},
        {"action": "set", "control": "USPPWSpinBox", "value": 6},
        {"action": "set", "control": "HUThresholdSpinBox", "value": 300},
        {"action": "run"},
    ]}, keep_alive=True, extra=inputs)
    sid = cf.LAST_SESSION_ID
    print("session id:", sid)

    if result["state"] == "SUCCEEDED":
        print("planning-job artifacts:")
        for a in result["artifacts"]:
            print("  ", a["path"])
    elif result["error"]:
        print("error:\n", result["error"])

    # job 2 — acoustic; SAME session -> reuses the live worker, Step-1 retained
    result = run("acoustic", {"acoustic": [
        {"action": "select_trajectory", "index": 0},
        {"action": "run"},
    ]}, keep_alive=True, session_id=sid)

    if result["state"] == "SUCCEEDED":
        print("acoustic-job artifacts:")
        for a in result["artifacts"]:
            print("  ", a["path"])
    elif result["error"]:
        print("error:\n", result["error"])

    # job 3 — thermal; reuse then close the session (keep_alive=False ends it)
    result = run("thermal", {"thermal": [
        {"action": "select_trajectory", "index": 0},
        {"action": "run"},
        {"action": "click", "control": "ExportSummary", "wait": False},
        {"action": "click", "control": "ExportMaps", "wait": False},
    ]}, keep_alive=False, session_id=sid)

    # belt-and-suspenders: make sure the session is gone
    print("\nDELETE /sessions/%s ->" % sid, cf._req("DELETE", "/sessions/%s" % sid))

    if result["state"] == "SUCCEEDED":
        print("thermal-job artifacts:")
        for a in result["artifacts"]:
            print("  ", a["path"])
    elif result["error"]:
        print("error:\n", result["error"])
