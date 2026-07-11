"""
Minimal BabelBrain job-server client (pure stdlib, no dependencies).

Start the server first, e.g. on the GPU machine:
    python BabelBrain/BabelBrain.py --serve --headless

Then run this client (adjust BASE, TOKEN and the JobSpec paths):
    python BabelBrain/server_client_example.py

It shows the whole lifecycle: submit -> follow progress -> collect artifacts.
Any language can talk to the same HTTP/JSON API; this is just a reference.
"""
import json
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8760"
TOKEN = None          # set to the --serve-token value if the server requires it


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = "Bearer " + TOKEN
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read().decode() or "{}")


def submit(spec):
    return _req("POST", "/jobs", spec)["job_id"]


def follow(job_id, poll=1.0):
    """Poll status + new events until the job reaches a terminal state."""
    seen = 0
    while True:
        for ev in _req("GET", "/jobs/%s/events?since=%d" % (job_id, seen))["events"]:
            print("  [%3d%%] %-9s %s" % (ev["percent"], ev["phase"], ev["message"]))
            seen += 1
        status = _req("GET", "/jobs/%s" % job_id)
        if status["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return status
        time.sleep(poll)


if __name__ == "__main__":
    print("capabilities:", _req("GET", "/capabilities"))

    # The advanced configuration is MANDATORY and client-owned. Start from the
    # server's clean defaults (or GET /currentconfig for the admin's baseline),
    # tweak what you need, and send the WHOLE dict with the job.
    config = _req("GET", "/defaultconfig")
    config["TrabecularProportion"] = 0.2          # example override

    # A JobSpec = input-selection fields + the mandatory config + `steps`.
    # `steps` maps planning/acoustic/thermal to an ORDERED list of actions; the
    # client controls both the settings and their order (like scripting).
    # Note: frequency / PPW / HU threshold are Step-1 controls -> planning actions.
    
    spec = {
        "t1w": "/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/T1.nii.gz",
        "simbnibs": "/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55",
        "trajectory": "/Users/spichardo/Documents/TempForSim/SDR_0p55/T1.txt",
        "thermal_profile": "/Users/spichardo/Documents/GitHub/BabelBrain/Profiles/Thermal_Profile_1.yaml",
        "transducer": "CTX_250",
        "ct_type": "CT",
        "ct": "/Users/spichardo/Documents/TempForSim/SDR_0p55/CT.nii.gz",
        "output_path": "/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55",
        "recalculate": True,          # answer "recalculate" to cached-result prompts

        "config": config,             # MANDATORY advanced configuration

        "steps": {
            "planning": [
                {"action": "set", "control": "USMaskkHzDropDown", "value": "250"},  # frequency
                {"action": "set", "control": "USPPWSpinBox", "value": 6},           # PPW
                {"action": "set", "control": "HUThresholdSpinBox", "value": 300},   # bone HU
                {"action": "run"},
            ],
            "acoustic": [
                {"action": "select_trajectory", "index": 0},
                # Steering control is transducer-specific:
                #   phased arrays (H317/DomeTx/REMOPD) -> "ZSteeringSpinBox"
                #   ring transducers (CTX/DPX)         -> "TPODistanceSpinBox"
                #     (GUI label "Z Steering (mm)"; a distance in mm within the
                #      device's post-Step-1 range). Omit to use the auto default.
                # e.g. CTX:  {"action": "set", "control": "TPODistanceSpinBox", "value": <mm>},
                {"action": "run"},
                # {"action": "select_trajectory", "index": 1}, {"action": "run"},
                # {"action": "click", "control": "CombineTrajectories"},
            ],
            "thermal": [
                {"action": "select_trajectory", "index": 0},
                {"action": "run"},
                {"action": "click", "control": "ExportSummary", "wait": False},
                {"action": "click", "control": "ExportMaps", "wait": False},
            ],
        },
    }

    job_id = submit(spec)
    print("submitted job", job_id)
    result = follow(job_id)
    print("final:", result["state"], "exit", result["exit_code"])
    if result["state"] == "SUCCEEDED":
        print("artifacts:")
        for a in result["artifacts"]:
            print("  ", a)
    elif result["error"]:
        print("error:\n", result["error"])
