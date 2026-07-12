"""
Minimal BabelBrain job-server client (pure stdlib, no dependencies).

Assumes the client and server SHARE a filesystem (it sends real server paths).
For the no-shared-filesystem case (upload inputs / download results), see
server_client_remote.py. Shared HTTP helpers live in client_functions.py.

Start the server first, e.g. on the GPU machine:
    python BabelBrain/BabelBrain.py --serve --headless

Then run this client (adjust cf.BASE / cf.TOKEN and the JobSpec paths):
    python BabelBrain/server_client_example.py

It shows the whole lifecycle: submit -> follow progress -> collect artifacts.
Any language can talk to the same HTTP/JSON API; this is just a reference.
"""
import client_functions as cf

cf.BASE = "http://127.0.0.1:8760"
cf.TOKEN = None          # set to the --serve-token value if the server requires it


if __name__ == "__main__":
    print("capabilities:", cf._req("GET", "/capabilities"))

    # The advanced configuration is MANDATORY and client-owned. Start from the
    # server's clean defaults (or GET /currentconfig for the admin's baseline),
    # tweak what you need, and send the WHOLE dict with the job.
    config = cf._req("GET", "/defaultconfig")
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

    job_id = cf.submit(spec)
    print("submitted job", job_id)
    result = cf.follow(job_id)
    print("final:", result["state"], "exit", result["exit_code"])
    if result["state"] == "SUCCEEDED":
        print("artifacts:")
        for a in result["artifacts"]:
            print("  ", a)
    elif result["error"]:
        print("error:\n", result["error"])
