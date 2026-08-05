"""
BabelBrain job-server client for the NO-SHARED-FILESYSTEM case (pure stdlib).

server_client_example.py assumes the client and server see the same disk (it
sends real server paths). This example instead:

  1. creates a server-side *temp* workspace (deleted when we close it),
  2. uploads the input files into it (each upload returns its server path),
  3. submits a job that references those uploaded paths + the workspace id
     (so outputs are collected inside the workspace),
  4. follows progress,
  5. downloads every artifact back to a local folder,
  6. deletes the workspace (server removes the tempdir).

Shared HTTP helpers live in client_functions.py. Start the server first (it does
NOT need to see your data):
    python BabelBrain/BabelBrain.py --serve --headless
"""
import os

import client_functions as cf

cf.BASE = "http://127.0.0.1:8760"
cf.TOKEN = None                    # set to the --serve-token value if required

# --- LOCAL inputs on THIS machine (the server never sees these paths) ---------
LOCAL = "/Users/spichardo/Documents/TempForSim/SDR_0p55"
M2M_DIR = LOCAL + "/m2m_SDR_0p55"          # whole SimNIBS folder, staged with structure
CT = LOCAL + "/CT.nii.gz"
T1 = LOCAL + "/T1W.nii.gz"
FT = M2M_DIR + "/final_tissues.nii.gz"
FTLog = M2M_DIR + "/charm_log.html"
TRAJ = LOCAL + "/T1.txt"
THERMAL_PROFILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Profiles", "Thermal_Profile_1.yaml"))
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "downloaded_artifacts")


if __name__ == "__main__":
    caps = cf._req("GET", "/capabilities")
    print("features:", caps.get("features"))

    # 1) server-managed temp workspace (deleted when we delete it below)
    ws = cf.create_workspace(mode="temp")
    ws_id, ws_root = ws["workspace_id"], ws["root"]
    print("workspace:", ws_id, "->", ws_root)

    try:
        # 2) upload inputs; each call returns the server-side path to reference
        m2m_server = os.path.split(cf.upload(ws_id, FT, "final_tissues.nii.gz"))[0]
        cf.upload(ws_id,FTLog,"charm_log.html")
        t1_server =cf.upload(ws_id, T1, "T1W.nii.gz")
        ct_server = cf.upload(ws_id, CT, "CT.nii.gz")
        traj_server = cf.upload(ws_id, TRAJ, "T1.txt")
        prof_server = cf.upload(ws_id, THERMAL_PROFILE, "Thermal_Profile_1.yaml")
        print("uploaded inputs into workspace")

        config = cf._req("GET", "/defaultconfig")
        spec = {
            "t1w": t1_server,
            "simbnibs": m2m_server,
            "trajectory": traj_server,
            "thermal_profile": prof_server,
            "transducer": "CTX_250",
            "ct_type": "CT",
            "ct": ct_server,
            "workspace": ws_id,            # outputs default into the workspace
            "recalculate": True,
            "config": config,
            "steps": {
                "planning": [
                    {"action": "set", "control": "USMaskkHzDropDown", "value": "250"},
                    {"action": "set", "control": "USPPWSpinBox", "value": 6},
                    {"action": "set", "control": "HUThresholdSpinBox", "value": 300},
                    {"action": "run"},
                ],
                "acoustic": [
                    {"action": "select_trajectory", "index": 0},
                    {"action": "run"},
                ],
                "thermal": [
                    {"action": "select_trajectory", "index": 0},
                    {"action": "run"},
                    {"action": "click", "control": "ExportSummary", "wait": False},
                    {"action": "click", "control": "ExportMaps", "wait": False},
                ],
            },
        }

        # 3) + 4) submit and follow
        job_id = cf.submit(spec)
        print("submitted job", job_id)
        result = cf.follow(job_id)
        print("final:", result["state"], "exit", result["exit_code"])

        # 5) pull every artifact back over HTTP (no shared disk needed)
        if result["state"] == "SUCCEEDED":
            for local, n in cf.download_all(job_id, result["artifacts"], DOWNLOAD_DIR):
                print("  downloaded %10d B -> %s" % (n, local))
        elif result["error"]:
            print("error:\n", result["error"])
    finally:
        # 6) close the workspace -> server deletes the tempdir + everything in it
        print("closing workspace:", cf.delete_workspace(ws_id))
