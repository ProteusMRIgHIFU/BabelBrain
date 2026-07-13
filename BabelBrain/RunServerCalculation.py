"""
Client-side offload of the BabelBrain pipeline to a remote job server.

When the user selects a remote server as the "computing engine"
(Config['ComputingBackend'] == 5, Config['RemoteServer'] set), the three local
workers (RunMaskGeneration / RunSimulation / RunThermalSim) should NOT spawn a
local GPU subprocess. Instead they hand the step to this class, which:

  1. (once) opens a persistent session on the server and uploads the inputs,
  2. submits the step as a job carrying the client's advanced configuration,
  3. follows progress and streams telemetry back to the GUI,
  4. downloads the produced artifacts to the same local paths the GUI expects,
  5. keeps the session alive so the next step reuses Step-1 state, and closes it
     when the pipeline is done.

This is introduced incrementally. THIS increment implements the connection
preflight and the advanced-config payload (the local AdvancedOptions that must be
sent to the server); submit/upload/download are wired in later increments.

It reuses the stdlib client helpers in client_functions.py and the saved-server
records in RemoteServers.py.
"""
import client_functions as cf
import RemoteServers

# Map the GUI step to the server's `steps` key (see server.py _STEP).
STEP_PLANNING = 'planning'
STEP_ACOUSTIC = 'acoustic'
STEP_THERMAL = 'thermal'


class RemoteNotReady(Exception):
    """Raised when the configured server can't be reached / used."""


class RunServerCalculation:
    """Prepares and drives one pipeline step on a remote server. One instance per
    step; a shared session on the server ties the steps together."""

    def __init__(self, mainApp, step):
        self._mainApp = mainApp
        self._step = step
        self._server = (mainApp.Config or {}).get('RemoteServer')
        if not self._server:
            raise RemoteNotReady("No remote server is configured for this session.")

    # -- server addressing -------------------------------------------------
    def _bind(self):
        """Point the shared client helpers at this job's server."""
        cf.BASE = RemoteServers.base_url(self._server)
        cf.TOKEN = self._server.get('token')

    def preflight(self):
        """Verify the server is reachable and capable. Returns the capabilities
        dict, or raises RemoteNotReady with a user-facing message."""
        ok, info = RemoteServers.test_connection(self._server)
        if not ok:
            raise RemoteNotReady("Remote server '%s' is not available:\n%s"
                                 % (self._server.get('name', '?'), info))
        caps = info.get('capabilities', {})
        needed = {'workspaces', 'uploads', 'artifact_download', 'persistent_session'}
        missing = needed - set(caps.get('features', []))
        if missing:
            raise RemoteNotReady("Server '%s' is missing required features: %s"
                                 % (self._server.get('name', '?'), ", ".join(sorted(missing))))
        return caps

    # -- advanced configuration -------------------------------------------
    def advanced_config_payload(self):
        """Build the advanced-configuration dict to send with every job: the
        server's own default advanced config as the key set, overlaid with the
        client's current AdvancedOptions values. Fetching the server default
        guarantees the key set matches what the server expects, while the overlay
        carries the local user's choices (the whole point of client-owned config).
        """
        self._bind()
        default = cf._req("GET", "/defaultconfig")
        cfg = dict(self._mainApp.Config or {})
        payload = {}
        for k, dv in default.items():
            payload[k] = cfg[k] if k in cfg else dv
        return payload

    # -- step execution (wired in later increments) ------------------------
    def run(self):
        raise NotImplementedError(
            "Remote step execution is being implemented incrementally; the "
            "job submit/upload/download path is not wired yet.")
