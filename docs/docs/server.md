# Server mode

BabelBrain can run as a **job server**: it exposes the pipeline over a small
HTTP/JSON API so language-agnostic clients (Python, C#, C++, `curl`, …) can
submit jobs, follow progress, and collect results over the network. It is the
same engine used by [scripting](Coding/scripting.md), with a network front-end and a
job queue.


**Important notes:**

* **Multiple workers** — one BabelBrain process drives per GPU available in the server, so
  jobs are processed simultaneously as many GPU are available.
* **Stateless server operation mode** - while persistant mode (data permanently stored in server) is feasible, the recommended operation is to use the server as stateless, where the client sends the data needed for calculations and recovers the artifacts.
* **Declarative jobs** — a job specifies inputs, which steps to run, and
  parameter overrides (no arbitrary remote code).
* **Local first** — bind to `localhost` by default. A bearer-token hook is
  built in;

## Starting the server

```bash
python BabelBrain/BabelBrain.py --serve --headless               # localhost:8760, no GUI shown
python BabelBrain/BabelBrain.py --serve --serve-host 164.5.34.1  # specify IP address of exposed network interface to accept requests
python BabelBrain/BabelBrain.py --serve --serve-host 164.5.34.1 --serve-port 9000 --serve-token SECRET # specify also different port number and secret TOKEN 
```

Token use is **HIGHLY RECOMMENDED**, and often required by IT, when using in a large network
### Running from the binary distribution
The frozen application works the same way (`BabelBrain.exe --serve …` /
`BabelBrain.app/Contents/MacOS/BabelBrain --serve …`). Stop the server with
`Ctrl-C`.

### Server mode options

| Option | Effect |
| --- | --- |
| `--serve` | Run as a job server. |
| `--serve-host` | Bind address (default `127.0.0.1`). |
| `--serve-port` | Port (default `8760`). |
| `--serve-token` | Require `Authorization: Bearer <token>` on every endpoint (or set `BABEL_SERVER_TOKEN`). |
| `--headless` | Run with no window (for a remote/headless GPU box). |

# Client configuration
In BabelBrain client operation (normal GUI mode), the uer can use a remote server in the same way as it would be a local GPU. No data will be preserved in the server. Be aware that during the execution of a session, files will be transferred back and forth between the client and server. Files will be deleted once the client closes its session or the server is stopped.

The user can configure a remote server in the list of "Computing Backends":

<img src="server_config_1.png" height=350px>

this will show a small dialog where the user can add a new server where IP address, port and optional secret token can be specified

<img src="server_config_2.png" height=250px>

Be aware that multiple server entries can be created.

Once defined, the configured servers will be available as a list of computing backend:

<img src="server_config_3.png" height=350px>



# Server API

| Method & path | Purpose |
| --- | --- |
| `GET /healthz` | Liveness + `{busy, queued}`. The only endpoint that never needs auth. |
| `GET /capabilities` | Server info, incl. the list of available `transducers`. |
| `GET /defaultconfig` | BabelBrain's **default** advanced configuration (clean baseline). |
| `GET /currentconfig` | The server's **current** advanced configuration (which may be needed to recover some paths for tools like SimNIBS). |
| `POST /jobs` | Submit a [JobSpec](#jobspec) (the `config` field is **mandatory**); returns `{job_id}` (`202`). |
| `GET /jobs` | List all jobs (summaries). |
| `GET /jobs/{id}` | Job status: `state`, `phase`, `percent`, `error`, `artifacts`, `exit_code`. |
| `GET /jobs/{id}/events?since=N` | Progress events from index `N` (polling). |
| `GET /jobs/{id}/stream` | Same events as **Server-Sent Events** (push). |
| `POST /jobs/{id}/cancel` | Cancel a queued job, or request cancellation of the running one (honored between steps). |

### Configuration is client-owned and mandatory

The [advanced configuration](Coding/advanced_config.md) is **owned by the client**, not
stored per-session on the server. Every job **must** carry its own `config`
object; a submission without it is rejected (`400`).

The usual flow is: fetch a baseline, tweak, and send the whole dictionary back:

* `GET /defaultconfig` — BabelBrain's pristine defaults (what the immense
  majority of clients want: a clean, deterministic starting point).
* `GET /currentconfig` — the server's current values (e.g. a sanitized site
  baseline an operator configured). Read-only.

In **server mode the configuration is memory-only**: the server never writes
`lastselection.yaml`, and a job's config affects only that job. (Persisting is
allowed only for a normal, local non-server session.)

### JobSpec

A `JobSpec` (POST body, JSON) has three parts: input-selection fields, the
mandatory `config`, and `steps`.

```json
{
  "t1w": "/data/sub-01/T1W.nii.gz",
  "simbnibs": "/data/sub-01/m2m_sub-01/",
  "trajectory": "/data/sub-01/target.txt",
  "thermal_profile": "/data/profile.yaml",
  "transducer": "CTX_500",
  "ct_type": "CT",
  "ct": "/data/sub-01/CT.nii.gz",
  "output_path": "/data/sub-01/output/",
  "recalculate": true,

  "config": { "...": "the full advanced-config object from /defaultconfig" },

  "steps": {
    "planning": [
      {"action": "set", "control": "USMaskkHzDropDown", "value": "500"},
      {"action": "set", "control": "USPPWSpinBox", "value": 6},
      {"action": "set", "control": "HUThresholdSpinBox", "value": 300},
      {"action": "run"}
    ],
    "acoustic": [
      {"action": "select_trajectory", "index": 0},
      {"action": "set", "control": "ZSteeringSpinBox", "value": 3.0},
      {"action": "set", "control": "XSteeringSpinBox", "value": 0.0},
      {"action": "run"},
      {"action": "select_trajectory", "index": 1},
      {"action": "run"},
      {"action": "click", "control": "CombineTrajectories"}
    ],
    "thermal": [
      {"action": "select_trajectory", "index": 0},
      {"action": "run"},
      {"action": "click", "control": "ExportSummary", "wait": false},
      {"action": "click", "control": "ExportMaps", "wait": false}
    ]
  }
}
```

* **Input fields** (`t1w`, `simbnibs`, `transducer`, `ct_type`, …) are the
  [`launch()` inputs](Coding/scripting.md#launch-selecting-inputs) — the SelFiles-level
  selections. Step-1 controls (frequency, PPW, HU threshold) are **not** here;
  they are `planning` actions, so the client controls them and their order.
* `config` — **mandatory** advanced configuration (see above).
* `recalculate` — `true` recomputes; `false` reloads cached results.
* `use_last_selection` — seed omitted input fields from the last GUI session.
* `steps` — an object with any of `planning` / `acoustic` / `thermal`, each an
  **ordered list of actions**. Steps run in pipeline order; actions run exactly
  in the order given.

### Actions

Each action targets the controls of its step (`planning` → the main window,
`acoustic`/`thermal` → the current trajectory tab). Control names are the same
as in [Scriptable operations](Coding/operations.md).

| Action | Fields | Effect |
| --- | --- | --- |
| `set` | `control`, `value` | Set a control (spin box / check box / drop-down / `USPPWSpinBox`). |
| `select_trajectory` | `index` | Select a trajectory tab (`acoustic`/`thermal` only). |
| `run` | *(optional)* `timeout_ms` | Click the step's run button and wait for completion. |
| `click` | `control`, *(optional)* `wait` (default `true`), `timeout_ms` | Click any button (e.g. `CombineTrajectories`, `CalculateMechAdj`, `ExportSummary`); use `"wait": false` for exports. |

This gives full, ordered control — e.g. *select trajectory 0 → set Z-steering →
run → select trajectory 1 → run → combine* — mirroring what you'd do in the GUI
or in a [script](Coding/operations.md).

### Job lifecycle

`QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED`. On `SUCCEEDED`, `artifacts`
lists the produced files (`.h5`, `.nii.gz`, `.csv`) under `output_path`; on
`FAILED`, `error` carries the traceback and `exit_code` is non-zero.

## Client examples

`curl` (remember `config` is mandatory — fetch a baseline first):

```bash
curl -s localhost:8760/capabilities
curl -s localhost:8760/defaultconfig > config.json      # baseline to embed in the job
# ...build a JobSpec (with "config": <config.json>) as spec.json, then:
JOB=$(curl -s -XPOST localhost:8760/jobs --data @spec.json | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
curl -s localhost:8760/jobs/$JOB            # poll status
curl -sN localhost:8760/jobs/$JOB/stream    # follow progress (SSE)
```

A complete, dependency-free **Python reference client** ships with the source at

* `BabelBrain/client_server_remote.py` - This mimics how the GUI client BabelBrain operates, where local temporary directories at the server are used to store files during the session life, the client sends input files to the server, then artifacts (results files) are recovered from the server.
* `BabelBrain/client_server_session.py` - This shows how to keep the session alive between Steps 1 to 3, which can be needed when some temporary state persistance is needed. The complete GUI client combines this session mode and the remote temporary file storage shown in `client_server_remote.py`.
* `BabelBrain/client_server_persistent.py` - This shows the case where file paths refer to local locations in the server, showing how to run simulations in the case results are desired to remain in the filesystem of the server.



# Security

For the initial local-premises use (a client on the same LAN as the GPU box) the
server binds to `localhost` and auth is optional. Before any exposure beyond a
trusted machine, **configure a token** (`--serve-token`) and put it behind TLS —
the server drives a GPU and reads/writes files, so treat it as an attack surface
from the outset.
