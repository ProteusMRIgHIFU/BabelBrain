"""
Persistent remote-server configuration for BabelBrain's client/server mode.

A client with no (or a weak) GPU can offload the three pipeline steps to a
BabelBrain job server running on a powerful workstation (see server.py). This
module stores the list of known servers so the user can pick one from the same
dropdown they use to pick a local GPU.

Configurations live in a small JSON file in the user's home (~/.babelbrain/
servers.json) — global to the machine, not per-dataset. Each entry is:

    {"name": "lab-workstation", "host": "10.0.0.5", "port": 8760, "token": null}

Stdlib-only (json + urllib) so it stays importable anywhere, including the frozen
app and subprocesses.
"""
import json
import os
import urllib.error
import urllib.request

CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.babelbrain')
SERVERS_FILE = os.path.join(CONFIG_DIR, 'servers.json')


def servers_path():
    return SERVERS_FILE


def _norm(s):
    """Normalise one server record; missing fields get sane defaults."""
    return {'name': str(s.get('name', '')).strip(),
            'host': str(s.get('host', '127.0.0.1')).strip() or '127.0.0.1',
            'port': int(s.get('port', 8760) or 8760),
            'token': (str(s['token']).strip() or None) if s.get('token') else None}


def load_servers():
    """Return the saved servers as a list of dicts (empty list if none/error)."""
    try:
        with open(SERVERS_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [_norm(s) for s in data if isinstance(s, dict) and s.get('name')]
    except (FileNotFoundError, ValueError, OSError):
        pass
    return []


def save_servers(servers):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(SERVERS_FILE, 'w') as f:
        json.dump([_norm(s) for s in servers], f, indent=2)


def get_server(name):
    for s in load_servers():
        if s['name'] == name:
            return s
    return None


def add_or_update(server):
    """Add a server (or replace the one with the same name). Returns the list."""
    server = _norm(server)
    servers = [s for s in load_servers() if s['name'] != server['name']]
    servers.append(server)
    save_servers(servers)
    return servers


def remove(name):
    servers = [s for s in load_servers() if s['name'] != name]
    save_servers(servers)
    return servers


def base_url(server):
    return "http://%s:%d" % (server['host'], int(server['port']))


def _get(url, token, timeout):
    headers = {}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    req = urllib.request.Request(url, headers=headers, method='GET')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or '{}')


def test_connection(server, timeout=5):
    """Probe a server. Returns (ok: bool, info_dict | error_str). On success the
    dict carries the server's /healthz and /capabilities so the UI can confirm
    the token works and show what the server offers (transducers, features)."""
    base = base_url(server)
    try:
        health = _get(base + '/healthz', None, timeout)          # healthz needs no auth
        caps = _get(base + '/capabilities', server.get('token'), timeout)
        return True, {'health': health, 'capabilities': caps}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Unauthorized (401): a bearer token is required or wrong."
        return False, "HTTP %d: %s" % (e.code, e.reason)
    except urllib.error.URLError as e:
        return False, "Cannot reach %s: %s" % (base, getattr(e, 'reason', e))
    except Exception as e:
        return False, str(e)
