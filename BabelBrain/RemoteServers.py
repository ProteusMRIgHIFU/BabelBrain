"""
Persistent remote-server configuration for BabelBrain's client/server mode.

A client with no (or a weak) GPU can offload the three pipeline steps to a
BabelBrain job server running on a powerful workstation (see server.py). This
module stores the list of known servers so the user can pick one from the same
dropdown they use to pick a local GPU.

Configurations live in a small JSON file in the user's home (~/.babelbrain/
servers.json) — global to the machine, not per-dataset. Each entry is:

    {"name": "lab-workstation", "host": "10.0.0.5", "port": 8760, "token": null}

An HTTPS server adds TLS fields (all optional): "https" (use https://), "cafile"
(trust a private-CA/self-signed cert), "client_cert"/"client_key" (mutual TLS),
and "insecure" (skip verification — testing only).

Stdlib-only (json + urllib) so it stays importable anywhere, including the frozen
app and subprocesses.
"""
import json
import os
import ssl
import urllib.error
import urllib.request

CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.babelbrain')
SERVERS_FILE = os.path.join(CONFIG_DIR, 'servers.json')


def servers_path():
    return SERVERS_FILE


def _opt_str(s, key):
    """A stripped string field, or None when absent/empty."""
    return (str(s[key]).strip() or None) if s.get(key) else None


def _norm(s):
    """Normalise one server record; missing fields get sane defaults. TLS fields
    (https/cafile/client_cert/client_key/insecure) are optional and default off,
    so existing http-only records keep working unchanged."""
    return {'name': str(s.get('name', '')).strip(),
            'host': str(s.get('host', '127.0.0.1')).strip() or '127.0.0.1',
            'port': int(s.get('port', 8760) or 8760),
            'token': _opt_str(s, 'token'),
            'https': bool(s.get('https', False)),
            'cafile': _opt_str(s, 'cafile'),
            'client_cert': _opt_str(s, 'client_cert'),
            'client_key': _opt_str(s, 'client_key'),
            'insecure': bool(s.get('insecure', False))}


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
    scheme = "https" if server.get('https') else "http"
    return "%s://%s:%d" % (scheme, server['host'], int(server['port']))


def ssl_context(server):
    """SSLContext for an https server record (None for plain http). Trusts the
    record's cafile when set, loads a client cert for mutual TLS, and — only if
    'insecure' — skips verification (testing only)."""
    if not server.get('https'):
        return None
    ctx = ssl.create_default_context(cafile=server.get('cafile'))
    if server.get('insecure'):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    if server.get('client_cert'):
        ctx.load_cert_chain(server['client_cert'], server.get('client_key'))
    return ctx


def _get(url, token, timeout, context=None):
    headers = {}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    req = urllib.request.Request(url, headers=headers, method='GET')
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode() or '{}')


def test_connection(server, timeout=5):
    """Probe a server. Returns (ok: bool, info_dict | error_str). On success the
    dict carries the server's /healthz and /capabilities so the UI can confirm
    the token works and show what the server offers (transducers, features)."""
    base = base_url(server)
    ctx = ssl_context(server)
    try:
        health = _get(base + '/healthz', None, timeout, ctx)     # healthz needs no auth
        caps = _get(base + '/capabilities', server.get('token'), timeout, ctx)
        return True, {'health': health, 'capabilities': caps}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Unauthorized (401): a bearer token is required or wrong."
        return False, "HTTP %d: %s" % (e.code, e.reason)
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', e)
        if isinstance(reason, ssl.SSLError):
            return False, ("TLS error talking to %s: %s. Check the server's "
                           "certificate and the 'cafile' setting (or set "
                           "'insecure' to skip verification for testing)."
                           % (base, reason))
        return False, "Cannot reach %s: %s" % (base, reason)
    except ssl.SSLError as e:
        return False, "TLS error talking to %s: %s" % (base, e)
    except Exception as e:
        return False, str(e)
