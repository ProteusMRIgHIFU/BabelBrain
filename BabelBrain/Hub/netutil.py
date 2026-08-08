'''
TLS setup for the Hub's HTTPS requests.

A PyInstaller-frozen app does not inherit the build machine's OpenSSL default CA
path, so plain ``urllib`` HTTPS calls fail certificate verification on the user's
machine ("CERTIFICATE_VERIFY_FAILED"). We therefore build an SSL context backed
by certifi's CA bundle, which the Hub bundles (certifi.where() resolves to the
bundled cacert.pem when frozen). Running from source, certifi is present in the
environment and this works unchanged; if certifi is somehow unavailable we fall
back to the system default context.
'''
from __future__ import annotations

import ssl


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - any failure -> system default trust store
        return ssl.create_default_context()
