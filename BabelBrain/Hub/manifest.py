'''
The curated release manifest.

The Hub does NOT read the raw GitHub Releases API — that would expose every
pre-release. Instead it fetches a self-contained ``releases.json`` that the
maintainers curate on the repository's default branch. This keeps full control
over which builds appear in the picker while still allowing pre-releases to be
published on GitHub.

Manifest shape (schema 1)::

    {
      "schema": 1,
      "hub": {"latest": "1.1.0", "min_supported": "1.0.0",
              "url": "https://.../BabelBrain-Hub-macos-arm64.zip", "sha256": "..."},
      "versions": [
        {"version": "0.8.8", "git_commit": "0799425", "channel": "stable",
         "recommended": true, "notes_url": "https://...",
         "assets": {
           "macos-arm64": {"url": "https://...zip", "sha256": "..."},
           "windows-x64": {"url": "https://...zip", "sha256": "..."}
         }}
      ]
    }

Only the ``assets`` entry matching the running platform is used.
'''
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import paths

MANIFEST_URL = (
    'https://raw.githubusercontent.com/ProteusMRIgHIFU/BabelBrain/main/releases.json'
)

SUPPORTED_SCHEMA = 1


def manifest_url() -> str:
    '''The manifest location. Overridable via BABEL_HUB_MANIFEST_URL, which
    accepts an http(s) URL, a file:// URL, or a bare local path. Useful for
    testing an unmerged catalog and for sites that host their own curated
    manifest.'''
    return os.environ.get('BABEL_HUB_MANIFEST_URL', '').strip() or MANIFEST_URL


@dataclass
class Asset:
    url: str
    sha256: str | None


@dataclass
class CatalogEntry:
    version: str
    channel: str                 # "stable" | "prerelease"
    git_commit: str | None
    recommended: bool
    notes_url: str | None
    asset: Asset | None          # asset for the current platform, if any

    @property
    def build_id(self) -> str:
        sc = self.git_commit[:7] if self.git_commit else None
        return f'{self.version}+{sc}' if sc else self.version


@dataclass
class HubUpdate:
    latest: str | None
    min_supported: str | None
    url: str | None
    sha256: str | None


@dataclass
class Manifest:
    entries: list[CatalogEntry]
    hub: HubUpdate
    from_cache: bool


def _fetch_raw(timeout: float) -> bytes:
    url = manifest_url()
    parsed = urllib.parse.urlparse(url)
    # Local file (bare path or file:// URL) — handy for testing / on-prem hosting.
    if parsed.scheme in ('', 'file'):
        path = urllib.request.url2pathname(parsed.path) if parsed.scheme == 'file' else url
        with open(path, 'rb') as f:
            return f.read()
    req = urllib.request.Request(url, headers={'User-Agent': 'BabelBrain-Hub'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse(data: dict, from_cache: bool) -> Manifest:
    schema = data.get('schema')
    if schema != SUPPORTED_SCHEMA:
        raise ValueError(f'unsupported manifest schema {schema!r} (need {SUPPORTED_SCHEMA})')

    pkey = paths.platform_key()
    entries: list[CatalogEntry] = []
    for raw in data.get('versions', []):
        assets = raw.get('assets') or {}
        a = assets.get(pkey)
        asset = Asset(url=a['url'], sha256=a.get('sha256')) if a and a.get('url') else None
        entries.append(CatalogEntry(
            version=str(raw.get('version', '0.0.0')),
            channel=str(raw.get('channel', 'stable')),
            git_commit=raw.get('git_commit'),
            recommended=bool(raw.get('recommended', False)),
            notes_url=raw.get('notes_url'),
            asset=asset,
        ))

    h = data.get('hub') or {}
    hub = HubUpdate(
        latest=h.get('latest'),
        min_supported=h.get('min_supported'),
        url=h.get('url'),
        sha256=h.get('sha256'),
    )
    return Manifest(entries=entries, hub=hub, from_cache=from_cache)


def fetch(timeout: float = 10.0, use_cache_on_error: bool = True) -> Manifest:
    '''Fetch and parse the manifest. On network/parse failure, fall back to the
    last cached copy if allowed; otherwise re-raise.'''
    try:
        raw = _fetch_raw(timeout)
        data = json.loads(raw)
        manifest = _parse(data, from_cache=False)
        # Cache only a manifest we could parse.
        try:
            cache = paths.manifest_cache_file()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        except OSError:
            pass
        return manifest
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        if use_cache_on_error:
            cache = paths.manifest_cache_file()
            if cache.is_file():
                data = json.loads(cache.read_text())
                return _parse(data, from_cache=True)
        raise


def visible_entries(manifest: Manifest, show_prereleases: bool) -> list[CatalogEntry]:
    '''Entries that have an asset for this platform, filtered by channel.'''
    out = []
    for e in manifest.entries:
        if e.asset is None:
            continue
        if e.channel == 'prerelease' and not show_prereleases:
            continue
        out.append(e)
    return out
