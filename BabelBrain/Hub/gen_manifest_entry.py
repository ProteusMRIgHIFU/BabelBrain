'''
Print a releases.json "versions" entry for a freshly built release.

The Hub reads a *curated* releases.json (only listed versions appear in the
launcher), so publishing a version is a deliberate step: run this after a
release build to get a ready-to-paste JSON entry, then add it to releases.json
on the default branch via a PR. Pre-releases you do not want users to see simply
never get added (or are added with ``--channel prerelease``).

The per-platform asset URLs point at the release's uploaded ``*-version.zip``
assets; the sha256 values come from the matching ``*-version.zip.sha256`` files
produced by the build workflow.

Example::

    python Hub/gen_manifest_entry.py \
        --version 0.8.8 --commit 0799425 --channel stable --recommended \
        --notes-url https://github.com/ProteusMRIgHIFU/BabelBrain/releases/tag/v0.8.8 \
        --asset macos-arm64=https://.../BabelBrain-macOS-arm64-version.zip:SHA \
        --asset windows-x64=https://.../BabelBrain-Windows-x64-version.zip:SHA
'''
from __future__ import annotations

import argparse
import json


def _parse_asset(spec: str) -> tuple[str, dict]:
    # platform=url:sha256  (url may contain ':' so split sha256 from the right)
    key, _, rest = spec.partition('=')
    url, _, sha = rest.rpartition(':')
    if not key or not url:
        raise argparse.ArgumentTypeError(
            f'bad --asset {spec!r}; expected platform=url:sha256')
    return key, {'url': url, 'sha256': sha}


def main() -> int:
    ap = argparse.ArgumentParser(description='Emit a releases.json versions[] entry')
    ap.add_argument('--version', required=True)
    ap.add_argument('--commit', default=None, help='git commit (short is fine)')
    ap.add_argument('--channel', default='stable', choices=['stable', 'prerelease'])
    ap.add_argument('--recommended', action='store_true')
    ap.add_argument('--notes-url', default=None)
    ap.add_argument('--asset', action='append', type=_parse_asset, default=[],
                    metavar='platform=url:sha256',
                    help='repeatable; platform in macos-arm64/macos-x64/windows-x64/linux-x64')
    args = ap.parse_args()

    entry = {
        'version': args.version,
        'git_commit': args.commit,
        'channel': args.channel,
        'recommended': args.recommended,
        'notes_url': args.notes_url,
        'assets': dict(args.asset),
    }
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
