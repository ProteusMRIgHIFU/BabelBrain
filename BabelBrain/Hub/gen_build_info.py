'''
Generate ``build_info.json`` for a BabelBrain build.

Run at CI build time (before PyInstaller) so the frozen bundle carries a stable
identity the Hub can use to tell builds apart — ``(version, git_commit)`` —
which is what lets a bundled dev 0.8.8 coexist with a released 0.8.8.

Usage::

    python Hub/gen_build_info.py --channel stable --out build_info.json

Version is read from version.txt; git metadata from the current checkout.
Override any field with the matching flag (useful in CI where the tag is known).
'''
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(['git', *args], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description='Write a BabelBrain build_info.json')
    ap.add_argument('--out', default='build_info.json')
    ap.add_argument('--version', default=None, help='override version.txt')
    ap.add_argument('--channel', default='stable', choices=['stable', 'prerelease', 'dev'])
    ap.add_argument('--commit', default=None, help='override git commit')
    ap.add_argument('--tag', default=None, help='release tag this build came from')
    args = ap.parse_args()

    version = args.version
    if version is None:
        vfile = Path('version.txt')
        version = vfile.read_text().strip() if vfile.is_file() else '0.0.0'

    info = {
        'version': version,
        'git_commit': args.commit or _git('rev-parse', 'HEAD'),
        'channel': args.channel,
        'tag': args.tag or _git('describe', '--tags', '--exact-match'),
        'built': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    Path(args.out).write_text(json.dumps(info, indent=2) + '\n')
    print(f'Wrote {args.out}: {info}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
