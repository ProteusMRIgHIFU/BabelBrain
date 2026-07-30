#!/usr/bin/env python3
"""Fetch / pin the external PlanTUS tool for BabelBrain.

BabelBrain drives the external PlanTUS planner (https://github.com/mlueckel/PlanTUS).
PlanTUS is actively developed, so BabelBrain pins one known-good commit. That pin
lives in two places kept in sync:

  * as a git submodule under BabelBrain/ExternalBin/PlanTUS/PlanTUS
    (the source of truth for people who clone BabelBrain with git), and
  * as ``PLANTUS_PIN`` below, so this script can also populate the folder for
    users who obtained BabelBrain as a zip / standalone build (no submodule),
    or repair a checkout that drifted.

Usage (from the repo root or anywhere)::

    python Scripts/fetch_plantus.py            # ensure PlanTUS is present at the pin
    python Scripts/fetch_plantus.py --force    # re-checkout the pin even if present
    python Scripts/fetch_plantus.py --check    # report status, do not modify

To bump the pinned version: update PLANTUS_PIN here AND move the submodule
gitlink (``cd`` into the submodule, ``git checkout <new-sha>``, ``git add`` it in
the parent repo, commit). Keep the two in sync.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PLANTUS_URL = "https://github.com/mlueckel/PlanTUS"
PLANTUS_PIN = "97d908b439c78a5d8b4fed9bba750ae2a99cce3b"

# A file that must exist inside a valid PlanTUS checkout.
PLANTUS_SENTINEL = "PlanTUS_wrapper.py"

# Default destination, relative to this file: <repo>/BabelBrain/ExternalBin/PlanTUS/PlanTUS
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = _REPO_ROOT / "BabelBrain" / "ExternalBin" / "PlanTUS" / "PlanTUS"


def _run(args, cwd=None, check=True):
    return subprocess.run(
        args, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, check=check,
    )


def _git_available():
    try:
        _run(["git", "--version"])
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _current_sha(dest: Path):
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def is_plantus_present(dest: Path = DEFAULT_DEST) -> bool:
    """True if a PlanTUS checkout at the pinned commit exists at ``dest``."""
    dest = Path(dest)
    if not (dest / PLANTUS_SENTINEL).is_file():
        return False
    sha = _current_sha(dest)
    # If it is a git checkout, require the pin; a bundled (non-git) copy that has
    # the sentinel is trusted as-is.
    if sha is None:
        return True
    return sha == PLANTUS_PIN


def ensure_plantus(dest: Path = DEFAULT_DEST, force: bool = False, verbose: bool = True) -> Path:
    """Make sure PlanTUS is checked out at ``PLANTUS_PIN`` under ``dest``.

    Returns the destination path. Raises RuntimeError on failure.
    """
    dest = Path(dest)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    if is_plantus_present(dest) and not force:
        log(f"PlanTUS already present at pinned commit: {dest}")
        return dest

    if not _git_available():
        raise RuntimeError(
            "git is required to fetch PlanTUS but was not found on PATH. "
            f"Install git, or manually place PlanTUS ({PLANTUS_URL} @ {PLANTUS_PIN}) at {dest}."
        )

    # Prefer the submodule path when BabelBrain is a git checkout: it respects the
    # gitlink recorded in the parent repo and keeps everything consistent.
    used_submodule = False
    parent_is_git = (_REPO_ROOT / ".git").exists()
    gitmodules = _REPO_ROOT / ".gitmodules"
    if parent_is_git and gitmodules.is_file() and Path(dest) == DEFAULT_DEST:
        try:
            rel = dest.relative_to(_REPO_ROOT)
            log(f"Initializing PlanTUS submodule: {rel}")
            _run(["git", "submodule", "update", "--init", "--", str(rel)], cwd=_REPO_ROOT)
            used_submodule = True
        except subprocess.CalledProcessError as e:
            log(f"submodule update failed, falling back to direct clone: {e.stderr.strip()}")

    if not used_submodule and not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"Cloning PlanTUS into {dest}")
        try:
            _run(["git", "clone", PLANTUS_URL, str(dest)])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone PlanTUS: {e.stderr.strip()}") from e

    # Force the exact pin (self-heals drift and applies --force).
    if _current_sha(dest) != PLANTUS_PIN:
        log(f"Checking out pinned commit {PLANTUS_PIN}")
        try:
            _run(["git", "fetch", "--quiet", "origin", PLANTUS_PIN], cwd=dest, check=False)
            _run(["git", "checkout", "--quiet", PLANTUS_PIN], cwd=dest)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to check out PlanTUS pin {PLANTUS_PIN}: {e.stderr.strip()}"
            ) from e

    if not is_plantus_present(dest):
        raise RuntimeError(
            f"PlanTUS checkout at {dest} is missing {PLANTUS_SENTINEL} after fetch."
        )
    log(f"PlanTUS ready at {dest} ({PLANTUS_PIN})")
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch/pin the external PlanTUS tool.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST),
                        help="Destination folder (default: bundled ExternalBin path).")
    parser.add_argument("--force", action="store_true",
                        help="Re-checkout the pinned commit even if already present.")
    parser.add_argument("--check", action="store_true",
                        help="Only report whether PlanTUS is present at the pin; make no changes.")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    if args.check:
        present = is_plantus_present(dest)
        sha = _current_sha(dest)
        print(f"dest:    {dest}")
        print(f"present: {present}")
        print(f"HEAD:    {sha or '(not a git checkout / missing)'}")
        print(f"pin:     {PLANTUS_PIN}")
        return 0 if present else 1

    try:
        ensure_plantus(dest, force=args.force)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
