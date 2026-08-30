# This Python file uses the following encoding: utf-8
'''
Identity of the running BabelBrain build.

Shared by the main window and the initial file-selection dialog, which both put
the build in their title bar. It lives in its own module because
``SelFiles.SelFiles`` is imported *by* ``BabelBrain.py`` — importing back the
other way for these two helpers would be circular.
'''
import json
import sys
from pathlib import Path


def _read_build_info():
    """``build_info.json`` of the *running frozen build*, or None.

    The file is stamped by ``Hub/gen_build_info.py`` just before PyInstaller
    runs and copied into the bundle afterwards, so it identifies this exact
    build by ``(version, git_commit, channel, built)``.

    Only consulted when frozen. A source checkout also has a git-tracked
    ``build_info.json`` next to ``BabelBrain.py``, and it describes whatever
    build was packaged last, not the working tree — labelling a source run with
    it would be actively misleading.

    Several candidate locations because the file is copied in after PyInstaller
    rather than bundled as a data file: on macOS ``_MEIPASS`` is
    ``Contents/Frameworks`` while the copy lands in ``Contents/Resources``
    (with no symlink, unlike the spec-declared datas); on Windows onedir it
    sits next to the .exe.
    """
    if not getattr(sys, 'frozen', False):
        return None
    meipass = getattr(sys, '_MEIPASS', None)
    candidates = []
    if meipass:
        candidates += [Path(meipass) / 'build_info.json',
                       Path(meipass).parent / 'Resources' / 'build_info.json']
    exe = Path(sys.executable).resolve()
    candidates += [exe.parent / 'build_info.json',
                   exe.parents[1] / 'Resources' / 'build_info.json']
    for c in candidates:
        try:
            if c.is_file():
                with open(c) as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    return info
        except (OSError, ValueError, IndexError):
            continue
    return None


def GetBuildLabel():
    """Short "which build is this?" tag, e.g. ``dev-c0cf1e6 · 2026-08-30``.

    Empty string for a plain version number, which is reserved for the two
    cases where the version alone is unambiguous:

    * **running from source** - the developer already knows what they checked
      out, and titles should look exactly as they always have;
    * **a stable release** (built from a ``v*`` tag, ``channel == 'stable'``).

    Everything else - a ``test-*`` pre-release tag, a manual CI dispatch, or a
    local ``create_unsigned_dmg.sh`` build - gets the commit and build date, so
    a tester can say which build they are looking at without digging.
    """
    info = _read_build_info()
    if info is None:
        return ''
    channel = str(info.get('channel') or 'stable')
    if channel == 'stable':
        return ''
    commit = str(info.get('git_commit') or '')[:7]
    built = str(info.get('built') or '')[:10]          # YYYY-MM-DD
    prefix = 'dev' if channel == 'dev' else str(info.get('tag') or 'test')
    label = prefix if not commit else f'{prefix}-{commit}'
    return f'{label} · {built}' if built else label


def TitleSuffix(label=None):
    """The build label formatted for a window title: ``' [dev-… · date]'``.

    Empty for source runs and stable releases. Bracketed, and using '·' rather
    than the ' - ' that separates the other title fields, so a dev/test build
    reads as an annotation instead of just another field.

    ``label`` may be passed by a caller that already has one (the main window
    keeps it in ``Config``), so both titles are formatted in exactly one place.
    """
    if label is None:
        label = GetBuildLabel()
    return ' [' + label + ']' if label else ''

