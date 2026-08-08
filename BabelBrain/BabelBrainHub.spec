# -*- mode: python ; coding: utf-8 -*-
'''
PyInstaller spec for the BabelBrain **Hub** launcher (hub.py).

Run from the BabelBrain/ directory, exactly like BabelBrain.spec:

    pyinstaller BabelBrainHub.spec --noconfirm --clean

The Hub is intentionally small: PySide6 + PyYAML + the Hub package. It is the
user-facing app (built as ``BabelBrain.app`` on macOS / ``BabelBrain.exe`` on
Windows) that lets users pick and swap BabelBrain versions.

The read-only "bundled" BabelBrain version is NOT embedded by this spec. CI
copies the freshly built (and, on macOS, signed + notarized) version into the
Hub after building it, at a deterministic, executable-relative location the Hub
resolves at runtime (see paths.bundled_version_dir):

  * macOS  : BabelBrain.app/Contents/Resources/bundled/BabelBrain.app
  * Windows: <hub dir>/bundled/  (BabelBrain.exe and support files)

Doing the copy in CI (with ditto on macOS) keeps the nested app's code
signature and notarization ticket intact — PyInstaller's data collection would
not preserve them.
'''
import platform

from PyInstaller.utils.hooks import collect_all, collect_submodules

is_mac = "Darwin" in platform.system()

# The Hub carries its own version, independent of any BabelBrain version.
hub_version = "1.0.0"

datas = []
binaries = []

# hub.py imports the Hub package dynamically (Hub.cli -> ui/installer/...).
hiddenimports = collect_submodules("Hub") + [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "yaml",
]

# Bundle certifi's CA bundle so the Hub can verify TLS for the manifest fetch
# and version downloads. Without this a frozen app has no CA path and every
# HTTPS request fails certificate verification (see Hub/netutil.py).
_cf_datas, _cf_bins, _cf_hidden = collect_all("certifi")
datas += _cf_datas
binaries += _cf_bins
hiddenimports += _cf_hidden + ["certifi"]

block_cipher = None

a = Analysis(
    ["hub.py"],
    pathex=["./"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep the Hub light: exclude the heavy scientific stack only the real
    # BabelBrain versions need.
    excludes=[
        "SimpleITK", "itk", "vtk", "vtkmodules", "nibabel", "trimesh",
        "BabelViscoFDTD", "cupy", "pyopencl", "mlx", "scipy", "skimage",
        "matplotlib", "pandas",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BabelBrain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=not is_mac,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    entitlements_file=None,
    icon=None if is_mac else ["Proteus-Alciato-logo.ico"],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=is_mac,
    upx_exclude=[],
    name="BabelBrain",
)

if is_mac:
    app = BUNDLE(
        coll,
        name="BabelBrain.app",
        # Distinct identifier from the nested versions so Launch Services treats
        # the launcher and the versions it runs as separate apps.
        bundle_identifier="com.ucalgary.babelbrain.hub",
        version=hub_version,
        icon="./Proteus-Alciato-logo.png",
    )
