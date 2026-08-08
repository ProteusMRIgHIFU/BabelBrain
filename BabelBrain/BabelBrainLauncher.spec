# -*- mode: python ; coding: utf-8 -*-
'''
PyInstaller spec for **BabelBrain.app** — the main app users double-click
(babelbrain_launcher.py).

Run from the BabelBrain/ directory:

    pyinstaller BabelBrainLauncher.spec --noconfirm --clean

This is a small thin launcher: it opens BabelBrain directly by running the
currently-selected version from the versions store (no picker). Like the Version
Selector it embeds no BabelBrain version; the installer seeds a default version
into the store. Kept structurally identical to BabelBrainHub.spec so both apps
stay in lock-step.
'''
import platform

from PyInstaller.utils.hooks import collect_all, collect_submodules

is_mac = "Darwin" in platform.system()

hub_version = "1.0.0"

datas = []
binaries = []

hiddenimports = collect_submodules("Hub") + [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "yaml",
]

# certifi is pulled in for parity with the selector (Hub.netutil imports it);
# the launcher itself does not fetch over HTTPS, but bundling it avoids a
# missing-module build warning and keeps the two apps identical.
_cf_datas, _cf_bins, _cf_hidden = collect_all("certifi")
datas += _cf_datas
binaries += _cf_bins
hiddenimports += _cf_hidden + ["certifi"]

block_cipher = None

a = Analysis(
    ["babelbrain_launcher.py"],
    pathex=["./"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
        bundle_identifier="com.ucalgary.babelbrain",
        version=hub_version,
        icon="./Proteus-Alciato-logo.png",
    )
