# -*- mode: python ; coding: utf-8 -*-
'''
PyInstaller spec for the **BabelBrain Uninstaller** (uninstaller.py).

Run from the BabelBrain/ directory:

    pyinstaller BabelBrainUninstaller.spec --noconfirm --clean

Builds BabelBrain-Uninstaller.app, the macOS-only app that removes BabelBrain,
every installed version and (optionally) the user's settings. macOS offers no
uninstall hook, so this has to be a real app the PKG installs alongside the
other two; Windows gets the same behaviour from the Inno uninstaller, which
calls the Version Selector with --purge-user-data.

Same shape as BabelBrainHub.spec: tiny (PySide6 + PyYAML + the Hub package),
no BabelBrain version embedded. certifi is not needed — the uninstaller makes
no network calls.
'''
import platform

from PyInstaller.utils.hooks import collect_submodules

is_mac = "Darwin" in platform.system()

# The launcher apps carry their own version, independent of any BabelBrain version.
hub_version = "1.0.0"

# uninstaller.py imports the Hub package dynamically (Hub.cli -> uninstall_ui/...).
hiddenimports = collect_submodules("Hub") + [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "yaml",
]

block_cipher = None

a = Analysis(
    ["uninstaller.py"],
    pathex=["./"],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep it light: exclude the heavy scientific stack only the real BabelBrain
    # versions need.
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
    name="BabelBrain-Uninstaller",
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
    name="BabelBrain-Uninstaller",
)

if is_mac:
    app = BUNDLE(
        coll,
        name="BabelBrain-Uninstaller.app",
        bundle_identifier="com.ucalgary.babelbrain.uninstaller",
        version=hub_version,
        icon="./Proteus-Alciato-logo.png",
    )
