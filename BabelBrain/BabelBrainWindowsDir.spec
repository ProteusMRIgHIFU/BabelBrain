# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files, collect_dynamic_libs
from PyInstaller import compat
from os import listdir
import platform
import glob
import os
datas = []
binaries = []
hiddenimports = []

missing_package_info = ['BabelViscoFDTD',\
                    'cupy','cupyx','cupy_backends',
                    'fastrlock',\
                    'skimage',\
                    ]

for mp in missing_package_info:
    modinfo = collect_all(mp)
    if modinfo is not []:
        print(mp)
    datas += modinfo[0]
    binaries += modinfo[1]
    hiddenimports += modinfo[2]
	
binaries+=[(os.environ['CONDA_PREFIX']+'/Library/bin/nvrtc-builtins64_117.dll','./')]

datas+=[('Babel_H317/default.yaml','./Babel_H317'),
        ('Babel_H246/default.yaml','./Babel_H246'),
        ('Babel_CTX500/default.yaml','./Babel_CTX500'),
        ('Babel_SingleTx/default.yaml','./Babel_SingleTx'),
        ('Babel_Thermal_SingleFocus/form.ui','./Babel_Thermal_SingleFocus'),
        ('Babel_H317/form.ui','./Babel_H317'),
        ('Babel_H246/form.ui','./Babel_H246'),
        ('Babel_CTX500/form.ui','./Babel_CTX500'),
        ('Babel_SingleTx/form.ui','./Babel_SingleTx'),
        ('ExternalBin/elastix/run_win.bat','./ExternalBin/elastix'),
        ('SelFiles/form.ui','./SelFiles'),
        ('GPUVoxelize/helper_math.h','./GPUVoxelize'),
        ('default.yaml','./'),
        ('version.txt','./'),
        ('form.ui','./'),
        ('../TranscranialModeling/H-317 XYZ Coordinates_revB update 1.18.22.csv','./TranscranialModeling'),
        ('../TranscranialModeling/MapPichardo.h5','./TranscranialModeling'),
        ('../ThermalProfiles/Profile_1.yaml','./ThermalProfiles'),
        ('../ThermalProfiles/Profile_2.yaml','./ThermalProfiles'),
        ('../ThermalProfiles/Profile_3.yaml','./ThermalProfiles'),
        ('../NeedleModel.stl','./'),
        ('../PlanningModels/Trajectory-20-60-F#1.stl','./PlanningModels'),
        ('../PlanningModels/Trajectory-30-70-F#1.stl','./PlanningModels'),
        ('../PlanningModels/Trajectory-50-90-F#1.stl','./PlanningModels')]

for l in glob.glob('ExternalBin'+os.sep+'**',recursive=True):
    if os.path.isfile(l):
        if 'Darwin' in platform.system() and 'mac' in l:
            binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
        elif 'Linux' in platform.system() and 'linux' in l:
            binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
        elif 'Windows' in platform.system() and 'windows' in l:
            print("Elastix for Windows")
            print(f"Elastix binaries: {[(l,'.'+os.sep+os.path.dirname(l))]}")
            binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
        elif  '.txt' in l:
            binaries+=[(l,'.'+os.sep+os.path.dirname(l))]

if 'Darwin' in platform.system() and 'arm64' not in platform.platform():
    hiddenimports+=['histoprint']
    libdir = compat.base_prefix + "/lib"
    mkllib = filter(lambda x : x.startswith('libmkl_'), listdir(libdir))
    if mkllib != []:
        print("MKL installed as part of numpy, importing that!")
        binaries+= map(lambda l: (libdir + "/" + l, './'), mkllib)
        print(binaries)
block_cipher = None
a = Analysis(
    ['BabelBrain.py'],
    pathex=['./','./..'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='BabelBrain',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    entitlements_file=None,
	icon=['Proteus-Alciato-logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BabelBrain',
)