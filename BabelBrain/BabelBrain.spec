# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all,  collect_submodules,collect_data_files
from PyInstaller import compat
from os import listdir
import platform



tmp_ret = collect_all('BabelViscoFDTD')
binaries = tmp_ret[1]
datas=tmp_ret[0]
tmp_ret = collect_all('trimesh')
hiddenimports=tmp_ret[2]
datas+=tmp_ret[0]

tmp_ret = collect_all('itk')
hiddenimports+=tmp_ret[2]

hiddenimports+=['vtkmodules','vtkmodules.all','vtkmodules.qt.QVTKRenderWindowInteractor','vtkmodules.util','vtkmodules.util.numpy_support','vtkmodules.numpy_interface', 'vtkmodules.numpy_interface.dataset_adapter']

itk_datas = collect_data_files('itk', include_py_files=True)
datas+= [x for x in itk_datas if '__pycache__' not in x[0]]

tmp_ret2 = collect_all('pydicom')
hiddenimports+=tmp_ret2[2]

datas+=[('Babel_H317/default.yaml','./Babel_H317'),
        ('Babel_H246/default.yaml','./Babel_H246'),
        ('Babel_CTX500/default.yaml','./Babel_CTX500'),
        ('default.yaml','./'),
        ('Babel_H317/form.ui','./Babel_H317'),
        ('Babel_H246/form.ui','./Babel_H246'),
        ('Babel_CTX500/form.ui','./Babel_CTX500'),
        ('form.ui','./'),
        ('../TranscranialModeling/H-317 XYZ Coordinates_revB update 1.18.22.csv','./TranscranialModeling'),
        ('Babel_Thermal_SingleFocus/form.ui','./Babel_Thermal_SingleFocus')]


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
    pathex=['./'],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BabelBrain',
)
app = BUNDLE(
    coll,
    name='BabelBrain.app',
    icon='./Proteus-Alciato-logo.png',
)
