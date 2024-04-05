# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all,  collect_submodules,collect_data_files
from PyInstaller import compat
from os import listdir
import platform
import glob
import os

commonDatas=[('Babel_H317/default.yaml','./Babel_H317'),
        ('Babel_H246/default.yaml','./Babel_H246'),
        ('Babel_CTX500/default.yaml','./Babel_CTX500'),
        ('Babel_SingleTx/default.yaml','./Babel_SingleTx'),
        ('Babel_SingleTx/defaultBSonix.yaml','./Babel_SingleTx'),
        ('Babel_REMOPD/default.yaml','./Babel_REMOPD'),
        ('Babel_I12378/default.yaml','./Babel_I12378'),
        ('Babel_ATAC/default.yaml','./Babel_ATAC'),
        ('default.yaml','./'),
        ('version.txt','./'),
        ('icons8-hourglass.gif','./'),
        ('form.ui','./'),
        ('Babel_H317/form.ui','./Babel_H317'),
        ('Babel_H246/form.ui','./Babel_H246'),
        ('Babel_CTX500/form.ui','./Babel_CTX500'),
        ('Babel_SingleTx/form.ui','./Babel_SingleTx'),
        ('Babel_SingleTx/formBx.ui','./Babel_SingleTx'),
        ('Babel_REMOPD/form.ui','./Babel_REMOPD'),
        ('Babel_I12378/form.ui','./Babel_I12378'),
        ('Babel_ATAC/form.ui','./Babel_ATAC'),
        ('GUIComponents/scrollbars.ui','./GUIComponents'),
        ('../TranscranialModeling/H-317 XYZ Coordinates_revB update 1.18.22.csv','./TranscranialModeling'),
        ('../TranscranialModeling/I12378.csv','./TranscranialModeling'),
        ('../TranscranialModeling/ATACArray.csv','./TranscranialModeling'),
        ('../TranscranialModeling/MapPichardo.h5','./TranscranialModeling'),
        ('../TranscranialModeling/WebbHU_SoS.csv','./TranscranialModeling'),
        ('../TranscranialModeling/WebbHU_Att.csv','./TranscranialModeling'),
        ('../TranscranialModeling/ct-calibration-low-dose-30-March-2023-v1.h5','./TranscranialModeling'),
        ('../TranscranialModeling/REMOPD_ElementPosition.mat','./TranscranialModeling'),
        ('Babel_Thermal/form.ui','./Babel_Thermal')]

with open('version.txt','r') as f:
    version=f.readlines()[0].strip()

if 'Darwin' in platform.system(): #for Mac
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

    datas+=commonDatas


    for l in glob.glob('ExternalBin'+os.sep+'**',recursive=True):
        if os.path.isfile(l):
            if 'Darwin' in platform.system() and 'mac' in l:
                binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
            elif 'Linux' in platform.system() and 'linux' in l:
                binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
            elif 'Windows' in platform.system() and 'windows' in l:
                binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
            elif  '.txt' in l:
                binaries+=[(l,'.'+os.sep+os.path.dirname(l))]
    print('binaries',binaries)
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
        hooksconfig={
            "matplotlib": {
                "backends": ["TkAgg","Qt5Agg","AGG","PDF","PS","SVG","PGF"],  # collect multiple backends
            },
        },
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
        version=version,
        bundle_identifier='com.ucalgary.babelbrain',
        icon='./Proteus-Alciato-logo.png',
    )
elif 'Windows' in platform.system(): #for Windows
    datas = []
    binaries = []
    hiddenimports = []

    missing_package_info = ['BabelViscoFDTD',\
                        'cupy','cupyx','cupy_backends',
                        'fastrlock',\
                        'skimage',\
                        'pyopencl',\
                        ]

    for mp in missing_package_info:
        modinfo = collect_all(mp)
        if modinfo is not []:
            print(mp)
        datas += modinfo[0]
        binaries += modinfo[1]
        hiddenimports += modinfo[2]
        
    binaries+=[(os.environ['CONDA_PREFIX']+'/Library/bin/nvrtc-builtins64_117.dll','./')]
    datas+=commonDatas
    datas+=[('ExternalBin/elastix/run_win.bat','./ExternalBin/elastix'),
            ('SelFiles/form.ui','./SelFiles'),
            ('GPUVoxelize/helper_math.h','./GPUVoxelize'),
            ('../Profiles/Thermal_Profile_1.yaml','./Profiles'),
            ('../Profiles/Thermal_Profile_2.yaml','./Profiles'),
            ('../Profiles/Thermal_Profile_3.yaml','./Profiles'),
            ('../Profiles/MultiFocus_Profile1.yaml','./Profiles'),
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
else: #for Linux
    datas = []
    binaries = []
    hiddenimports = []

    missing_package_info = ['BabelViscoFDTD',\
                        'cupy','cupyx','cupy_backends',
                        'fastrlock',\
                        'skimage',\
                        'pyopencl',\
                        ]

    for mp in missing_package_info:
        modinfo = collect_all(mp)
        if modinfo is not []:
            print(mp)
        datas += modinfo[0]
        binaries += modinfo[1]
        hiddenimports += modinfo[2]
        
    binaries+=[(os.environ['CONDA_PREFIX']+'/lib64/libnvrtc-builtins.so','./')]
    datas+=commonDatas
    datas+=[('SelFiles/form.ui','./SelFiles'),
            ('GPUVoxelize/helper_math.h','./GPUVoxelize'),
            ('../Profiles/Profile_1.yaml','./Profiles'),
            ('../Profiles/Profile_2.yaml','./Profiles'),
            ('../Profiles/Profile_3.yaml','./Profiles'),
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
