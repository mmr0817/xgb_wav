# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（Windows）。

用法（在仓库根目录）：
    pyinstaller --noconfirm --clean scripts/xgb_wav.spec
输出：
    dist/xgb_wav/xgb_wav.exe
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = [
    "xgboost", "sklearn", "sklearn.ensemble", "sklearn.preprocessing",
    "sklearn.utils._typedefs", "scipy.signal", "soundfile", "librosa",
    "torchaudio", "torchaudio.models.wav2vec2", "torchaudio.pipelines",
    "torchaudio.functional", "pandas", "joblib", "yaml",
]
hiddenimports += collect_submodules("librosa")
hiddenimports += collect_submodules("torchaudio.models")

datas = collect_data_files("librosa")

# 如需把模型一起打进 exe，取消下面两行注释（体积会更大）
# datas += [("models", "models")]
# 或者保持默认：models/ 放在 exe 同级目录，由 GUI 自动探测

a = Analysis(
    ["scripts/gui_entry.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook", "tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="xgb_wav",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # 图形界面，不弹控制台
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="xgb_wav",
)
