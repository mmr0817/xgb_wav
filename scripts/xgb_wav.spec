# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（Windows / macOS / Linux 通用）。

用法（在仓库根目录）：
    pyinstaller --noconfirm --clean --distpath dist --workpath build scripts/xgb_wav.spec
输出：
    dist/xgb_wav/xgb_wav.exe

注意：PyInstaller 执行 spec 时会先把工作目录切到 spec 所在目录（scripts/），
所以这里所有路径都必须基于 SPECPATH 计算，不能写相对仓库根的 "scripts/xxx"
（否则会变成 scripts/scripts/xxx 而报 file not found）。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()   # .../<repo>/scripts
ROOT = SPEC_DIR.parent                # .../<repo>
ENTRY = SPEC_DIR / "gui_entry.py"
SRC = ROOT / "src"

hiddenimports = [
    "xgboost", "sklearn", "sklearn.ensemble", "sklearn.preprocessing",
    "sklearn.utils._typedefs", "scipy.signal", "soundfile", "librosa",
    "torchaudio", "torchaudio.models.wav2vec2", "torchaudio.pipelines",
    "torchaudio.functional", "torchaudio.models.wav2vec2.utils",
    "pandas", "joblib", "yaml",
]
hiddenimports += collect_submodules("librosa")
hiddenimports += collect_submodules("torchaudio.models")

datas = collect_data_files("librosa")

# 如需把模型一起打进 exe，取消下面两行注释（体积会更大）
# datas += [(str(ROOT / "models"), "models")]
# 默认：models/ 放在 exe 同级目录，由 GUI 自动探测

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
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
