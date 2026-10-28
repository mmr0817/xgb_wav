#!/usr/bin/env python
"""环境自检：直接在 Windows 上双击/命令行运行 `python doctor.py`。

会检查并打印：Python 版本/位数、控制台编码、依赖是否齐全、tkinter 是否可用、
模型文件是否完整，最后用 `models/ref_sweep.wav` 跑一次真实推理。
（排查“命令行一执行就出错”时，把输出或生成的 `doctor_report.txt` 发出来即可。）
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from xgb_wav._console import setup_console  # noqa: E402

OUT: list[str] = []
FAILED: list[str] = []


def say(text: str = "") -> None:
    print(text)
    OUT.append(text)


def section(title: str) -> None:
    say("")
    say("=" * 64)
    say(title)
    say("=" * 64)


def main() -> int:
    setup_console()
    say("xgb_wav 环境自检")
    say(f"仓库目录: {REPO}")

    # ---------- 1. 系统 / Python ----------
    section("[1] 系统与 Python")
    say(f"平台      : {platform.platform()}")
    say(f"机器      : {platform.machine()}")
    say(f"Python    : {sys.version.split()[0]}  ({'64-bit' if sys.maxsize > 2**32 else '32-bit'})")
    say(f"解释器    : {sys.executable}")
    say(f"工作目录  : {Path.cwd()}")
    say(f"标准输出编码: {getattr(sys.stdout, 'encoding', '?')}")
    say(f"控制台代码页: {os.device_encoding(1) if os.name == 'nt' else 'n/a'}")
    if sys.version_info < (3, 10):
        FAILED.append("Python 需要 3.10 及以上")
    if sys.maxsize <= 2**32:
        FAILED.append("需要 64 位 Python")

    # ---------- 2. 依赖 ----------
    section("[2] 依赖包")
    deps = ["numpy", "scipy", "sklearn", "xgboost", "librosa", "soundfile",
            "pandas", "joblib", "torch", "torchaudio", "yaml"]
    for name in deps:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            say(f"  OK    {name:<12} {ver}")
        except Exception as e:
            say(f"  FAIL  {name:<12} {type(e).__name__}: {e}")
            FAILED.append(f"缺少依赖 {name}（pip install -r requirements.txt）")

    # ---------- 3. GUI ----------
    section("[3] 图形界面 (tkinter)")
    try:
        import tkinter

        say(f"  OK    tkinter {tkinter.TkVersion}")
        if os.name != "nt":
            if not os.environ.get("DISPLAY"):
                say("  WARN  无 DISPLAY：无桌面环境下无法开窗口（用 test.py 做命令行推理）")
    except Exception as e:
        # 容器/精简环境没有 tkinter 属正常：命令行不受影响，只影响 GUI
        say(f"  WARN  tkinter 不可用: {e}")
        say("        （Docker/精简环境正常，命令行检测不受影响；"
            "要 GUI 请用 python.org 安装包或 Windows exe）")

    # ---------- 4. 模型文件 ----------
    section("[4] 模型文件 models/")
    need = {"detector.json": 0, "fusion.json": 0, "xgb_pipeline.joblib": 0,
            "w2v_ckpt.pt": 300 * 1024 * 1024, "ref_sweep.wav": 0}
    for fn, min_size in need.items():
        p = REPO / "models" / fn
        if not p.exists():
            say(f"  FAIL  {fn} 不存在")
            FAILED.append(f"models/{fn} 缺失")
        else:
            size = p.stat().st_size
            flag = "OK  " if size >= min_size else "WARN"
            say(f"  {flag}  {fn:<22} {size / 1024 / 1024:.1f} MB")
            if size < min_size:
                FAILED.append(f"models/{fn} 体积异常（可能没解压完整）")

    # ---------- 4b. xgboost 版本一致性（模型由 3.4.1 保存）----------
    try:
        import json as _json

        import joblib

        pack = joblib.load(REPO / "models" / "xgb_pipeline.joblib")
        mv = tuple(int(v) for v in _json.loads(
            pack["model"].get_booster().save_config())["version"])
        say(f"  模型版本 : xgboost {mv}")
        try:
            import xgboost

            lib = tuple(int(x) for x in xgboost.__version__.split("+")[0].split(".")[:3])
            say(f"  库版本   : xgboost {xgboost.__version__}")
            if lib < mv:
                say("  FAIL  xgboost 版本低于模型版本")
                FAILED.append(
                    f'xgboost 版本过低（{xgboost.__version__} < {mv}）：'
                    f'pip install "xgboost==3.4.1"（该版本要求 Python >= 3.12）')
        except ImportError:
            FAILED.append("xgboost 未安装：pip install -r requirements.txt")
    except Exception as e:
        say(f"  FAIL  加载 xgb_pipeline.joblib 失败: {type(e).__name__}: {e}")
        FAILED.append('xgboost 版本不匹配，先执行 pip install "xgboost==3.4.1"（需 Python 3.12+）')

    # ---------- 5. 端到端 ----------
    section("[5] 端到端推理自检（models/ref_sweep.wav）")
    try:
        from xgb_wav.pipeline import FusionDetector

        det = FusionDetector.from_dir(REPO / "models", device="auto")
        say(f"  模型加载成功，设备={det.device}，{det.rule.summary()}")
        df = det.predict_paths([REPO / "models" / "ref_sweep.wav"])
        say(f"  推理成功：{len(df)} 条")
        if len(df):
            r = df.iloc[0]
            say(f"  → {r['file']}: label={r['label']} score={r['score']:.4f} "
                f"xgb={r['xgb_proba']:.4f} w2v={r['w2v_proba']:.4f}")
            say("  （ref_sweep.wav 是合成参考扫频，判定结果没有业务含义，"
                "只用于验证「读音频-提特征-两路推理-融合」链路是否通）")
    except Exception:
        say("  推理失败，完整堆栈：")
        tb = traceback.format_exc()
        say(tb)
        FAILED.append("端到端推理失败（见上方堆栈）")

    # ---------- 结论 ----------
    section("结论")
    if FAILED:
        say(f"发现 {len(FAILED)} 个问题：")
        for f in FAILED:
            say(f"  - {f}")
    else:
        say("全部检查通过 ✅  可以运行：")
        say("  GUI:  run_gui.bat（或 python run_gui.py）")
        say("  CLI:  run_cli.bat（或 python test.py --input <文件夹>）")

    report = REPO / "doctor_report.txt"
    try:
        report.write_text("\n".join(OUT) + "\n", encoding="utf-8-sig")
        print(f"\n报告已写入: {report}")
    except Exception as e:
        print(f"[warn] 写报告失败: {e}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
