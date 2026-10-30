"""xgb_wav 桌面应用入口（GUI）。

- 源码运行：在应用目录执行  python main.py
- 打包后：这就是 xgb_wav.exe 的入口（PyInstaller 从本文件启动）
"""

from __future__ import annotations

import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    # 源码运行时把 src/ 加入搜索路径，免安装
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xgb_wav._console import setup_console  # noqa: E402
from xgb_wav.perf import apply_low_power  # noqa: E402
from xgb_wav.settings import load_settings  # noqa: E402

if __name__ == "__main__":
    setup_console()
    # 省电模式必须在导入 torch/gui 之前应用（线程数只在进程启动时决定）
    apply_low_power(bool(load_settings().get("low_power", False)))

    from xgb_wav.gui import main as gui_main  # noqa: E402

    gui_main()
