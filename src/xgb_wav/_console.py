"""控制台编码兜底。

Windows 默认代码页（英文系统 cp437 / 部分系统 cp936）无法编码中文，
`print("模型就绪")` 会直接抛 UnicodeEncodeError 导致程序崩掉。
这里做两件事：
  1. 尝试把控制台代码页切到 UTF-8（Win10+ 支持，中文显示正常）；
  2. 把 stdout/stderr 的编码设为 UTF-8 且 errors="replace"，
     即使切换失败也不会因为一个字符而中断程序。
"""

from __future__ import annotations

import os
import sys


def setup_console() -> None:
    """在程序入口调用一次即可；非 Windows 平台什么都不做。"""
    if os.name != "nt":
        # Linux/macOS 终端一般已是 UTF-8；若被重定向到 ascii 也能兜住
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            except Exception:
                pass
        return

    try:  # 切控制台代码页到 65001(UTF-8)
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetConsoleOutputCP(65001)
        k32.SetConsoleCP(65001)
    except Exception:
        pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
