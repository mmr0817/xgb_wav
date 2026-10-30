"""用户设置的读写（GUI 与命令行共用）。

settings.json 放在「程序目录」：
  - 打包成 exe：exe 同级目录（便于随程序一起分发/备份）
  - 源码运行：仓库根目录
目前保存：判定阈值 threshold（GUI 里可实时调整）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

SETTINGS_NAME = "settings.json"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def settings_path() -> Path:
    return app_dir() / SETTINGS_NAME


def load_settings() -> Dict[str, object]:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: Dict[str, object]) -> Path:
    path = settings_path()
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return path


def get_threshold(default: float) -> float:
    try:
        return float(load_settings().get("threshold", default))
    except (TypeError, ValueError):
        return float(default)


def set_threshold(value: float) -> Path:
    data = load_settings()
    data["threshold"] = round(float(value), 6)
    return save_settings(data)
