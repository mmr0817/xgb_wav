"""性能模式（低功耗 / 省电）。

用途：在工作机上长时间跑批时，把推理速度**降到约 1/4**，减少 CPU 占用与发热：

  · 强制 CPU（不用 GPU）
  · torch / OpenMP / MKL 线程数压到 1
  · wav2vec2 批大小压到 1~4（更细粒度，占用更平滑）
  · 每批之间主动 sleep（low_power_sleep 秒，默认 0.30）

开关优先级：命令行 --low-power > 环境变量 XGB_WAV_LOW_POWER=1 > settings.json 的 "low_power"
"""

from __future__ import annotations

import os
import time

ENV_FLAG = "XGB_WAV_LOW_POWER"
DEFAULT_SLEEP = 0.06


def is_low_power() -> bool:
    val = os.environ.get(ENV_FLAG, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def enable_low_power(sleep: float = DEFAULT_SLEEP) -> None:
    """开启低功耗：必须在 import torch 之前设置线程环境变量，效果最好。"""
    os.environ[ENV_FLAG] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("XGB_WAV_LOW_POWER_SLEEP", str(sleep))
    try:  # torch 若已导入，直接改运行时线程数
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def disable_low_power() -> None:
    os.environ.pop(ENV_FLAG, None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.pop(key, None)
    os.environ.pop("XGB_WAV_LOW_POWER_SLEEP", None)


def sleep_seconds() -> float:
    try:
        return float(os.environ.get("XGB_WAV_LOW_POWER_SLEEP", DEFAULT_SLEEP))
    except (TypeError, ValueError):
        return DEFAULT_SLEEP


def apply_low_power(flag: bool) -> None:
    """按开关设置环境变量；由各入口在启动时调用一次。"""
    if flag:
        enable_low_power()
    else:
        disable_low_power()


def throttle(times: int = 1) -> None:
    """在批与批之间主动等待（只在低功耗模式生效）。"""
    if not is_low_power() or times <= 0:
        return
    time.sleep(sleep_seconds() * times)
