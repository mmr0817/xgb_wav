"""数据发现、标签与文件名解析工具。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

PathLike = Union[str, Path]

# 训练时使用的目录 -> 标签（推理时不需要，保留用于训练脚本）
LABEL_MAP: Dict[str, int] = {"good": 0, "bad": 1}
LABEL_CN = {"good": "好", "bad": "坏", "critical": "临界", "transient": "瞬态噪声"}

_PREFIX_RE = re.compile(r"^(OK[_\-]|NG[_\-]|GOOD[_\-]|BAD[_\-]|良品[_\-]?|不良品[_\-]?|坏[_\-]?|好[_\-]?)+")
_STAMP_RE = re.compile(r"(\d+)[-_](\d{1,2})[-:：](\d{2})[-:：](\d{2})(?:\.(\d+))?")
_CHANNEL_RE = re.compile(r"-(P|F)\.wav$", re.IGNORECASE)


def channel_of(path: PathLike) -> str:
    """采集通道：'P' / 'F' / 'none'（来自文件名后缀）。"""
    m = _CHANNEL_RE.search(Path(path).name)
    return m.group(1).upper() if m else "none"


def batch_of(path: PathLike) -> str:
    """设备/批次键：文件名中时间戳之前的前缀（无则用整条文件名）。"""
    name = _PREFIX_RE.sub("", Path(path).stem)
    m = _STAMP_RE.search(name)
    prefix = name[: m.start()].strip("-_ ") if m else ""
    return f"dev:{prefix}" if prefix else f"rec:{name}"


def discover_wavs(root: PathLike, extensions: Sequence[str] = (".wav", ".WAV")) -> List[Path]:
    """递归收集 wav 文件（排序保证可复现）。"""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    if root.is_file():
        return [root.resolve()]
    return sorted(p.resolve() for p in root.rglob("*") if p.suffix in extensions)


def discover_labelled(root: PathLike) -> Tuple[List[Path], List[int]]:
    """从 {good,bad} 子目录收集样本与标签（训练用）。"""
    root = Path(root).expanduser()
    paths: List[Path] = []
    labels: List[int] = []
    for name, label in LABEL_MAP.items():
        folder = root / name
        if not folder.exists():
            folder = root / LABEL_CN.get(name, name)
        if not folder.exists():
            print(f"[Warning] 未找到目录: {folder}")
            continue
        files = discover_wavs(folder)
        paths.extend(files)
        labels.extend([label] * len(files))
        print(f"  {name:<6}({label}): {len(files)} 个文件")
    if not paths:
        raise FileNotFoundError(f"{root} 下没有找到 good/bad 的 wav")
    return paths, labels
