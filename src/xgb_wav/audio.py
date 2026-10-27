"""音频读取（16 kHz 单声道、峰值归一化、定长），与训练时一致。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

PathLike = Union[str, Path]


def load_audio(path: PathLike, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """读取音频并重采样为单声道 float32。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频不存在: {path}")
    if sf is not None:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    elif librosa is not None:
        wav, sr = librosa.load(str(path), sr=None, mono=True)
    else:  # pragma: no cover
        raise ImportError("需要 soundfile 或 librosa 读取音频")
    wav = np.asarray(wav, dtype=np.float32)
    if int(sr) != int(target_sr):
        if librosa is None:  # pragma: no cover
            raise ImportError("重采样需要 librosa")
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
        sr = target_sr
    return wav, int(sr)


def pad_or_truncate(wav: np.ndarray, length: int, mode: str = "constant") -> np.ndarray:
    """定长处理（与训练时一致：居中 pad / 居中截断）。"""
    n = len(wav)
    if n == length:
        return wav
    if n > length:
        start = (n - length) // 2
        return wav[start:start + length]
    pad_total = length - n
    left = pad_total // 2
    return np.pad(wav, (left, pad_total - left), mode=mode)


def load_for_w2v(path: PathLike, sample_rate: int = 16000, max_duration: float = 2.5) -> np.ndarray:
    """训练/推理统一的预处理：读数 -> 峰值归一化 -> 定长。"""
    wav, _ = load_audio(path, sample_rate)
    peak = float(np.max(np.abs(wav)) + 1e-8)
    wav = wav / peak
    return pad_or_truncate(wav, int(max_duration * sample_rate))
