"""
新威风格测量特征（自原仓库 src/features/xinwei_features.py vendor，逻辑保持原样）

包含：
  - 参考扫频对齐（互相关）
  - librosa 通用统计（MFCC/质心/带宽/rolloff/ZCR/RMS/contrast/chroma）
  - 1/3 倍频程带能量
  - 相对参考的 H(f) 带能量比
  - THD 近似（谐波能量比）
  - 指定时刻短时频段 power（Blackman-Harris 风格窗）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from scipy import signal as sps
except ImportError:
    sps = None


PathLike = Union[str, Path]


@dataclass
class XinWeiFeatureConfig:
    sample_rate: int = 44100  # 测量特征用较高采样率更接近新威
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    # 1/3 倍频程中心频率范围
    fmin: float = 100.0
    fmax: float = 10000.0
    # THD：相对基波（用扫频瞬时较难，改为全带谐波结构代理）
    n_harmonics: int = 5
    # 指定时刻（秒）上的短时频段 power
    probe_times: Tuple[float, ...] = (0.4, 0.8, 1.2, 1.6)
    probe_bands: Tuple[Tuple[float, float], ...] = (
        (2000, 3000),
        (5000, 6000),
        (6000, 7000),
        (7000, 8000),
        (9000, 10000),
    )
    probe_win_ms: float = 20.0
    align: bool = True
    # 额外窄带特征（每个带 3 维：对数带能量、带内时间变异、占全带能量比）
    # 留空 () 则不加；用于针对性捕捉某些频点的干扰/缺陷（如 5k/11k）
    extra_bands: Tuple[Tuple[float, float], ...] = ()


def load_mono(path: PathLike, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    path = str(path)
    if sf is not None:
        y, file_sr = sf.read(path, always_2d=True)
        y = y.mean(axis=1).astype(np.float32)
        file_sr = int(file_sr)
    elif librosa is not None:
        y, file_sr = librosa.load(path, sr=None, mono=True)
        y = y.astype(np.float32)
        file_sr = int(file_sr)
    else:
        raise ImportError("需要 soundfile 或 librosa 读取音频")
    if sr is not None and file_sr != sr:
        if librosa is None:
            raise ImportError("重采样需要 librosa")
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return y.astype(np.float32), file_sr


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def align_to_ref(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """FFT 互相关对齐 y 到 ref 时间轴（返回对齐后的 y）"""
    n = len(y) + len(ref) - 1
    nfft = _next_pow2(n)
    Y = np.fft.rfft(y, n=nfft)
    R = np.fft.rfft(ref, n=nfft)
    corr = np.fft.irfft(Y * np.conj(R), n=nfft)
    lag = int(np.argmax(np.abs(corr)))
    if lag > nfft // 2:
        lag = lag - nfft
    # lag > 0: y 相对 ref 超前，需要把 y 右移
    if lag > 0:
        y = np.pad(y, (lag, 0))[: len(y) + lag]
        y = y[: len(ref)] if len(y) > len(ref) else np.pad(y, (0, len(ref) - len(y)))
    elif lag < 0:
        y = y[-lag:]
        if len(y) < len(ref):
            y = np.pad(y, (0, len(ref) - len(y)))
        else:
            y = y[: len(ref)]
    else:
        if len(y) < len(ref):
            y = np.pad(y, (0, len(ref) - len(y)))
        else:
            y = y[: len(ref)]
    return y.astype(np.float32)


def third_octave_edges(fmin: float, fmax: float) -> np.ndarray:
    """生成 1/3 倍频程边界"""
    # 标准：中心频率 1000 * 2^(k/3)
    centers = []
    k = -20
    while True:
        fc = 1000.0 * (2 ** (k / 3.0))
        if fc > fmax * 1.1:
            break
        if fc >= fmin * 0.9:
            centers.append(fc)
        k += 1
    centers = np.array(centers, dtype=np.float64)
    # 边界 = 中心 * 2^(±1/6)
    edges = []
    for fc in centers:
        edges.append(fc * (2 ** (-1 / 6)))
    edges.append(centers[-1] * (2 ** (1 / 6)))
    return np.array(edges, dtype=np.float64)


def band_energies(power: np.ndarray, freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """power: [F], freqs: [F] -> energies [n_bands]"""
    out = []
    for i in range(len(edges) - 1):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out.append(float(power[m].mean()) if m.any() else 0.0)
    return np.asarray(out, dtype=np.float32)


def stft_power(y: np.ndarray, sr: int, n_fft: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    if librosa is None:
        raise ImportError("需要 librosa")
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S.astype(np.float32), freqs.astype(np.float32)


def approx_thd(power_mean: np.ndarray, freqs: np.ndarray, n_harm: int = 5) -> float:
    """
    粗略 THD 代理：在谱峰基频附近找 f0，累加 2f0..n*f0 能量 / 基波能量。
    扫频信号上仅为结构异常敏感特征，非严格电声 THD。
    """
    if power_mean.sum() <= 0:
        return 0.0
    # 忽略很低频
    valid = freqs > 80
    if not valid.any():
        return 0.0
    idx = np.argmax(power_mean * valid)
    f0 = float(freqs[idx])
    if f0 < 80:
        return 0.0
    def energy_at(f):
        # 取 ±8% 带宽
        bw = max(20.0, f * 0.08)
        m = (freqs >= f - bw) & (freqs <= f + bw)
        return float(power_mean[m].sum()) if m.any() else 0.0
    e1 = energy_at(f0) + 1e-12
    eh = sum(energy_at(f0 * k) for k in range(2, n_harm + 1))
    return float(np.sqrt(eh / e1))


def probe_band_powers(
    y: np.ndarray,
    sr: int,
    times: Sequence[float],
    bands: Sequence[Tuple[float, float]],
    win_ms: float = 20.0,
) -> np.ndarray:
    """指定时刻短时 FFT 频段能量，Blackman-Harris 近似用 blackmanharris 窗"""
    n_win = max(16, int(sr * win_ms / 1000.0))
    if sps is not None:
        window = sps.windows.blackmanharris(n_win).astype(np.float32)
    else:
        window = np.blackman(n_win).astype(np.float32)
    n_fft = _next_pow2(n_win)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    feats = []
    for t in times:
        center = int(t * sr)
        start = center - n_win // 2
        end = start + n_win
        if start < 0 or end > len(y):
            # 越界填 0
            feats.extend([0.0] * len(bands))
            continue
        seg = y[start:end] * window
        spec = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
        for lo, hi in bands:
            m = (freqs >= lo) & (freqs < hi)
            feats.append(float(spec[m].mean()) if m.any() else 0.0)
    return np.log1p(np.asarray(feats, dtype=np.float32))


def librosa_stats(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))],
        [float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))],
        [float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))],
        [float(np.mean(librosa.feature.zero_crossing_rate(y)))],
        [float(np.mean(librosa.feature.rms(y=y)))],
        librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1),
        librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
    ]
    return np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in feat])


def extract_xinwei_features(
    path: PathLike,
    ref_wav: Optional[np.ndarray] = None,
    ref_sr: Optional[int] = None,
    cfg: Optional[XinWeiFeatureConfig] = None,
) -> np.ndarray:
    """
    提取单条音频的新威风格特征向量。
    ref_wav: 原始扫频源（与 cfg.sample_rate 一致更佳）
    """
    cfg = cfg or XinWeiFeatureConfig()
    if librosa is None:
        raise ImportError("需要 librosa")

    y, sr = load_mono(path, sr=cfg.sample_rate)

    # 对齐到参考
    if ref_wav is not None and cfg.align:
        ref = ref_wav
        if ref_sr is not None and ref_sr != cfg.sample_rate:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=ref_sr, target_sr=cfg.sample_rate)
        # 长度对齐
        if len(ref) < len(y):
            ref = np.pad(ref, (0, len(y) - len(ref)))
        else:
            ref = ref[: len(y)]
        y_aligned = align_to_ref(y, ref)
    else:
        ref = None
        y_aligned = y

    # 1) librosa 统计
    f_lib = librosa_stats(y_aligned, cfg.sample_rate, cfg.n_mfcc)

    # 2) 平均功率谱
    S, freqs = stft_power(y_aligned, cfg.sample_rate, cfg.n_fft, cfg.hop_length)
    p_mean = S.mean(axis=1)

    # 3) 1/3 倍频程
    edges = third_octave_edges(cfg.fmin, cfg.fmax)
    bands = band_energies(p_mean, freqs, edges)
    bands_log = np.log1p(bands)

    # 4) H(f) 带比（相对参考）
    if ref is not None:
        S_ref, _ = stft_power(ref.astype(np.float32), cfg.sample_rate, cfg.n_fft, cfg.hop_length)
        p_ref = S_ref.mean(axis=1) + 1e-12
        # 插值到同一 freqs — 已同参数
        h_ratio = np.log((p_mean + 1e-12) / p_ref)
        # 降维：按 1/3 倍频程聚合 H
        h_bands = band_energies(np.exp(h_ratio) * p_ref, freqs, edges)  # 等价于待测带能
        h_bands = np.log((h_bands + 1e-12) / (band_energies(p_ref, freqs, edges) + 1e-12))
        residual = np.array([float(np.mean(np.abs(p_mean - p_ref)) / (np.mean(p_ref) + 1e-12))], dtype=np.float32)
    else:
        h_bands = np.zeros_like(bands_log)
        residual = np.zeros(1, dtype=np.float32)

    # 5) THD 代理
    thd = np.array([approx_thd(p_mean, freqs, cfg.n_harmonics)], dtype=np.float32)

    # 6) 指定时刻频段 power
    probes = probe_band_powers(
        y_aligned, cfg.sample_rate, cfg.probe_times, cfg.probe_bands, cfg.probe_win_ms
    )

    # 7) 可选：额外窄带特征（每带 3 维）
    extra = np.zeros(0, dtype=np.float32)
    if cfg.extra_bands:
        total = float(p_mean.sum()) + 1e-12
        vals = []
        for lo, hi in cfg.extra_bands:
            m = (freqs >= lo) & (freqs < hi)
            if not m.any():
                vals.extend([0.0, 0.0, 0.0])
                continue
            e = float(p_mean[m].sum())
            band_over_time = S[m, :].sum(axis=0)
            cv = float(band_over_time.std() / (band_over_time.mean() + 1e-12))
            vals.extend([float(np.log1p(e)), float(np.log1p(cv)), float(np.log1p(e / total))])
        extra = np.asarray(vals, dtype=np.float32)

    feat = np.concatenate(
        [f_lib, bands_log, h_bands.astype(np.float32), residual, thd, probes, extra]
    )
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32)


def load_reference(path: PathLike, sr: int) -> Tuple[np.ndarray, int]:
    return load_mono(path, sr=sr)