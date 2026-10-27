"""特征提取（XGB 分支），带缓存。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .features import XinWeiFeatureConfig, extract_xinwei_features, load_reference


def load_ref(ref_path: Optional[Path], cfg: XinWeiFeatureConfig):
    """加载参考扫频；不存在时返回 (None, None)，对齐类特征退化为 0。"""
    if ref_path is None:
        return None, None
    rp = Path(ref_path).expanduser()
    if not rp.exists():
        print(f"[Warn] 参考音频不存在，跳过对齐类特征: {rp}")
        return None, None
    wav, sr = load_reference(rp, cfg.sample_rate)
    return wav, sr


def extract_features(
    paths: Sequence[Path],
    ref_wav: Optional[np.ndarray],
    ref_sr: Optional[int],
    cfg: XinWeiFeatureConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """逐条提特征；单条失败会跳过并记录索引。"""
    feats, ok_paths, ok_idx = [], [], []
    expected = None
    for i, p in enumerate(paths):
        try:
            f = extract_xinwei_features(p, ref_wav=ref_wav, ref_sr=ref_sr, cfg=cfg)
        except Exception as e:
            print(f"[Skip] {p}: {e}")
            continue
        f = np.asarray(f, dtype=np.float32).ravel()
        if expected is None:
            expected = f.shape[0]
        elif f.shape[0] != expected:
            print(f"[Skip] {p}: 特征维度 {f.shape[0]} != {expected}")
            continue
        feats.append(f)
        ok_paths.append(str(p))
        ok_idx.append(i)
        if verbose and (i + 1) % 200 == 0:
            print(f"  特征进度 {i + 1}/{len(paths)}")
    if not feats:
        raise RuntimeError("没有任何样本成功提取特征")
    return np.stack(feats), ok_paths, ok_idx


def _cfg_stamp(cfg: XinWeiFeatureConfig) -> str:
    return json.dumps(cfg.__dict__, sort_keys=True, ensure_ascii=False, default=list)


class FeatureExtractor:
    """封装「参考扫频 + 特征配置」，便于推理与训练共用。"""

    def __init__(self, feat_cfg: XinWeiFeatureConfig, ref_path: Optional[Path] = None):
        self.cfg = feat_cfg
        self.ref_path = Path(ref_path).expanduser() if ref_path else None
        self.ref_wav, self.ref_sr = load_ref(self.ref_path, feat_cfg)

    def transform(self, paths: Sequence[Path], verbose: bool = False) -> Tuple[np.ndarray, List[str], List[int]]:
        return extract_features(paths, self.ref_wav, self.ref_sr, self.cfg, verbose=verbose)

    # ---------- 训练用缓存 ----------
    def fit_cache(self, cache_path: Path, paths: Sequence[Path], labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """有缓存且匹配则复用，否则重新提取并写盘。"""
        cache_path = Path(cache_path)
        stamp = _cfg_stamp(self.cfg)
        ref_stamp = "" if self.ref_path is None else str(self.ref_path)
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            cached_paths = [str(p) for p in z["paths"]]
            if ("cfg_stamp" in z.files and str(z["cfg_stamp"]) == stamp
                    and "ref_stamp" in z.files and str(z["ref_stamp"]) == ref_stamp
                    and cached_paths == [str(p) for p in paths]):
                print(f"加载特征缓存 {cache_path} {z['X'].shape}")
                return z["X"], z["y"], cached_paths
            print("[Info] 特征缓存失效，重新提取")
        X, ok_paths, ok_idx = self.transform(paths, verbose=True)
        y = np.asarray([int(labels[i]) for i in ok_idx], dtype=np.int64)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, y=y, paths=np.array(ok_paths, dtype=object),
                 cfg_stamp=np.array(stamp), ref_stamp=np.array(ref_stamp))
        print(f"已保存特征缓存 {X.shape} -> {cache_path}")
        return X, y, ok_paths
