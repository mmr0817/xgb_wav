"""端到端推理：文件/文件夹 -> (XGB 概率, wav2vec2 概率) -> 融合分 -> 标签 -> CSV。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

from .data import batch_of, channel_of, discover_wavs
from .featurize import FeatureExtractor
from .features import XinWeiFeatureConfig
from .fusion import FusionRule
from .model_xgb import predict_bad_proba
from .model_w2v import load_checkpoint, predict_proba as w2v_predict

PathLike = Union[str, Path]


@dataclass
class DetectorConfig:
    xgb_model: Path
    w2v_ckpt: Path
    fusion: Path
    ref_sweep: Optional[Path] = None
    feat_cfg: Optional[XinWeiFeatureConfig] = None
    w2v_sample_rate: int = 16000
    w2v_max_duration: float = 2.5
    w2v_hidden: int = 256
    batch_size: int = 32


class FusionDetector:
    """XGB + wav2vec2 融合判别器（生产用）。"""

    def __init__(self, cfg: DetectorConfig, device: str = "cpu"):
        self.cfg = cfg
        pack = joblib.load(cfg.xgb_model)
        self.xgb_model = pack["model"]
        self.scaler = pack["scaler"]
        self.feat_cfg = cfg.feat_cfg or XinWeiFeatureConfig(**pack.get("feat_cfg", {}))
        self.rule = FusionRule.from_json(cfg.fusion)
        self.extractor = FeatureExtractor(self.feat_cfg, cfg.ref_sweep)
        self.device = device
        self.w2v_model = load_checkpoint(
            cfg.w2v_ckpt, device=device, hidden_size=cfg.w2v_hidden,
            backbone=str(pack.get("w2v_backbone") or "") or None)
        self.feature_dim = int(pack.get("feature_dim", 0)) or None

    # ---------- 构造 ----------
    @classmethod
    def from_dir(cls, model_dir: PathLike, device: str = "auto") -> "FusionDetector":
        model_dir = Path(model_dir).expanduser()
        meta_path = model_dir / "detector.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"缺少 {meta_path}（模型目录应包含 detector.json）")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = DetectorConfig(
            xgb_model=model_dir / meta["xgb_model"],
            w2v_ckpt=model_dir / meta["w2v_ckpt"],
            fusion=model_dir / meta["fusion"],
            ref_sweep=(model_dir / meta["ref_sweep"]) if meta.get("ref_sweep") else None,
            w2v_sample_rate=int(meta.get("w2v_sample_rate", 16000)),
            w2v_max_duration=float(meta.get("w2v_max_duration", 2.5)),
            w2v_hidden=int(meta.get("w2v_hidden", 256)),
            batch_size=int(meta.get("batch_size", 32)),
        )
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        return cls(cfg, device=device)

    # ---------- 推理 ----------
    def predict_paths(self, paths: Sequence[PathLike], progress=None) -> pd.DataFrame:
        paths = [Path(p) for p in paths]
        if not paths:
            return pd.DataFrame(columns=["path", "label", "score", "xgb_proba", "w2v_proba"])

        t0 = time.time()
        X, ok_paths, ok_idx = self.extractor.transform(paths)
        if self.feature_dim and X.shape[1] != self.feature_dim:
            raise ValueError(f"特征维度不一致：模型 {self.feature_dim}，当前 {X.shape[1]}")
        xgb_p = predict_bad_proba(self.xgb_model, self.scaler.transform(X))
        ok_files = [paths[i] for i in ok_idx]
        w2v_p = w2v_predict(self.w2v_model, ok_files, device=self.device,
                            batch_size=self.cfg.batch_size,
                            sample_rate=self.cfg.w2v_sample_rate,
                            max_duration=self.cfg.w2v_max_duration)
        score = self.rule.score(xgb_p, w2v_p)
        pred = self.rule.decide(score)

        df = pd.DataFrame({
            "path": [str(p) for p in ok_files],
            "file": [p.name for p in ok_files],
            "label": [self.rule.label_of(v) for v in pred],
            "pred": pred.astype(int),
            "score": np.round(score, 6),
            "xgb_proba": np.round(xgb_p, 6),
            "w2v_proba": np.round(w2v_p, 6),
            "threshold": self.rule.threshold,
            "channel": [channel_of(p) for p in ok_files],
            "batch": [batch_of(p) for p in ok_files],
        })
        failed = set(range(len(paths))) - set(ok_idx)
        if failed:
            print(f"[Warn] {len(failed)} 个文件特征提取失败，未纳入结果: "
                  f"{[paths[i].name for i in list(failed)[:3]]}")
        df.attrs["elapsed"] = time.time() - t0
        df.attrs["skipped"] = len(failed)
        return df

    def predict_dir(self, folder: PathLike, recursive: bool = True) -> pd.DataFrame:
        folder = Path(folder).expanduser()
        paths = discover_wavs(folder)
        if not recursive:
            paths = [p for p in paths if p.parent == folder]
        return self.predict_paths(paths)

    # ---------- 落盘 ----------
    @staticmethod
    def save_csv(df: pd.DataFrame, out_csv: PathLike, summary: bool = True) -> Path:
        out_csv = Path(out_csv).expanduser()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        if summary and len(df):
            n_bad = int((df["pred"] == 1).sum())
            print(f"已写出 {out_csv}：共 {len(df)} 条，判定 bad {n_bad} 条，"
                  f"good {len(df) - n_bad} 条，阈值 {df['threshold'].iloc[0]:.4f}")
        return out_csv
