"""分数融合与「宁杀勿漏」判定规则。

规则与验证时一致：
  score = mean( (logit(p_i) - mu_i) / sd_i )
  bad   = score >= threshold
其中 mu/sd 由验证集统计、threshold = val 中 bad 的最低融合分 - margin。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

PathLike = Union[str, Path]


def logit(p: Union[float, np.ndarray], eps: float = 1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class FusionRule:
    members: Dict[str, Dict[str, float]] = field(default_factory=dict)
    threshold: float = 0.0
    margin: float = 0.0
    target_recall: float = 0.995
    label_map: Dict[str, str] = field(default_factory=lambda: {"0": "good", "1": "bad"})
    meta: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: PathLike) -> "FusionRule":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            members=data.get("members", {}),
            threshold=float(data.get("threshold", 0.0)),
            margin=float(data.get("margin", 0.0)),
            target_recall=float(data.get("target_recall", 0.995)),
            label_map=data.get("label_map", {"0": "good", "1": "bad"}),
            meta={k: v for k, v in data.items() if k not in {"members", "threshold"}},
        )

    def _z(self, name: str, proba: Union[float, np.ndarray]):
        m = self.members.get(name)
        if m is None:
            raise KeyError(f"融合配置缺少成员 {name}")
        return (logit(proba) - float(m["logit_mean"])) / float(m["logit_std"] + 1e-9)

    def score(self, xgb_proba, w2v_proba):
        """两个成员的概率 -> 融合分（标量或数组）。"""
        return (self._z("xgb", xgb_proba) + self._z("w2v", w2v_proba)) / 2.0

    def decide(self, score):
        return (np.asarray(score) >= self.threshold).astype(int)

    def label_of(self, pred) -> str:
        return self.label_map.get(str(int(pred)), str(int(pred)))

    def summary(self) -> str:
        return (f"threshold={self.threshold:.4f} margin={self.margin} "
                f"target_recall={self.target_recall} members={list(self.members)}")
