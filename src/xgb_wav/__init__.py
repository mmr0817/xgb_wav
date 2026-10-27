"""xgb_wav —— 工业音频坏件检测（新威特征 XGB + wav2vec2 融合）。

对外入口：
  from xgb_wav import FusionDetector
  det = FusionDetector.from_dir("models")
  df = det.predict_paths([Path("a.wav"), Path("b.wav")])
"""

from __future__ import annotations

__version__ = "1.0.0"

from .pipeline import FusionDetector  # noqa: E402

__all__ = ["FusionDetector", "__version__"]
