"""冒烟测试：融合规则与音频预处理（无需模型权重）。

运行：
    pytest -q tests/test_smoke.py
"""

from __future__ import annotations

import numpy as np

from xgb_wav.audio import pad_or_truncate
from xgb_wav.data import batch_of, channel_of
from xgb_wav.fusion import FusionRule, logit


def test_logit_roundtrip():
    p = np.array([0.001, 0.5, 0.999])
    assert np.allclose(1 / (1 + np.exp(-logit(p))), p)


def test_pad_or_truncate_centered():
    wav = np.arange(10, dtype=np.float32)
    padded = pad_or_truncate(wav, 14)
    assert padded.shape[0] == 14
    # 居中补齐：左侧 2 个 0，右侧 2 个 0
    assert np.allclose(padded[:2], 0) and np.allclose(padded[-2:], 0)
    assert np.allclose(padded[2:12], wav)

    cut = pad_or_truncate(wav, 6)
    assert cut.shape[0] == 6
    assert np.allclose(cut, wav[2:8])


def test_channel_and_batch_parse():
    assert channel_of("501-16：42：07-WAV-In 1L-1-F.wav") == "F"
    assert channel_of("x-1L-1-P.wav") == "P"
    assert channel_of("123.wav") == "none"
    assert batch_of("良品_9.1_16-30-10.24.wav").startswith("dev:")


def test_fusion_rule_monotonic():
    rule = FusionRule(
        members={"xgb": {"logit_mean": 0.0, "logit_std": 1.0},
                 "w2v": {"logit_mean": 0.0, "logit_std": 1.0}},
        threshold=0.0,
    )
    s_good = rule.score(0.01, 0.02)
    s_bad = rule.score(0.99, 0.95)
    assert s_bad > s_good
    assert rule.decide(s_bad) == 1 and rule.decide(s_good) == 0
    assert rule.label_of(1) == "bad" and rule.label_of(0) == "good"
