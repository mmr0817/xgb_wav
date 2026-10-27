"""wav2vec2 后端（mean pooling + MLPHead），结构与训练时完全一致。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .audio import load_for_w2v

PathLike = Union[str, Path]


def wav2vec2_base_arch() -> Optional[dict]:
    """返回 WAV2VEC2_BASE 的结构参数（纯字典，**不触发权重下载**）。

    torchaudio 的 bundle 把结构参数存在 `_params` 里；少数版本若无该属性，
    返回 None，调用方退回到 `get_model()`（该路径会联网下载 ~360 MB）。
    """
    params = getattr(torchaudio.pipelines.WAV2VEC2_BASE, "_params", None)
    if not isinstance(params, dict):
        return None
    arch = dict(params)
    conv = arch.get("extractor_conv_layer_config")
    if conv is not None:
        arch["extractor_conv_layer_config"] = [tuple(int(v) for v in layer) for layer in conv]
    return arch


def build_backbone_from_arch(arch: dict) -> nn.Module:
    """按结构参数在本地构建 wav2vec2 主干（完全离线）。"""
    from torchaudio.models.wav2vec2 import wav2vec2_model

    kwargs = dict(arch)
    conv = kwargs.get("extractor_conv_layer_config")
    if conv is not None:
        kwargs["extractor_conv_layer_config"] = [tuple(int(v) for v in layer) for layer in conv]
    return wav2vec2_model(**kwargs)


def load_wav2vec_backbone(checkpoint_path: Optional[str] = None,
                          arch: Optional[dict] = None) -> Tuple[nn.Module, int]:
    """加载 wav2vec2 主干。

    优先级：
      1) `arch`（detector.json 里记录的结构参数）→ 本地构建，**不联网**；
      2) 本地 fairseq 权重 → 导入；
      3) torchaudio bundle 的结构参数 → 本地构建，**不联网**；
      4) 最后才 `WAV2VEC2_BASE.get_model()`（会联网下载）。
    离线部署时还会用自有 `w2v_ckpt.pt` 覆盖全部权重，所以 1/3 与 4 数值等价。
    """
    if arch:
        hidden = int(arch.get("encoder_embed_dim", 768))
        return build_backbone_from_arch(arch), hidden
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser()
        if ckpt.exists():
            try:
                from torchaudio.models.wav2vec2.utils import import_fairseq_model

                state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
                model, _ = import_fairseq_model(state.get("model", state))
                return model, 768
            except Exception as e:  # pragma: no cover
                print(f"[Warn] 本地 wav2vec 权重加载失败，回退预训练模型: {e}")
    base_arch = wav2vec2_base_arch()
    if base_arch is not None:
        return build_backbone_from_arch(base_arch), int(base_arch.get("encoder_embed_dim", 768))
    return torchaudio.pipelines.WAV2VEC2_BASE.get_model(), 768


class MLPHead(nn.Module):
    """与 rs20262 训练时同构：Dropout-Linear-GELU-Dropout-Linear。"""

    def __init__(self, in_dim: int, hidden: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Wav2VecClassifier(nn.Module):
    """wav2vec2 帧特征 -> 时间维均值池化 -> MLP 分类。"""

    def __init__(self, num_labels: int = 2, hidden_size: int = 256, dropout: float = 0.1,
                 backbone: Optional[str] = None, freeze_feature_encoder: bool = True,
                 arch: Optional[dict] = None):
        super().__init__()
        self.wav2vec2, dim = load_wav2vec_backbone(backbone, arch)
        if freeze_feature_encoder and hasattr(self.wav2vec2, "feature_extractor"):
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
        self.classifier = MLPHead(dim, hidden_size, num_labels, dropout)
        self.num_labels = num_labels

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        feats, _ = self.wav2vec2.extract_features(input_values)
        h = feats[-1].mean(dim=1)
        logits = self.classifier(h)
        out = {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
        if labels is not None:
            out["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return out


def load_checkpoint(path: PathLike, device: Union[str, torch.device] = "cpu",
                    hidden_size: int = 256, dropout: float = 0.1,
                    backbone: Optional[str] = None,
                    arch: Optional[dict] = None) -> Wav2VecClassifier:
    """构建模型并加载 state_dict（忽略训练时才有的 buffer，如 class_weight）。"""
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少 wav2vec 权重: {ckpt_path}")
    model = Wav2VecClassifier(hidden_size=hidden_size, dropout=dropout,
                              backbone=backbone, arch=arch)
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] w2v 权重缺少键 {len(missing)} 个（例如 {missing[:2]}）")
    if unexpected:
        print(f"[Info] w2v 权重有多余键 {len(unexpected)} 个（例如 {unexpected[:2]}），已忽略")
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_proba(model: Wav2VecClassifier, paths: Sequence[PathLike],
                  device: Union[str, torch.device] = "cpu", batch_size: int = 32,
                  sample_rate: int = 16000, max_duration: float = 2.5) -> np.ndarray:
    """返回每条音频的 bad 概率。"""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    probs: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = np.stack([load_for_w2v(p, sample_rate, max_duration) for p in chunk])
        x = torch.from_numpy(batch).float().to(device)
        out = model(input_values=x)
        probs.append(out["probs"][:, 1].detach().cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float64)
