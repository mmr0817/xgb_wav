"""生产训练：用全部数据训练 XGB + wav2vec2，并用 K 折 OOF 标定「宁杀勿漏」阈值。

流程：
  1) 发现 good/bad 全部样本（不含 critical/transient）
  2) 提特征（带缓存）
  3) K 折 OOF：每折训练 XGB 与 wav2vec2，得到全量 OOF 概率
  4) 在 OOF 上计算 logit 标准化统计量，阈值 = min(OOF 中 bad 的融合分) - margin
  5) 用全部数据训练最终 XGB 与 wav2vec2
  6) 写出 models/{xgb_pipeline.joblib, w2v_ckpt.pt, fusion.json, detector.json}

用法：
  python scripts/train_production.py --data ../rs20262/data/raw --out models \
      --oof-folds 3 --oof-epochs 35 --final-epochs 60 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgb_wav.audio import load_for_w2v  # noqa: E402
from xgb_wav.data import discover_labelled  # noqa: E402
from xgb_wav.features import XinWeiFeatureConfig  # noqa: E402
from xgb_wav.featurize import FeatureExtractor  # noqa: E402
from xgb_wav.model_w2v import Wav2VecClassifier  # noqa: E402
from xgb_wav.model_xgb import (  # noqa: E402
    build_params, compute_scale_pos_weight, fit_classifier, make_classifier,
    predict_bad_proba,
)

DEFAULT_FEAT = dict(sample_rate=44100, n_mfcc=13, n_fft=2048, hop_length=512,
                    fmin=100.0, fmax=10000.0, n_harmonics=5, align=True)
SR = 16000
MAX_DUR = 2.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="raw 目录（含 good/bad）")
    p.add_argument("--out", default="models")
    p.add_argument("--ref", default=None, help="参考扫频 wav（默认 <out>/ref_sweep.wav）")
    p.add_argument("--cache", default=None, help="特征缓存（默认 <out>/features_cache.npz）")
    p.add_argument("--oof-folds", type=int, default=3)
    p.add_argument("--oof-epochs", type=int, default=35)
    p.add_argument("--final-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def w2v_batch(paths: List[Path], batch_size: int):
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        wav = np.stack([load_for_w2v(p, SR, MAX_DUR) for p in chunk])
        yield torch.from_numpy(wav).float()


def train_w2v(paths: List[Path], labels: np.ndarray, epochs: int, batch_size: int,
              device: torch.device, lr: float = 1e-4, seed: int = 42,
              log_every: int = 10) -> Wav2VecClassifier:
    """在给定数据上训练 wav2vec2 分类器（固定轮数，无早停）。"""
    set_seed(seed)
    model = Wav2VecClassifier(num_labels=2, hidden_size=256, dropout=0.1)
    for p in model.wav2vec2.feature_extractor.parameters():
        p.requires_grad = False
    model.to(device)

    class_weight = torch.tensor(
        [len(labels) / (2 * max(1, int((labels == c).sum()))) for c in (0, 1)],
        dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=0.01)
    steps = math.ceil(len(paths) / batch_size)
    total = max(1, steps * epochs)
    warmup = int(total * 0.1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    order = np.arange(len(paths))

    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(order)
        shuf = [paths[i] for i in order]
        y_shuf = torch.from_numpy(labels[order]).long()
        loss_sum, n, ptr = 0.0, 0, 0
        for xb in w2v_batch(shuf, batch_size):
            bs = xb.shape[0]
            yb = y_shuf[ptr:ptr + bs].to(device)
            ptr += bs
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_values=xb.to(device))["logits"]
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            loss_sum += float(loss)
            n += 1
        if epoch % log_every == 0 or epoch == epochs:
            print(f"    [w2v] epoch {epoch}/{epochs} loss={loss_sum / max(1, n):.4f}")
    model.eval()
    return model


@torch.no_grad()
def w2v_probs(model: Wav2VecClassifier, paths: List[Path], batch_size: int,
              device: torch.device) -> np.ndarray:
    out = []
    for xb in w2v_batch(paths, batch_size):
        logits = model(input_values=xb.to(device))["logits"]
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit_xgb(X: np.ndarray, y: np.ndarray, seed: int = 42, n_estimators: int = 800):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    params = build_params(n_estimators=n_estimators,
                          scale_pos_weight=compute_scale_pos_weight(y),
                          seed=seed, device="cpu", max_depth=4, learning_rate=0.05)
    params.pop("early_stopping_rounds", None)  # 全量训练，无验证集
    clf = make_classifier(params)
    fit_classifier(clf, Xs, y)
    return clf, scaler


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(args.ref).expanduser() if args.ref else (out / "ref_sweep.wav")
    cache = Path(args.cache).expanduser() if args.cache else (out / "features_cache.npz")

    paths, labels_list = discover_labelled(data_dir)
    y0 = np.asarray(labels_list, dtype=np.int64)
    print(f"样本 {len(paths)}（bad {int(y0.sum())} / good {int((y0 == 0).sum())}）")

    feat_cfg = XinWeiFeatureConfig(**DEFAULT_FEAT)
    extractor = FeatureExtractor(feat_cfg, ref if ref.exists() else None)
    X, y, kept = extractor.fit_cache(cache, paths, y0)
    paths = [Path(p) for p in kept]
    print(f"特征 {X.shape}（参考音频: {'有' if extractor.ref_wav is not None else '无'}）")

    # ---------------- OOF 标定 ----------------
    if args.oof_folds < 2:
        raise SystemExit("--oof-folds 至少为 2（需要 OOF 来标定阈值）")
    oof_x = np.full(len(y), np.nan)
    oof_w = np.full(len(y), np.nan)
    skf = StratifiedKFold(args.oof_folds, shuffle=True, random_state=args.seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        t0 = time.time()
        clf, sc = fit_xgb(X[tr], y[tr], seed=args.seed)
        oof_x[te] = predict_bad_proba(clf, sc.transform(X[te]))
        m = train_w2v([paths[i] for i in tr], y[tr], args.oof_epochs,
                      args.batch_size, device, seed=args.seed)
        oof_w[te] = w2v_probs(m, [paths[i] for i in te], args.batch_size, device)
        print(f"  fold {k}/{args.oof_folds} 完成（{time.time() - t0:.0f}s）")

    ok = ~np.isnan(oof_x) & ~np.isnan(oof_w)
    lx, lw = np.log(np.clip(oof_x[ok], 1e-7, 1 - 1e-7) /
                    (1 - np.clip(oof_x[ok], 1e-7, 1 - 1e-7))), \
             np.log(np.clip(oof_w[ok], 1e-7, 1 - 1e-7) /
                    (1 - np.clip(oof_w[ok], 1e-7, 1 - 1e-7)))
    mu_x, sd_x = float(lx.mean()), float(lx.std() + 1e-9)
    mu_w, sd_w = float(lw.mean()), float(lw.std() + 1e-9)
    fused = (((lx - mu_x) / sd_x) + ((lw - mu_w) / sd_w)) / 2.0
    yv = y[ok]
    thr = float(fused[yv == 1].min() - args.margin)
    pred = (fused >= thr).astype(int)
    oof_metrics = {
        "n": int(len(yv)), "bad": int(yv.sum()),
        "threshold": thr, "margin": args.margin,
        "recall": float((pred[yv == 1] == 1).mean()),
        "precision": float(pred[yv == 1].sum() / max(1, pred.sum())),
        "fp": int(((pred == 1) & (yv == 0)).sum()),
        "fn": int(((pred == 0) & (yv == 1)).sum()),
    }
    print(f"OOF 标定：阈值={thr:.4f} 召回={oof_metrics['recall']:.4f} "
          f"精度={oof_metrics['precision']:.4f} 误杀={oof_metrics['fp']}/{int((yv == 0).sum())} "
          f"漏检={oof_metrics['fn']}")

    # ---------------- 全量训练 ----------------
    print("训练最终 XGB（全部数据）…")
    clf, scaler = fit_xgb(X, y, seed=args.seed)
    import joblib

    joblib.dump({"model": clf, "scaler": scaler, "feat_cfg": feat_cfg.__dict__,
                 "feature_dim": int(X.shape[1]), "n_train": int(len(y))},
                out / "xgb_pipeline.joblib")
    print("训练最终 wav2vec2（全部数据）…")
    w2v = train_w2v(paths, y, args.final_epochs, args.batch_size, device, seed=args.seed)
    torch.save({"model_state_dict": w2v.state_dict(),
                "config": {"num_labels": 2, "hidden_size": 256, "dropout": 0.1,
                           "sample_rate": SR, "max_duration": MAX_DUR,
                           "epochs": args.final_epochs, "n_train": int(len(y))}},
               out / "w2v_ckpt.pt")

    fusion = {
        "fusion": "mean_standardized_logit",
        "members": {
            "xgb": {"model": "xgb_pipeline.joblib", "logit_mean": mu_x, "logit_std": sd_x},
            "w2v": {"model": "w2v_ckpt.pt", "logit_mean": mu_w, "logit_std": sd_w},
        },
        "threshold": thr, "margin": args.margin, "target_recall": 0.995,
        "label_map": {"0": "good", "1": "bad"},
        "oof_folds": args.oof_folds, "oof_metrics": oof_metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "fusion.json").write_text(json.dumps(fusion, indent=2, ensure_ascii=False), encoding="utf-8")
    detector = {
        "xgb_model": "xgb_pipeline.joblib", "w2v_ckpt": "w2v_ckpt.pt",
        "fusion": "fusion.json",
        "ref_sweep": "ref_sweep.wav" if ref.exists() else None,
        "w2v_sample_rate": SR, "w2v_max_duration": MAX_DUR, "w2v_hidden": 256,
        "batch_size": args.batch_size,
    }
    (out / "detector.json").write_text(json.dumps(detector, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n模型已写出到 {out}: xgb_pipeline.joblib / w2v_ckpt.pt / fusion.json / detector.json")


if __name__ == "__main__":
    main()

