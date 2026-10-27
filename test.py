#!/usr/bin/env python
"""xgb_wav 评测脚本：直接用 `models/` 里已训练好的权重跑测试，输出指标与报表。

三种用法
--------
1) 只评「训练时的留出标定集」（复现同一划分，指标最可信，推荐验收用）
       python test.py --data ../rs20262/data/raw --holdout

2) 全量评测某个带标签目录（good/、bad/ 两个子目录）
       python test.py --data ../rs20262/data/raw --sweep

3) 无标签，只推理并导出 CSV（单文件或目录都可以）
       python test.py --input D:\\wavs
       python test.py --input D:\\wavs\\19-24-31.24.wav

产物（默认写到 outputs/eval/）
------------------------------
    predictions.csv  每条音频：融合分 / XGB 概率 / wav2vec2 概率 / 预测 / 真值
    metrics.json     指标汇总（含混淆矩阵、分通道、两路单独表现）
    REPORT.md        可读报表（Markdown）

退出码：0 正常；1 未达到 --min-recall；2 参数或数据错误

说明
----
- 评分口径与训练完全一致：score = mean((logit(p) - mu) / sd)，bad <=> score >= threshold。
- `--holdout` 依赖 `discover_labelled` 的稳定顺序（good -> bad，各自按文件名排序）
  配合 train_test_split(seed=42, calib_frac=0.1) 复现训练时的划分，并会与
  `models/fusion.json` 中记录的标定集规模做一致性校验。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from xgb_wav._console import setup_console  # noqa: E402
from xgb_wav.data import discover_labelled, discover_wavs  # noqa: E402
from xgb_wav.pipeline import FusionDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="xgb_wav 评测：用已训练权重跑测试集并输出指标/报表",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="带标签根目录（内含 good/ 与 bad/ 子目录）")
    src.add_argument("--input", help="待检测的 wav 文件或目录（无标签，只输出预测）")

    p.add_argument("--models", default="models", help="模型目录（含 detector.json）")
    p.add_argument("--out-dir", default="outputs/eval", help="报表输出目录")
    p.add_argument("--out-csv", default=None,
                   help="预测 CSV 路径（默认 <out-dir>/predictions.csv）")

    p.add_argument("--holdout", action="store_true",
                   help="只评训练时的留出标定集（与 --data 配合，推荐验收用）")
    p.add_argument("--calib-frac", type=float, default=0.1, help="留出比例，需与训练一致")
    p.add_argument("--seed", type=int, default=42, help="划分随机种子，需与训练一致")

    p.add_argument("--threshold", type=float, default=None,
                   help="覆盖模型阈值（默认用 models/fusion.json）")
    p.add_argument("--sweep", action="store_true", help="附阈值扫描表，便于权衡漏检/误杀")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（快速冒烟）")
    p.add_argument("--min-recall", type=float, default=None,
                   help="召回低于该值时以退出码 1 结束（产线准入门禁）")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def binary_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> Dict[str, float]:
    """与训练脚本同口径的指标（召回优先）。"""
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n_bad, n_good = tp + fn, tn + fp
    recall = tp / max(1, n_bad)
    precision = tp / max(1, tp + fp)
    specificity = tn / max(1, n_good)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    auc = float("nan")
    if n_bad and n_good:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, score))
        except Exception:  # pragma: no cover
            pass
    return {
        "n": int(len(y)), "n_bad": n_bad, "n_good": n_good, "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision, "specificity": specificity,
        "f1": f1, "accuracy": (tp + tn) / max(1, len(y)),
        "false_kill_rate": fp / max(1, n_good), "miss_rate": fn / max(1, n_bad),
        "auc": auc,
    }


def _candidate_thresholds(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """候选阈值：真实分数分位点 + 每个分位点之间的中点 + bad 最低分（零漏检边界）。"""
    qs = np.quantile(score, np.linspace(0.0, 1.0, 201))
    mid = (qs[1:] + qs[:-1]) / 2.0
    cands = [qs, mid]
    bad = score[y == 1]
    if bad.size:
        cands.append(np.array([float(bad.min()) - 1e-6]))
    good = score[y == 0]
    if good.size:
        cands.append(np.array([float(good.max()) + 1e-6]))
    return np.unique(np.round(np.concatenate(cands), 6))


def threshold_sweep(y: np.ndarray, score: np.ndarray, n: int = 11) -> Dict[str, object]:
    """扫描阈值：返回展示用网格 + 精确候选点上的最优解。

    - best_f1      : 精确候选点中 F1 最高
    - best_recall_995: 召回 >= 0.995 时误杀最小
    - zero_miss    : 零漏检（阈值刚好低于 bad 的最低分）
    """
    lo, hi = float(np.min(score)), float(np.max(score))
    pad = max(0.05, (hi - lo) * 0.05)
    grid = np.linspace(lo - pad, hi + pad, n)
    grid_rows = [binary_metrics(y, score, float(t)) for t in grid]
    cand_rows = [binary_metrics(y, score, float(t)) for t in _candidate_thresholds(y, score)]

    best_f1 = max(cand_rows, key=lambda r: (r["f1"], -r["false_kill_rate"], r["threshold"]))
    strict = [r for r in cand_rows if r["recall"] >= 0.995]
    best_strict = (min(strict, key=lambda r: (r["false_kill_rate"], -r["recall"], -r["threshold"]))
                   if strict else best_f1)
    zero_fn = [r for r in cand_rows if r["fn"] == 0]
    # 同分时取较大阈值：即「零漏检所能容忍的最高阈值」，更有参考价值
    zero_best = (min(zero_fn, key=lambda r: (r["false_kill_rate"], -r["threshold"]))
                 if zero_fn else None)
    return {"sweep_grid": grid_rows, "best_f1": best_f1,
            "best_recall_995": best_strict, "zero_miss": zero_best}


def member_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """两路成员单独表现（阈值 0.5），用于诊断谁在贡献。"""
    if "true" not in df.columns:
        return {}
    y = df["true"].to_numpy(dtype=int)
    out: Dict[str, Dict[str, float]] = {}
    for name, col in (("xgb_only", "xgb_proba"), ("w2v_only", "w2v_proba")):
        if col in df.columns:
            out[name] = binary_metrics(y, df[col].to_numpy(dtype=float), 0.5)
    return out


def channel_metrics(df: pd.DataFrame, thr: float) -> Dict[str, Dict[str, float]]:
    """按采集通道（文件名 -P/-F）分组看指标。"""
    if "true" not in df.columns:
        return {}
    return {str(ch): binary_metrics(g["true"].to_numpy(dtype=int),
                                    g["score"].to_numpy(dtype=float), thr)
            for ch, g in df.groupby("channel")}


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def prepare_labelled(args: argparse.Namespace) -> Tuple[List[Path], np.ndarray, str]:
    """返回 (paths, labels, 范围描述)。--holdout 时复现训练时的划分。"""
    paths, labels = discover_labelled(args.data)
    y = np.asarray(labels, dtype=int)
    scope = "全量带标签数据"

    if args.holdout:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(y))
        _, ca = train_test_split(idx, test_size=args.calib_frac, stratify=y,
                                 random_state=args.seed)
        ca = np.sort(ca)
        paths = [paths[i] for i in ca]
        y = y[ca]
        scope = f"留出标定集(calib_frac={args.calib_frac}, seed={args.seed})"

    if args.limit and args.limit < len(paths):
        keep = np.linspace(0, len(paths) - 1, args.limit).astype(int)
        paths = [paths[i] for i in keep]
        y = y[keep]
        scope += f" | 仅 {args.limit} 条"
    return paths, y, scope


# --------------------------------------------------------------------------- #
# 打印与报表
# --------------------------------------------------------------------------- #
BAR = "-" * 78


def print_table(rows: Sequence[Tuple[str, Dict[str, float]]]) -> None:
    print(f"{'scheme':<20}{'thr':>8}{'recall':>8}{'prec':>8}{'F1':>8}"
          f"{'false-kill':>17}{'miss':>6}{'AUC':>8}")
    print("-" * 84)
    for name, m in rows:
        fk = f"{m['fp']}/{m['n_good']}({m['false_kill_rate']:.1%})"
        auc = "n/a" if np.isnan(m["auc"]) else f"{m['auc']:.4f}"
        print(f"{name:<20}{m['threshold']:>8.4f}{m['recall']:>8.4f}{m['precision']:>8.4f}"
              f"{m['f1']:>8.4f}{fk:>17}{m['fn']:>6d}{auc:>8}")


def print_confusion(m: Dict[str, float]) -> None:
    print("混淆矩阵:")
    print(f"{'':<14}{'pred bad':>10}{'pred good':>11}")
    print(f"{'true bad':<14}{m['tp']:>10}{m['fn']:>11}")
    print(f"{'true good':<13}{m['fp']:>10}{m['tn']:>11}")


def _auc_str(m: Dict[str, float]) -> str:
    return "n/a" if np.isnan(m["auc"]) else f"{m['auc']:.4f}"


def write_report(path: Path, ctx: Dict[str, object], main: Dict[str, float],
                 members: Dict[str, Dict[str, float]],
                 channels: Dict[str, Dict[str, float]],
                 sweep: Optional[Dict[str, object]]) -> None:
    L: List[str] = ["# xgb_wav 评测报告", "", "| 项目 | 值 |", "|---|---|"]
    for k, v in ctx.items():
        L.append(f"| {k} | {v} |")
    L.append("")

    L += ["## 主指标（当前阈值）", "", "| 指标 | 值 |", "|---|---|",
          f"| 样本数 | {main['n']}（bad {main['n_bad']} / good {main['n_good']}） |",
          f"| 阈值 | {main['threshold']:.4f} |",
          f"| **召回率** | **{main['recall']:.4f}**（漏检 {main['fn']}） |",
          f"| 精度 | {main['precision']:.4f} |",
          f"| 特异度 | {main['specificity']:.4f} |",
          f"| F1 | {main['f1']:.4f} |",
          f"| 准确率 | {main['accuracy']:.4f} |",
          f"| **误杀率** | **{main['false_kill_rate']:.2%}**（{main['fp']}/{main['n_good']}） |",
          f"| AUC | {_auc_str(main)} |",
          "", "### 混淆矩阵", "", "| | 预测 bad | 预测 good |", "|---|---|---|",
          f"| **真实 bad** | {main['tp']} (TP) | {main['fn']} (FN·漏检) |",
          f"| **真实 good** | {main['fp']} (FP·误杀) | {main['tn']} (TN) |", ""]

    if members:
        L += ["## 两路成员单独表现（阈值 0.5）", "",
              "| 成员 | 召回 | 精度 | F1 | 误杀 | 漏检 |", "|---|---|---|---|---|---|"]
        for name, m in members.items():
            L.append(f"| {name} | {m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | "
                     f"{m['fp']}/{m['n_good']} | {m['fn']} |")
        L.append("")

    if channels:
        L += ["## 分通道", "", "| 通道 | 样本 | 召回 | 误杀 | 漏检 |", "|---|---|---|---|---|"]
        for ch, m in channels.items():
            L.append(f"| {ch} | {m['n']} | {m['recall']:.4f} | "
                     f"{m['fp']}/{m['n_good']} | {m['fn']} |")
        L.append("")

    if sweep:
        L += ["## 阈值扫描", "", "| 阈值 | 召回 | 精度 | F1 | 误杀率 | 漏检 |",
              "|---|---|---|---|---|---|"]
        for m in sweep["sweep_grid"]:  # type: ignore[index]
            L.append(f"| {m['threshold']:.3f} | {m['recall']:.4f} | {m['precision']:.4f} | "
                     f"{m['f1']:.4f} | {m['false_kill_rate']:.2%} | {m['fn']} |")
        b, s = sweep["best_f1"], sweep["best_recall_995"]  # type: ignore[index]
        L += ["", f"- F1 最优：阈值 {b['threshold']:.3f} → F1 {b['f1']:.4f}，"
                  f"召回 {b['recall']:.4f}，误杀 {b['false_kill_rate']:.2%}",
              f"- 召回≥99.5% 中误杀最小：阈值 {s['threshold']:.3f} → "
              f"误杀 {s['false_kill_rate']:.2%}，召回 {s['recall']:.4f}"]
        z = sweep.get("zero_miss")
        if z:
            L.append(f"- 零漏检：阈值 {z['threshold']:.3f} → 误杀 {z['false_kill_rate']:.2%}")
            L.append(f"- 当前阈值 {main['threshold']:.3f} 距零漏检边界尚有 "
                     f"{z['threshold'] - main['threshold']:.3f} 余量（即训练时的 margin）")
        L.append("")

    L += ["## 说明", "",
          "- 评分口径与训练一致：`score = mean((logit(p_i) - mu_i) / sd_i)`，"
          "`bad ⇔ score >= threshold`；阈值与 mu/sd 来自 `models/fusion.json`。",
          "- 未开启 `--holdout` 时评测集可能与训练集重叠，**指标偏乐观**，"
          "仅用于回归自测；验收请用 `--holdout` 或全新产线数据。"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    setup_console()
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else out_dir / "predictions.csv"

    print(BAR)
    print("xgb_wav 评测（使用已训练权重，不再训练）")
    print(BAR)

    t_load = time.time()
    det = FusionDetector.from_dir(args.models, device=args.device)
    if args.batch_size:
        det.cfg.batch_size = args.batch_size
    print(f"模型目录 : {Path(args.models).resolve()}")
    print(f"融合规则 : {det.rule.summary()}")
    print(f"设备     : {det.device} (加载 {time.time() - t_load:.1f}s)")

    thr = float(args.threshold) if args.threshold is not None else float(det.rule.threshold)

    # ---- 数据 ----
    y: Optional[np.ndarray] = None
    if args.data:
        paths, y, scope = prepare_labelled(args)
    else:
        paths = discover_wavs(args.input)
        if args.limit:
            paths = paths[: args.limit]
        scope = "无标签推理"
    print(f"评测范围 : {scope}")
    print(f"样本数   : {len(paths)}")
    if not paths:
        print("[错误] 没有找到 wav 文件")
        return 2

    # ---- 推理 ----
    t0 = time.time()
    df = det.predict_paths(paths)
    elapsed = time.time() - t0
    if df.empty:
        print("[错误] 推理结果为空（所有文件都读取失败？）")
        return 2
    n_fail = int(df.attrs.get("skipped", 0))
    print(f"推理完成 : {len(df)} 条 / {elapsed:.1f}s "
          f"({elapsed / max(1, len(df)) * 1000:.0f} ms/条)，失败 {n_fail} 条")

    df["threshold"] = thr
    df["pred"] = (df["score"].to_numpy() >= thr).astype(int)
    df["label"] = [det.rule.label_of(v) for v in df["pred"]]

    ctx: Dict[str, object] = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "模型目录": str(Path(args.models).resolve()),
        "设备": det.device,
        "评测范围": scope,
        "阈值来源": "命令行 --threshold" if args.threshold is not None else "models/fusion.json",
        "阈值": f"{thr:.4f}",
    }

    # ================= 无标签模式 =================
    if y is None:
        n_bad = int(df["pred"].sum())
        cols = ["path", "file", "label", "pred", "score", "xgb_proba", "w2v_proba",
                "threshold", "channel", "batch"]
        df[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
        ctx["预测结果"] = f"共 {len(df)} 条，bad {n_bad}，good {len(df) - n_bad}"
        print(BAR)
        print(f"预测结果：共 {len(df)} 条，判定 bad {n_bad} 条，good {len(df) - n_bad} 条")
        print(f"CSV      : {out_csv}")
        (out_dir / "metrics.json").write_text(
            json.dumps({"context": ctx}, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    # ================= 带标签模式 =================
    label_of_path = {str(p): int(v) for p, v in zip(paths, y)}
    df["true"] = df["path"].map(label_of_path)
    n_lost = int(df["true"].isna().sum())
    if n_lost:
        print(f"[警告] {n_lost} 条预测无法匹配标签，已剔除")
        df = df[df["true"].notna()].reset_index(drop=True)
    df["true"] = df["true"].astype(int)
    df["correct"] = (df["true"] == df["pred"]).astype(int)

    yy = df["true"].to_numpy(dtype=int)
    sc = df["score"].to_numpy(dtype=float)
    main_m = binary_metrics(yy, sc, thr)
    ctx["样本数"] = f"{main_m['n']}（bad {main_m['n_bad']} / good {main_m['n_good']}）"

    print(BAR)
    print("主指标")
    print(BAR)
    print_table([("model threshold", main_m)])
    print()
    print_confusion(main_m)

    members = member_metrics(df)
    if members:
        print()
        print("两路成员单独表现（阈值 0.5）")
        print_table(list(members.items()))

    chans = channel_metrics(df, thr)
    if len(chans) > 1:
        print()
        print("分通道")
        print_table([(f"channel={k}", v) for k, v in chans.items()])

    sweep = threshold_sweep(yy, sc) if args.sweep else None
    if sweep:
        print()
        print("阈值扫描（关键点）")
        rows = [("best F1", sweep["best_f1"]), ("recall>=99.5%", sweep["best_recall_995"])]
        if sweep.get("zero_miss"):
            rows.append(("zero miss", sweep["zero_miss"]))
        print_table(rows)

    calib = det.rule.meta.get("calibration") or {}
    if args.holdout and calib.get("n"):
        n_rec = int(calib["n"])
        flag = "一致" if n_rec == main_m["n"] else "不一致（数据或划分参数已变）"
        print(f"\n标定集一致性：本次 {main_m['n']} 条 vs fusion.json 记录 {n_rec} 条 -> {flag}")

    metrics = {"main": main_m, "members": members, "channels": chans,
               "sweep": {k: v for k, v in (sweep or {}).items() if k != "sweep_grid"},
               "context": ctx}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "REPORT.md", ctx, main_m, members, chans, sweep)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n报表      : {out_dir / 'REPORT.md'}")
    print(f"指标 JSON : {out_dir / 'metrics.json'}")
    print(f"预测 CSV  : {out_csv}")

    if args.min_recall is not None and main_m["recall"] < args.min_recall:
        print(f"\n[失败] 召回 {main_m['recall']:.4f} < 要求 {args.min_recall:.4f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())




