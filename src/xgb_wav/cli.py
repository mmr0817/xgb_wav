"""命令行入口：批量检测并输出 CSV。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._console import setup_console
from .data import discover_wavs
from .settings import set_threshold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="xgb_wav：工业音频坏件检测（XGB + wav2vec2 融合）")
    p.add_argument("--models", default="models", help="模型目录（含 detector.json）")
    p.add_argument("--input", required=True, help="待检测的 wav 文件或文件夹")
    p.add_argument("--output", default="predictions.csv", help="输出 CSV 路径")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threshold", type=float, default=None,
                   help="覆盖判定阈值（默认用 models/fusion.json）")
    p.add_argument("--save-threshold", action="store_true",
                   help="把 --threshold 写入程序目录的 settings.json（GUI 会读取）")
    p.add_argument("--batch-size", type=int, default=None, help="wav2vec2 推理 batch（默认取模型配置）")
    p.add_argument("--low-power", action="store_true",
                   help="省电模式：CPU+单线程+限速，速度约 1/4，占用/发热更低")
    p.add_argument("--auto-threshold", choices=["auto", "all_bad", "all_good"], default=None,
                   help="按本批分数自动定阈：auto=找自然分界，all_bad=全命中（防漏检），"
                        "all_good=全部放行（防误杀）")
    p.add_argument("--info", action="store_true", help="只打印模型信息后退出")
    return p.parse_args()


def main() -> None:
    setup_console()
    args = parse_args()
    if args.low_power:
        from .perf import enable_low_power

        enable_low_power()
        print("省电模式已开启：CPU / 单线程 / 限速（约 1/4 速度）")
    from .pipeline import FusionDetector

    det = FusionDetector.from_dir(args.models, device=args.device)
    if args.threshold is not None:
        det.set_threshold(args.threshold)
        print(f"已覆盖阈值: {det.threshold:.4f}")
        if args.save_threshold:
            print(f"阈值已写入: {set_threshold(det.threshold)}")
    print(f"模型目录: {Path(args.models).resolve()}")
    print(f"融合规则: {det.rule.summary()}")
    print(f"设备: {det.device}")
    if args.info:
        meta = det.rule.meta
        print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        return
    if args.batch_size:
        det.cfg.batch_size = args.batch_size

    src = Path(args.input).expanduser()
    paths = discover_wavs(src)
    print(f"待检测: {len(paths)} 个 wav（{src}）")
    df = det.predict_paths(paths)
    if args.auto_threshold:
        from .autothr import apply_mode

        auto_thr = apply_mode(df["score"].to_numpy(), args.auto_threshold, det.threshold)
        det.set_threshold(auto_thr)
        df["threshold"] = auto_thr
        df["pred"] = (df["score"].to_numpy() >= auto_thr).astype(int)
        df["label"] = [det.rule.label_of(v) for v in df["pred"]]
        n_bad = int(df["pred"].sum())
        print(f"自动定阈({args.auto_threshold}): {auto_thr:.4f}"
              f" → bad {n_bad} / good {len(df) - n_bad}")
        if args.save_threshold:
            print(f"阈值已写入: {set_threshold(auto_thr)}")
    out = det.save_csv(df, args.output)
    print(f"耗时 {df.attrs.get('elapsed', 0):.1f}s，失败 {df.attrs.get('skipped', 0)} 条")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
