"""命令行入口：批量检测并输出 CSV。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import discover_wavs
from .pipeline import FusionDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="xgb_wav：工业音频坏件检测（XGB + wav2vec2 融合）")
    p.add_argument("--models", default="models", help="模型目录（含 detector.json）")
    p.add_argument("--input", required=True, help="待检测的 wav 文件或文件夹")
    p.add_argument("--output", default="predictions.csv", help="输出 CSV 路径")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=None, help="wav2vec2 推理 batch（默认取模型配置）")
    p.add_argument("--info", action="store_true", help="只打印模型信息后退出")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    det = FusionDetector.from_dir(args.models, device=args.device)
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
    out = det.save_csv(df, args.output)
    print(f"耗时 {df.attrs.get('elapsed', 0):.1f}s，失败 {df.attrs.get('skipped', 0)} 条")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
