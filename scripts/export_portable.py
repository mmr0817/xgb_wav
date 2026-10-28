"""把 xgb_pipeline.joblib 导出成跨平台可移植的产物。

背景：joblib/pickle 里嵌的是 XGBoost 的二进制 booster，跨平台/跨构建反序列化时
可能报 `input stream corrupted`（用户在 Windows 上实测遇到）。改用 XGBoost 官方
原生格式：

    models/xgb_model.ubj      Booster.save_model() 写出的 UBJ（官方推荐的可移植格式）
    models/xgb_scaler.json    StandardScaler 的 mean_/scale_ 等标量 + 特征配置

推理端优先读这两个文件，读不到才回退 joblib。

用法：
    python scripts/export_portable.py --models models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgb_wav._console import setup_console  # noqa: E402
from xgb_wav.model_xgb import export_portable, scaler_from_json  # noqa: E402


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--samples", type=int, default=2000, help="等价性校验用的随机样本数")
    args = ap.parse_args()

    out = Path(args.models).expanduser().resolve()
    jbl = out / "xgb_pipeline.joblib"
    if not jbl.exists():
        print(f"[错误] 找不到 {jbl}")
        return 2

    pack = joblib.load(jbl)
    clf, scaler = pack["model"], pack["scaler"]
    feat_cfg = pack.get("feat_cfg", {}) or {}
    feature_dim = int(pack.get("feature_dim", getattr(scaler, "n_features_in_", 0)))

    names = export_portable(out, clf, scaler, feat_cfg, feature_dim)
    ubj = out / names["xgb_model_ubj"]
    js = out / names["xgb_scaler"]
    print(f"已写出 {ubj}  ({ubj.stat().st_size / 1024:.0f} KB)")
    print(f"已写出 {js}  ({js.stat().st_size / 1024:.1f} KB)")

    # ---- 等价性校验：原 sklearn 管线 vs UBJ+JSON 重建 ----
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(args.samples, feature_dim)).astype(np.float32)
    if hasattr(clf, "predict_proba"):
        p_ref = np.asarray(clf.predict_proba(scaler.transform(X)))[:, 1]
    else:
        import xgboost as xgb

        p_ref = clf.predict(xgb.DMatrix(X)) if hasattr(clf, "predict") else None

    import xgboost as xgb

    sc2 = scaler_from_json(json.loads(js.read_text(encoding="utf-8")))
    b2 = xgb.Booster(model_file=str(ubj))
    p_new = b2.predict(xgb.DMatrix(sc2.transform(X)))

    diff = float(np.abs(np.asarray(p_ref) - p_new).max())
    print(f"等价性校验：{args.samples} 条随机样本，最大概率差 = {diff:.3e}")
    if diff > 1e-6:
        print("[失败] UBJ+JSON 与 joblib 结果不一致")
        return 1

    # 版本信息
    cfg = json.loads(b2.save_config())
    print(f"模型版本 = {cfg.get('version')}")
    print("OK：Windows/Linux/macOS 均可直接加载 xgb_model.ubj + xgb_scaler.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
