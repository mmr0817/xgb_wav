"""XGBoost 训练/推理封装（与 xgb_xinwei 训练时保持一致）。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import xgboost as _xgb
    from xgboost import XGBClassifier

    _XGB_VERSION: Tuple[int, ...] = tuple(
        int(p) for p in str(_xgb.__version__).split(".")[:2] if p.isdigit()
    )
except ImportError:  # pragma: no cover
    _xgb = None
    XGBClassifier = None  # type: ignore
    _XGB_VERSION = (0, 0)

EARLY_STOPPING_ROUNDS = 50


def build_params(
    n_estimators: int = 800,
    scale_pos_weight: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )
    if str(device).startswith("cuda") or device == "gpu":
        params.update(device="cuda", tree_method="hist")
    else:
        params.update(tree_method="hist")
    if _XGB_VERSION >= (2, 0):
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return params


def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    return n_neg / n_pos


def make_classifier(params: Dict[str, Any]) -> "XGBClassifier":
    if XGBClassifier is None:
        raise SystemExit("未安装 xgboost，请先 pip install -r requirements.txt")
    return XGBClassifier(**params)


def _disable_early_stopping(clf) -> None:
    try:
        if clf.get_params().get("early_stopping_rounds") is not None:
            clf.set_params(early_stopping_rounds=None)
    except (AttributeError, TypeError, ValueError):
        pass


def fit_classifier(clf, X_tr, y_tr, X_val=None, y_val=None, verbose: bool = False):
    kwargs: Dict[str, Any] = {"verbose": verbose}
    if X_val is not None and y_val is not None and len(y_val) > 0:
        kwargs["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]
        if (1, 6) <= _XGB_VERSION < (2, 0):
            kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    else:
        _disable_early_stopping(clf)
    clf.fit(X_tr, y_tr, **kwargs)
    return clf


def best_iteration_or_none(clf) -> Optional[int]:
    bi = getattr(clf, "best_iteration", None)
    if bi is None:
        return None
    try:
        return int(bi)
    except (TypeError, ValueError):
        return None


def predict_bad_proba(clf, X: np.ndarray, use_best_iteration: bool = True) -> np.ndarray:
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if _xgb is not None and isinstance(clf, _xgb.Booster):
        # 原生 Booster（来自 xgb_model.ubj）：不需要 sklearn 包装层
        return np.asarray(clf.predict(_xgb.DMatrix(X)), dtype=np.float64)
    if hasattr(clf, "predict_bad_proba"):
        return np.asarray(clf.predict_bad_proba(X), dtype=np.float64)
    bi = best_iteration_or_none(clf) if use_best_iteration else None
    if bi is not None:
        try:
            proba = clf.predict_proba(X, iteration_range=(0, bi + 1))
        except (TypeError, ValueError):
            proba = clf.predict_proba(X)
    else:
        proba = clf.predict_proba(X)
    return np.asarray(proba)[:, 1]


def scaler_to_json(scaler, feat_cfg: dict, feature_dim: int) -> Dict[str, Any]:
    """StandardScaler -> 纯 JSON（不依赖 pickle，跨平台可读）。"""
    return {
        "kind": "StandardScaler",
        "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "var": np.asarray(getattr(scaler, "var_", []), dtype=np.float64).tolist(),
        "n_features_in": int(getattr(scaler, "n_features_in_", feature_dim)),
        "with_mean": bool(getattr(scaler, "with_mean", True)),
        "with_std": bool(getattr(scaler, "with_std", True)),
        "feat_cfg": dict(feat_cfg or {}),
        "feature_dim": int(feature_dim),
    }


def scaler_from_json(meta: Dict[str, Any]):
    """从 scaler_to_json 的字典重建 StandardScaler。"""
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.mean_ = np.asarray(meta["mean"], dtype=np.float64)
    sc.scale_ = np.asarray(meta["scale"], dtype=np.float64)
    var = meta.get("var")
    sc.var_ = np.asarray(var, dtype=np.float64) if var else np.square(sc.scale_)
    sc.n_features_in_ = int(meta.get("n_features_in", sc.mean_.shape[0]))
    sc.with_mean = bool(meta.get("with_mean", True))
    sc.with_std = bool(meta.get("with_std", True))
    return sc


def export_portable(out_dir, clf, scaler, feat_cfg: dict, feature_dim: int) -> Dict[str, str]:
    """导出可移植产物：xgb_model.ubj + xgb_scaler.json（返回文件名映射）。

    相比 joblib/pickle，UBJ 是 XGBoost 官方格式，跨平台/跨版本最稳，
    可以避免 Windows 上 `input stream corrupted` 这类反序列化问题。
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    booster = clf.get_booster() if hasattr(clf, "get_booster") else clf
    ubj = out / "xgb_model.ubj"
    js = out / "xgb_scaler.json"
    booster.save_model(str(ubj))
    js.write_text(json.dumps(scaler_to_json(scaler, feat_cfg, feature_dim),
                             indent=2, ensure_ascii=False), encoding="utf-8")
    return {"xgb_model_ubj": ubj.name, "xgb_scaler": js.name}


class EnsembleModel:
    """多随机种子概率平均（可选），接口与 XGBClassifier 对齐。"""

    def __init__(self, models):
        if not models:
            raise ValueError("EnsembleModel 至少需要一个模型")
        self.models = list(models)

    @property
    def best_iteration(self):
        iters = [i for i in (best_iteration_or_none(m) for m in self.models) if i is not None]
        return max(iters) if iters else None

    def predict_bad_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        probs = []
        for m in self.models:
            bi = best_iteration_or_none(m)
            if bi is not None:
                try:
                    p = m.predict_proba(X, iteration_range=(0, bi + 1))[:, 1]
                except (TypeError, ValueError):
                    p = m.predict_proba(X)[:, 1]
            else:
                p = m.predict_proba(X)[:, 1]
            probs.append(np.asarray(p, dtype=np.float64))
        return np.mean(probs, axis=0)

    def predict_proba(self, X: np.ndarray, iteration_range=None) -> np.ndarray:
        p = self.predict_bad_proba(X)
        return np.column_stack([1.0 - p, p])
