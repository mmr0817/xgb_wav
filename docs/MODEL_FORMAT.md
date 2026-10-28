# 模型文件格式（为什么有 UBJ 和 joblib 两份）

## 结论先说

推理端**优先加载**：

```
models/xgb_model.ubj      # XGBoost 官方二进制格式（Booster.save_model）
models/xgb_scaler.json    # StandardScaler 的 mean_/scale_/var_ + 特征配置（纯 JSON）
```

只有这两个文件缺失时，才回退旧的 `models/xgb_pipeline.joblib`（joblib/pickle）。

## 为什么要改

在 Windows 上实测到：

```
xgboost._c_api.XGBoostError: input stream corrupted
   at xgboost/core.py:1906 in __setstate__
   at joblib/numpy_pickle.py load_build
```

Python 3.12 + xgboost 3.4.1 版本完全正确、文件哈希与训练机一致，仍无法反序列化。
根因是 **pickle 里嵌套的 XGBoost 二进制 buffer 跨平台加载不稳定**（pickle 的
`__setstate__` 直接把字节喂给 `XGBoosterUnserializeFromBuffer`）。

改用官方格式后：

- `.ubj` 是 XGBoost 自己保证可移植的模型格式，加载报错信息也更明确；
- `xgb_scaler.json` 只有浮点数组和配置，任何语言/平台都能读，不涉及 pickle；
- 不再依赖 `sklearn` 版本的 pickle 兼容性。

## 等价性验证

```
$ python scripts/export_portable.py --models models
已写出 models/xgb_model.ubj  (758 KB)
已写出 models/xgb_scaler.json  (9.0 KB)
等价性校验：2000 条随机样本，最大概率差 = 0.000e+00
OK：Windows/Linux/macOS 均可直接加载 xgb_model.ubj + xgb_scaler.json
```

实际推理端到端（8 条 demo / 204 条留出集）与 joblib 路径**逐位一致**。

## 文件结构

| 文件 | 内容 |
|---|---|
| `xgb_model.ubj` | XGBoost Booster（758 KB，树的完整结构 + 权重） |
| `xgb_scaler.json` | `{mean[], scale[], var[], n_features_in, with_mean, with_std, feat_cfg{}, feature_dim}` |
| `xgb_pipeline.joblib` | 旧格式，保留以兼容老代码 |

## 训练时自动生成

`scripts/train_production.py` 的 `save_bundle()` 现在会同时写出 UBJ + JSON，
并在 `detector.json` 中登记：

```json
{
  "xgb_model": "xgb_pipeline.joblib",
  "xgb_model_ubj": "xgb_model.ubj",
  "xgb_scaler": "xgb_scaler.json"
}
```

## 只用旧产物怎么办

```bash
python scripts/export_portable.py --models models   # 转换 + 校验，无需重训
```

## 相关版本约束

- 模型由 `xgboost 3.4.1` 保存 → 推理环境也必须是 `3.4.1`（该版本要求 Python ≥ 3.12）
- `requirements.txt` 已锁定 `xgboost==3.4.1`
- `doctor.py` 会打印「模型版本 vs 库版本」，不一致直接报错
