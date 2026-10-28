# models/ 目录说明

| 文件 | 体积 | 是否入库 | 说明 |
|---|---|---|---|
| `detector.json` | <1 KB | ✅ | 推理入口：指明下面各文件的相对路径与预处理参数 |
| `fusion.json` | ~1 KB | ✅ | 融合统计量（logit 均值/方差）、阈值、margin、OOF 指标 |
| **`xgb_model.ubj`** | ~0.75 MB | ✅ | **XGBoost 原生 UBJ 模型（首选）**，跨平台/跨版本最稳 |
| **`xgb_scaler.json`** | ~9 KB | ✅ | **StandardScaler 参数 + 特征配置（纯 JSON）**，与 UBJ 配套 |
| `xgb_pipeline.joblib` | ~0.8 MB | ✅ | 旧格式（模型 + scaler + 配置），仅作兼容回退 |
| `ref_sweep.wav` | ~0.3 MB | ✅ | 参考扫频，对齐类特征必需 |
| `w2v_ckpt.pt` | ~360 MB | ❌（体积超限） | wav2vec2 权重 |

> **为什么同时保留两种格式**：`joblib/pickle` 里嵌的是 XGBoost 的二进制 buffer，
> 跨平台（Linux 训练 → Windows 推理）时曾出现 `XGBoostError: input stream corrupted`。
> 推理端现在**优先读 `xgb_model.ubj` + `xgb_scaler.json`**（纯官方格式 + 纯 JSON），
> 读不到才回退 `xgb_pipeline.joblib`。两者实测逐位一致（2000 条随机样本最大概率差 0）。
> 详见 [`docs/MODEL_FORMAT.md`](../docs/MODEL_FORMAT.md)。

## 重新生成可移植产物

已有 `xgb_pipeline.joblib` 时，无需重训：

```bash
python scripts/export_portable.py --models models
```

会写出 `xgb_model.ubj` / `xgb_scaler.json` 并自动做等价性校验。

## 获取 `w2v_ckpt.pt`

三选一：

1. **自己训练**（推荐，数据不出本地）
   ```bash
   python scripts/train_production.py --data data/raw --out models \
       --oof-folds 3 --oof-epochs 35 --final-epochs 60 --batch-size 128
   ```
   会同时刷新 `xgb_pipeline.joblib` / `fusion.json` / `detector.json`。

2. **Git LFS**（需要仓库开启 LFS）
   ```bash
   git lfs install
   git lfs track "models/*.pt"
   git add .gitattributes models/w2v_ckpt.pt && git commit -m "weights" && git push
   ```

3. **Release 附件**：把 `w2v_ckpt.pt` 上传到 GitHub Release，下载后放到 `models/` 即可。

> 校验：`sha256sum models/w2v_ckpt.pt`，与 Release 页面公布的值比对。
