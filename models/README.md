# models/ 目录说明

| 文件 | 体积 | 是否入库 | 说明 |
|---|---|---|---|
| `detector.json` | <1 KB | ✅ | 推理入口：指明下面各文件的相对路径与预处理参数 |
| `fusion.json` | ~1 KB | ✅ | 融合统计量（logit 均值/方差）、阈值、margin、OOF 指标 |
| `xgb_pipeline.joblib` | ~0.5 MB | ✅ | XGB 模型 + StandardScaler + 特征配置 |
| `ref_sweep.wav` | ~0.3 MB | ✅ | 参考扫频，对齐类特征必需 |
| `w2v_ckpt.pt` | ~360 MB | ❌（体积超限） | wav2vec2 权重 |

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
