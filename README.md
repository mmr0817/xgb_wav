# xgb_wav · 工业音频坏件检测

用**新威测量特征 + XGBoost** 与 **wav2vec2 端到端** 两路判别器做分数融合，
面向产线音频的 **good / bad** 二分类，并按「**宁可误杀、不可放过**」标定阈值。

- 输入：任意 wav（44.1 kHz 立体声扫频录音，2–2.5 s）
- 输出：判定结果 + 融合分 + 两路概率 → **CSV**（GUI 一键导出）
- 形态：Python 库 / 命令行 / **图形界面（tkinter）** / Windows 单文件 exe

## 性能（留出集，46 条 bad / 259 条 good）

| 方法 | 召回(R) | 精度(P) | F1 | 误杀 | AUC |
|---|---|---|---|---|---|
| XGB（新威特征） | 0.9565 | 0.7458 | 0.8381 | 15/261 | 0.9948 |
| wav2vec2 端到端 | 0.9783 | 0.8491 | 0.9091 | 8/261 | 0.9980 |
| **XGB + wav2vec2 融合（宁杀勿漏，margin 0.2）** | **1.0000** | **0.9020** | **0.9485** | **5/259（1.9%）** | **0.9996** |

> 阈值规则：`score = mean((logit(p_i) − μ_i) / σ_i)`，`bad ⇔ score ≥ threshold`；
> `threshold = min(验证集 bad 的融合分) − margin`。融合配置固化为 `models/fusion.json`。

## 目录结构

```
xgb_wav/
├── models/                   # 模型与配置（生产 checkpoint）
│   ├── detector.json         #   推理入口配置
│   ├── xgb_pipeline.joblib   #   XGB + StandardScaler + 特征配置
│   ├── w2v_ckpt.pt           #   wav2vec2 权重
│   ├── fusion.json           #   融合统计量 + 阈值 + margin
│   └── ref_sweep.wav         #   参考扫频（对齐类特征需要）
├── src/xgb_wav/
│   ├── features.py           # 114 维新威特征（MFCC/1-3倍频程/H(f)/THD/定时刻频段）
│   ├── featurize.py          # 特征提取 + 缓存
│   ├── model_xgb.py          # XGB 训练/推理
│   ├── model_w2v.py          # wav2vec2 + mean pooling + MLP 头
│   ├── fusion.py             # logit 标准化融合 + 宁杀勿漏判定
│   ├── pipeline.py           # FusionDetector（端到端）
│   ├── cli.py                # 命令行
│   └── gui.py                # 图形界面
├── scripts/
│   ├── train_production.py   # 全量训练 + K 折 OOF 标定阈值
│   ├── build_exe.bat         # Windows 打包（PyInstaller）
│   └── xgb_wav.spec          # PyInstaller 配置
├── tests/test_smoke.py
└── docs/METRICS.md
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`torch`/`torchaudio`（wav2vec2）、`xgboost`、`scikit-learn`、`librosa`、`soundfile`、
`numpy`、`scipy`、`pandas`、`joblib`、`pyyaml`。GPU 可选（自动检测，CPU 亦可跑）。

> wav2vec2 主干使用 `torchaudio.pipelines.WAV2VEC2_BASE`，首次运行会自动下载（约 360 MB），
> 之后缓存在 `~/.cache/torch/hub/checkpoints/`。

## 使用

### 1) 图形界面

```bash
python -m xgb_wav.gui          # 未打包时
# 或双击 xgb_wav.exe（打包后）
```

界面流程：选择模型目录（默认 `models/`）→ 选择 wav 文件或文件夹 → 开始检测 →
表格显示判定/分数/两路概率/通道/批次 → 自动导出 CSV（可另存）。

### 2) 命令行

```bash
python -m xgb_wav.cli --models models --input /path/to/wavs --output result.csv
# 只看模型信息
python -m xgb_wav.cli --models models --input dummy --info
```

### 3) Python API

```python
from xgb_wav import FusionDetector
det = FusionDetector.from_dir("models")           # device="auto"
df = det.predict_dir("/path/to/wavs")             # 或 det.predict_paths([...])
det.save_csv(df, "result.csv")
print(df[["file", "label", "score"]])
```

### CSV 字段

| 列 | 含义 |
|---|---|
| `path` / `file` | 文件绝对路径 / 文件名 |
| `label` | `good` / `bad` |
| `pred` | 0/1 |
| `score` | 融合分（越大越像 bad） |
| `xgb_proba` / `w2v_proba` | 两路 bad 概率 |
| `threshold` | 本次判定使用的阈值 |
| `channel` / `batch` | 从文件名解析的通道（P/F/none）与批次 |

## 训练（用你自己的数据）

数据按目录组织（只训练 good/bad；临界/瞬态噪声建议单独放，不参与训练）：

```
data/raw/
├── good/*.wav
└── bad/*.wav
```

```bash
python scripts/train_production.py \
    --data data/raw --out models \
    --oof-folds 3 --oof-epochs 35 --final-epochs 60 --batch-size 128
```

脚本会：
1. 提取 114 维特征（缓存到 `models/features_cache.npz`）；
2. 做 K 折 **OOF**，得到全量样本的「拿不到自己」的概率；
3. 用 OOF 统计量标定融合阈值（`min(OOF bad) − margin`，默认 margin=0.2）；
4. 用**全部数据**训练最终 XGB 与 wav2vec2；
5. 写出 `xgb_pipeline.joblib` / `w2v_ckpt.pt` / `fusion.json` / `detector.json`。

> 单卡 24 GB 时 batch 128 足够；显存不足会自动 OOM，请调小 `--batch-size`。
> 2031 条样本、3 折 OOF + 全量训练约 20 分钟（RTX PRO 4000 级别）。

## 打包 Windows exe

PyInstaller **不支持跨平台交叉编译**，exe 必须在 Windows 上构建：

```bat
:: 在 Windows 的仓库根目录执行（需先 pip install -r requirements.txt pyinstaller）
scripts\build_exe.bat
```

生成 `dist\xgb_wav\xgb_wav.exe`。首次启动会检查同目录下的 `models\`，
请把 `models\` 一并拷到 exe 同级目录（或打包时用 `--add-data` 内置，见 spec 注释）。

体积提示：torch + wav2vec2 会让产物达到 1.5–3 GB，属于正常现象。

## 模型与阈值说明

- **为什么不直接输出概率**：两路模型概率尺度不同，融合前各自做 logit 标准化；
  阈值定义在融合分空间，便于「宁杀勿漏」时只调一个数。
- **margin 的含义**：阈值取「验证集里最像 good 的那个 bad 的分数」再减去 margin，
  margin 越大越保守（召回↑、误杀↑）。默认 0.2 在留出集上得到 100% 召回、1.9% 误杀。
- **换产线/批次**：建议用新批次的留出集重新跑 `train_production.py`，
  或至少重算 `fusion.json` 里的 `logit_mean/logit_std` 与 `threshold`。

## 已知限制

1. 训练数据来自特定设备/批次，跨批次泛化会下降（按设备分组验证时 acc 会掉到 ~0.95）。
2. 极端边界样本（人工复核也无法确定的）建议移出训练集，否则会拉低上限。
3. wav2vec2 分支对采样率敏感：内部统一重采样到 16 kHz、峰值归一化、居中补齐到 2.5 s。

## License

MIT，见 [LICENSE](LICENSE)。

