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

# 顺手把本包装进当前环境（让 `python -m xgb_wav.*` 和 `xgb-wav-gui` 可用）
pip install -e .
```

> 不想安装也可以：仓库根目录直接 `python run_gui.py`（脚本自己把 `src/` 加进 `sys.path`）。

依赖：`torch`/`torchaudio`（wav2vec2）、`xgboost`、`scikit-learn`、`librosa`、`soundfile`、
`numpy`、`scipy`、`pandas`、`joblib`、`pyyaml`。GPU 可选（自动检测，CPU 亦可跑）。

> wav2vec2 主干按 `models/detector.json` 中的 `w2v_arch` **本地构建**，再用自带 `w2v_ckpt.pt`
> 覆盖全部权重，因此**完全离线**：运行时不下载任何预训练权重（旧版会下载约 360 MB）。
> 已实测「清空 torch 缓存」与「联网」两条路径输出逐位一致。

## 使用

### 1) 图形界面

```bash
python run_gui.py              # 零安装直接跑（推荐）
# 或安装后： python -m xgb_wav.gui   /   xgb-wav-gui
# Windows 打包后：双击 xgb_wav.exe
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

### 4) 评测 / 回归自测（`test.py`）

用已训练权重直接跑评测，**不重新训练**：

```bash
# 推荐：只评训练时的留出标定集（复现同一划分，指标最可信）
python test.py --data ../rs20262/data/raw --holdout --sweep

# 全量带标签目录评测（与训练集重叠 → 指标偏乐观，仅作回归自测）
python test.py --data ../rs20262/data/raw --sweep

# 无标签：只推理并导出 CSV（文件或目录都行）
python test.py --input D:\wavs

# 产线门禁：召回低于 99.5% 时进程返回 1
python test.py --data <带good/bad的目录> --holdout --min-recall 0.995
```

产物默认写到 `outputs/eval/`（`--out-dir` 可改）：

| 文件 | 内容 |
|---|---|
| `predictions.csv` | 每条音频的融合分、两路概率、预测、真值、是否正确 |
| `metrics.json` | 主指标 + 混淆矩阵 + 分通道 + 两路成员单独表现 |
| `REPORT.md` | Markdown 报表（含阈值扫描：F1 最优 / 零漏检边界 / margin 余量） |

`--threshold` 可临时覆盖阈值做 A/B，`--limit N` 快速冒烟，`--device cpu` 强制 CPU。

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
2. 分层切出 **10% 留出标定集**（`--calib-frac`，默认 0.1），其余 90% 训练；
3. 在留出集上算 logit 标准化统计量与阈值（`min(留出 bad) − margin`，默认 margin=0.2）；
4. （可选）`--oof-folds >1` 改用 K 折 OOF 标定——更慢，且折模型偏弱时阈值过于保守；
5. 写出 `xgb_pipeline.joblib` / `w2v_ckpt.pt` / `fusion.json` / `detector.json`。

> 单卡 24 GB 时 batch 128 足够；显存不足会自动 OOM，请调小 `--batch-size`。
> 2031 条样本、留出标定 + 最终训练约 12 分钟（RTX PRO 4000 级别）。

## 测试

```bash
# 1) 单元/冒烟测试（无显示器也能跑）
PYTHONPATH=src python -m pytest tests/test_smoke.py -q

# 2) 用已训练权重评测（见上文「评测 / 回归自测」）
python test.py --data ../rs20262/data/raw --holdout --sweep

# 3) GUI 冒烟测试（需要图形环境；无 DISPLAY 会自动跳过）
python tests/gui_smoke.py --input /tmp/demo_wav
#   Linux 无桌面但有 X 时：DISPLAY=:1 python tests/gui_smoke.py
```

GUI 冒烟测试会真实创建窗口并模拟「选择文件夹 → 开始检测」，
校验模型加载、按钮状态、表格行数、自动导出 CSV 与错误恢复，并把截图写到
`outputs/gui_smoke/`。当前结果：**16/16 项通过**。

## 打包 Windows exe

PyInstaller **不支持跨平台交叉编译**，exe 必须在 Windows 上构建。完整图文步骤见
**[`docs/BUILD_WINDOWS.md`](docs/BUILD_WINDOWS.md)**。

### Windows 快速开始（推荐，全部双击即可）

| 步骤 | 双击 | 作用 |
|---|---|---|
| 1 | `windows_setup.bat` | 自动找 Python → 建 `.venv` → 装依赖（失败换清华镜像）→ 跑自检 |
| 2 | `run_gui.bat` | 打开图形界面（出错时窗口停留，能看到报错） |
| 3 | `run_cli.bat` | 把 wav 文件夹拖进来回车，批量检测并导出 CSV 到 `outputs\cli\` |
| 排错 | `python doctor.py` | 生成 `doctor_report.txt`（Python 位数、依赖、控制台编码、模型完整性、端到端推理） |

> 英文版 Windows 的控制台代码页（cp437）无法编码中文，旧版会直接抛
> `UnicodeEncodeError` 退出。现在所有入口都会把控制台切到 UTF-8 并做
> `errors="replace"` 兜底，不会再因为一个字符崩掉。

### 用 GitHub Actions 构建 exe（没有 x64 Windows 机器时首选）

> ⚠️ **Apple Silicon 上的 Parallels 跑的是 Windows 11 ARM64，而 PyTorch 没有
> `win_arm64` 轮子**（PyPI 上 2.6–2.9 均为 0 个），`pip install torch` 必然失败。
> 所以「Mac + Parallels」无法直接跑本项目，请用下面任一方式：
> ① 在 macOS 上原生跑（`python run_gui.py`，torch 有 macOS arm64 轮子）；
> ② 用本仓库的 Actions 在 GitHub 的 x64 Windows 上打包 exe。

1. 先上传权重到 Release（只需一次）：GitHub → Releases → *Draft a new release* →
   把本地 `models/w2v_ckpt.pt`（361 MB）拖进去 → Publish。
2. 打开 Actions → **build-windows-exe** → *Run workflow*。
3. 跑完在 Artifact 里下载 `xgb_wav-windows-x64.zip`，解压即用（内含 `xgb_wav.exe` + `_internal` + `models`）。

工作流文件：`.github/workflows/build-windows-exe.yml`（x64 + CPU torch + 自检 + 打包 + 上传产物）。

### 本地打包（有 x64 Windows 时）

```bat
cd /d D:\xgb_wav
scripts\build_exe.bat
```

脚本会依次：检查 Python → 装 CPU 版 torch 等依赖 → 校验 `models\` 5 个文件 →
**先跑一遍 CLI 自检** → PyInstaller 打包 → 把 `models\` 复制到 exe 同级。

产物 `dist\xgb_wav\`（`xgb_wav.exe` + `_internal\` + `models\`），**整个目录一起分发**，
目标机器无需 Python、无需联网。体积约 1.2–2.0 GB（torch 占大头）。

常用开关：`set SKIP_PIP=1`（跳过装依赖）、`set NO_SELFTEST=1`（跳过自检）、
`set CUDA=1`（有 NVIDIA 卡时装 CUDA 版 torch）。

## Docker（推荐：ARM64 Windows 也能跑）

> Linux 容器里的 PyTorch 同时有 amd64/arm64 轮子，**绕开「Windows ARM64 装不上 torch」的问题**。
> 镜像含权重，运行期完全离线；只提供命令行（GUI 用 macOS 原生或 Windows exe）。

```bash
docker pull ghcr.io/mmr0817/xgb_wav:latest
docker run --rm -v "D:/wavs:/data" -v "$PWD/outputs:/out" \
  ghcr.io/mmr0817/xgb_wav:latest --input /data --out-dir /out
```

| 需求 | 用哪个 |
|---|---|
| 一键跑（Windows/macOS，含 ARM） | `docker_run.bat`（拖入 wav 文件夹）/ `docker run ...` |
| 自己构建镜像 | `docker_build.bat`（Windows）/ `bash docker_build.sh` |
| 详细说明（挂载、镜像加速、GHCR 权限、性能） | [`docs/DOCKER.md`](docs/DOCKER.md) |

镜像约 2.3 GB（含 361 MB 权重）；纯 CPU 约 100 ms/条。

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

