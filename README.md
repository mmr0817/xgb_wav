# xgb_wav · 工业音频坏件检测

XGBoost + wav2vec2 双模型融合的 wav 坏件检测**桌面应用**（Windows / macOS / Linux）。
纯本地运行、**完全离线**，不依赖网络与云端。

<img src="assets/app.png" width="96" alt="icon">

## 安装（Windows，推荐）

1. 下载 `xgb_wav_Setup_1.0.0.exe`
2. 双击安装（默认装到 `%LOCALAPPDATA%\Programs\xgb_wav`，**无需管理员权限**）
3. 开始菜单 / 桌面快捷方式启动

卸载：设置 → 应用 → xgb_wav。

## 从源码运行

```bash
# 需要 Python 3.12 或 3.13（64 位）
python -m pip install -r requirements.txt
python main.py            # 打开图形界面
```

## 界面用法

| 区域 | 说明 |
|---|---|
| 模型目录 | 默认 `./models`，随程序分发 |
| 输入 | 选择一批 wav 文件或整个文件夹 |
| **判定阈值** | 滑块 / 输入框，**实时重判**（不重新推理），自动保存到 `settings.json` |
| **自动定阈…** | 按**本批**分数分布给候选阈值（未知 / 全坏 / 全好三种假设），选中后一键套用 |
| 开始检测 | 显示进度；完成后表格按 bad（红）/ good（绿）着色 |
| 导出 CSV | 按**当前阈值**导出，UTF-8-BOM，Excel 直接可读 |
| **省电模式** | 勾选后强制 CPU + 单线程 + 限速，速度 ≈ **1/4**，占用/发热更低；状态记入 `settings.json`（CLI 用 `--low-power`） |

## 阈值怎么调（误杀率偏高时）

融合分与判定规则：

```
score = mean( (logit(p_xgb) - mu_xgb)/sd_xgb , (logit(p_w2v) - mu_w2v)/sd_w2v )
bad  <=>  score >= threshold
```

- **调高阈值**：误杀↓、漏检↑（保守）
- **调低阈值**：漏检↓、误杀↑（激进）
- 默认 `0.3135`（训练时标定：留出集召回 100%、误杀 1.7%）

真实产线误杀偏高时：把阈值从 0.31 往上拖（如 0.5 / 0.6），
表格里 bad/good 数量会**实时**变化，选一个可接受的点再「导出 CSV」。
设置写入程序目录的 `settings.json`，下次启动自动沿用；「恢复模型默认」可还原。

## 按批自优化阈值（自动定阈）

同一批数据的两类风险并不对称：整批全是好件时怕**误杀**，整批全是坏件时怕**漏检**。
点「**自动定阈…**」，按本批分数分布给出候选阈值（只列候选，点「应用所选阈值」才真正改阈值）：

| 模式 | 怎么算 | 用在什么时候 |
|---|---|---|
| 未知（自动找分界） | 最大间隙 / Otsu 双峰二分 | 好坏混杂，想快速定一个分界 |
| 本批全是坏件（防漏检） | 「命中 100% / 99.5% / 95%」对应的**最高**可行阈值 | 确认这批是坏件，尽量全拦住 |
| 本批全是好件（防误杀） | 「放行 100% / 99.5% / 95%」对应的**最低**可行阈值 | 确认这批是好件，尽量不误杀 |

「未知」模式只有在**两侧都成规模（各 ≥10%）、分界处确有低谷、且离模型标定值 ≤0.30** 时才推荐数据分界，
否则推荐沿用模型标定值；离标定值太远的数据分界会**标红**——那多半是**类别内部**的次级分界，套用会把坏件放走。

示例（`123.zip`，70 条、实际全是坏件）：

| 候选 | 阈值 | bad | good | 说明 |
|---|---|---|---|---|
| 模型标定值（推荐） | 0.3135 | 66 | 4 | 沿用训练标定 |
| 最大间隙分界（红） | 2.4334 | 4 | 66 | 小的一侧仅 4 条，像整批同类；离标定值 2.12 |
| 本批全坏·全部命中 | -0.0111 | 70 | 0 | 取本批最小分，漏 0 条（误杀风险↑） |

命令行等价：`--auto-threshold auto|all_bad|all_good`（结果写进 CSV 的 `threshold` 列，
再加 `--save-threshold` 才会写进 `settings.json`）。

## 命令行

```bash
python -m xgb_wav.cli --models models --input D:\wavs --output result.csv
python -m xgb_wav.cli --input D:\wavs --output result.csv --threshold 0.6 --save-threshold
python -m xgb_wav.cli --input D:\wavs --output result.csv --auto-threshold all_bad   # 按本批自动定阈
python -m xgb_wav.cli --input D:\wavs --output result.csv --low-power                # 省电模式
```

## 目录结构（运行时最小集）

```
xgb_wav/
├── main.py                 应用入口
├── src/xgb_wav/            核心代码（音频 / 特征 / 两路模型 / 融合 / 界面）
├── models/                 模型与配置（detector.json 为入口）
├── assets/app.ico          图标
├── requirements.txt
├── VERSION
└── LICENSE
```

构建 / 训练 / 测试 / 打包脚本单独放在发布包的 `tools/` 目录（与 `xgb_wav/` 同级），
本仓库只包含运行集。

## 模型与指标

| 项目 | 值 |
|---|---|
| 融合方式 | 两路 logit 标准化后取均值 |
| 默认阈值 | 0.3135（margin 0.2） |
| 留出集（204 条） | 召回 **1.0000**、精度 0.9118、误杀 3/173（1.7%）、AUC 0.9983 |

`models/detector.json` 指明各文件；`xgb_model.ubj` + `xgb_scaler.json` 为可移植格式
（跨平台加载稳定），`xgb_pipeline.joblib` 仅作兼容回退。

## 许可

MIT
