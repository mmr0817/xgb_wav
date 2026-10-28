# Docker 部署（Windows / macOS / Linux）

> **一句话**：把「Python + PyTorch + 模型」全部封进一个 Linux 镜像，
> 宿主机什么都不用装。**Windows ARM64（Apple Silicon 上的 Parallels）也能跑**，
> 因为 Linux 容器里的 torch 同时提供 amd64 与 arm64 轮子——这正好绕开
> 「PyTorch 没有 Windows ARM64 轮子」的死结。

## 0. 为什么用 Docker

| 痛点 | Docker 怎么解决 |
|---|---|
| Windows ARM64 装不上 torch | 容器是 **Linux**，torch 有 `linux/arm64` 轮子 |
| 装依赖要 2–3 GB、还常失败 | 依赖全在镜像里，一条 `docker run` 即可 |
| 产线机器断网 | 模型权重已打进镜像，运行期**零下载** |
| 不同电脑环境不一致 | 镜像固定 Python 3.12 + 固定依赖版本 |

> 容器只提供**命令行**（输出 CSV / 指标报表）。图形界面请用
> macOS 原生 `python run_gui.py` 或 Windows x64 的 `xgb_wav.exe`。

## 1. 装 Docker Desktop（Windows）

1. 下载 <https://www.docker.com/products/docker-desktop/> 安装（Win10/11 64 位）。
2. 首次启动会装 **WSL2** 后端，按提示重启一次。
3. 托盘出现小鲸鱼图标且**不再动画**，说明引擎就绪。
4. Settings → Resources → File sharing：勾上你要挂载的盘符（如 `D:`）。

> Apple Silicon 的 Mac 上装的是 Docker Desktop for Mac，跑的是 **linux/arm64** 容器，同样可用。

## 2. 方式 A（推荐）：拉现成镜像

镜像由 GitHub Actions 在 x64+arm64 上构建，自动按你的机器架构选择：

```bash
docker pull ghcr.io/mmr0817/xgb_wav:latest
```

> 若提示 `denied`：打开 GitHub → 你的头像 → **Packages** → `xgb_wav`
> → *Package settings* → *Change visibility* → **Public**（或用
> `docker login ghcr.io -u mmr0817` 登录一次）。

### 跑一次检测

Windows **cmd**：

```bat
mkdir outputs
docker run --rm -v "D:\wavs:/data" -v "%CD%\outputs:/out" ^
  ghcr.io/mmr0817/xgb_wav:latest --input /data --out-dir /out
```

PowerShell（注意 `${PWD}`）：

```powershell
docker run --rm -v "D:\wavs:/data" -v "${PWD}\outputs:/out" `
  ghcr.io/mmr0817/xgb_wav:latest --input /data --out-dir /out
```

macOS / Linux：

```bash
mkdir -p outputs
docker run --rm -v "$PWD/data:/data" -v "$PWD/outputs:/out" \
  ghcr.io/mmr0817/xgb_wav:latest --input /data --out-dir /out
```

结果：`outputs/predictions.csv`（+ `REPORT.md` 若做了带标签评测）。

### 或者双击脚本

`docker_run.bat`：双击 → 把 wav 文件夹拖进窗口 → 回车，结果落在 `outputs\`。
（默认用 `xgb_wav:1.0`；想用 GHCR 镜像就把它里面的 `set "IMAGE=..."` 改成
`ghcr.io/mmr0817/xgb_wav:latest`）

## 3. 方式 B：本地构建

```bat
:: Windows：双击 docker_build.bat，或
docker build -t xgb_wav:1.0 .
```

```bash
# macOS / Linux
bash docker_build.sh
```

- 镜像约 **3.9 GB**（实测；torch 769 MB + llvmlite 173 MB + 模型权重 362 MB），首次构建 10–20 分钟。
- 需要能访问 Docker Hub（拉 `python:3.12-slim`）。国内网络请在 Docker Desktop →
  Settings → Docker Engine 里加 registry mirror，例如：

```json
{
  "registry-mirrors": ["https://docker.m.daocloud.io"]
}
```

- 也可以直接换基础镜像（不需要改 Dockerfile）：

```bash
docker build --build-arg BASE_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
             -t xgb_wav:1.0 .
```

- 构建时 amd64 会自动走 PyTorch 官方 CPU 源（`download.pytorch.org/whl/cpu`，约 196 MB），
  arm64 走 PyPI 的 CPU 轮子；两者都不需要显卡。


## 4. 常用用法

| 场景 | 命令（IMAGE = `ghcr.io/mmr0817/xgb_wav:latest` 或 `xgb_wav:1.0`） |
|---|---|
| 整个目录 | `docker run --rm -v "D:\wavs:/data" -v "%CD%\outputs:/out" IMAGE --input /data --out-dir /out` |
| 单个文件 | `--input /data/xxx.wav --out-dir /out` |
| 带标签评测（指标+阈值扫描） | `--data /data --holdout --sweep --out-dir /out`（`/data` 下需有 `good/`、`bad/`） |
| 临时改阈值 | 追加 `--threshold 0.5` |
| 环境自检 | `docker run --rm --entrypoint python IMAGE doctor.py` |
| 看帮助 | `docker run --rm IMAGE --help` |
| 交互式 shell | `docker run --rm -it --entrypoint bash IMAGE` |
| 用 GPU | 不支持（镜像为 CPU 版，产线也不需要） |

## 5. 卷挂载（Windows 必看）

| 规则 | 说明 |
|---|---|
| 用**绝对路径** | `-v "D:\wavs:/data"` ✅ ；`-v .\wavs:/data` 可能被解析到 WSL 里 |
| 盘符要共享 | Docker Desktop → Settings → Resources → File sharing 勾选该盘 |
| 输出目录先建好 | 宿主机 `outputs\` 不存在时 Docker 会建目录但**属主是 root**，Windows 下一般无感 |
| 中文/空格路径 | 可用，但建议英文短路径（`D:\wavs`），避免转义麻烦 |
| 只读挂载输入 | 想更安全可加 `:ro`：`-v "D:\wavs:/data:ro"` |
| **Linux 宿主机** | 容器内是 uid 10001，若 `outputs/` 属主不是你，会报 `Permission denied`；加 `-u $(id -u):$(id -g)` 即可（Windows/macOS 无需） |

## 6. 性能参考（镜像内实测）

`docker run` 在 x86_64、4 线程下的实测值（8 条 8 kHz wav）：

| 项目 | 数值 |
|---|---|
| 容器启动 + 模型加载 + 8 条推理 | **5.0 s**（其中推理 2.8 s） |
| 单条推理 | **约 350 ms/条** |
| 镜像体积 | **3.84 GB**（torch 769 MB、llvmlite 173 MB、权重 362 MB） |

> 同机用 CUDA 版 torch（`--device cpu`）实测约 100 ms/条；容器内是纯 CPU wheel，
> 没有 MKL 加速，慢 3 倍左右。调大 `OMP_NUM_THREADS` 提升很小（实测 346 ms）。
> 产线按 ~3 条/秒 估算即可；几千条几分钟跑完。

ARM64 容器（Apple Silicon / Windows on ARM）通常再慢 2–4 倍，实测为准。

## 6b. 容器内自检（可选）

```bash
docker run --rm --entrypoint python xgb_wav:1.0 doctor.py
```

实测输出：Python 3.12.12 / xgboost 3.4.1（模型版本 3.4.1）/ 端到端推理成功 / 全部检查通过。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `docker: command not found` / 连接失败 | Docker Desktop 没启动；等托盘鲸鱼图标稳定 |
| `pull access denied for ghcr.io/...` | 包不是 public：GitHub → Packages → xgb_wav → settings → Change visibility → Public；或 `docker login ghcr.io` |
| 拉基础镜像超时（国内） | 配 registry mirror（见 §3），或改用方式 A 的 GHCR 镜像 |
| `bind source path does not exist` | 路径写错或盘符没共享 |
| CSV 没生成 | 看容器输出；常见是 `--input /data` 目录里没有 `.wav`（只认 `.wav/.WAV`） |
| 中文乱码 | 容器内输出为 UTF-8；Windows 用 `chcp 65001` 或直接看 CSV（UTF-8-BOM，Excel 正常） |
| 想改阈值/换模型 | 挂载自己的模型目录并加 `--models /mymodels`：`-v "D:\mymodels:/mymodels:ro"` |
| 镜像太大 | 权重占 361 MB；如只想要轻量镜像，可把 `COPY models/` 删掉、改为运行时挂载 `-v models:/app/models` |

## 8. 和 exe 方案怎么选

| | Docker 镜像 | Windows exe |
|---|---|---|
| 适用机器 | 任何装了 Docker 的机器（含 ARM64） | 仅 **x64** Windows |
| 目标机要装什么 | Docker Desktop（~1 GB） | 什么都不用装 |
| 体积 | ~2.3 GB（镜像） | ~1.5–2 GB（目录） |
| 图形界面 | ❌ 只有命令行 | ✅ 有 |
| 构建难度 | CI 一键 / 本地一条命令 | 必须 x64 Windows 或 CI |

> 结论：**产线/临时验证用 Docker（尤其你是 ARM64）**；
> 需要给人双击就用的话，用 GitHub Actions 构建 exe。


## 9. 离线镜像包（拷到 Mac / 无网机器上用）

镜像已打包成可直接导入的文件（**不需要联网、不需要 GitHub**）：

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `xgb_wav_docker_amd64.tar.gz` | **1.2 GB** | `59d9aeef7af34001ef9d9ea2786bf98066b59650022998f617495a1e7ffcee7e` |
| `xgb_wav_docker_amd64.tar.gz.sha256` | 校验文件 | — |

> 架构是 **linux/amd64**（本机 x86_64，无法交叉构建 arm64）。

### macOS 上使用（含 Apple Silicon）

1. 安装并启动 **Docker Desktop for Mac**，等托盘鲸鱼图标稳定。
2. 把 tar.gz 与仓库里的 `docker_load.command` 放同一目录，**双击 `docker_load.command`**
   （首次被 Gatekeeper 拦 → 右键 → 打开）。等价于：

   ```bash
   docker load -i xgb_wav_docker_amd64.tar.gz
   docker run --rm --entrypoint python xgb_wav:1.0 doctor.py
   ```

3. 批量检测：

   ```bash
   mkdir -p ~/xgb_out
   docker run --rm -v "$HOME/wavs:/data:ro" -v "$HOME/xgb_out:/out" \
     xgb_wav:1.0 --input /data --out-dir /out
   ```

**Apple Silicon 提速**（该镜像为 amd64，需要模拟运行）：

| 方式 | 操作 | 效果 |
|---|---|---|
| 开 Rosetta 模拟 | Docker Desktop → Settings → General → 勾选 *Use Rosetta for x86/amd64 emulation* | 明显更快（先试这个） |
| 构建原生 arm64 | 仓库根目录 `bash docker_build.sh`（10–20 分钟） | 最快，覆盖同名 `xgb_wav:1.0` |

也可显式声明平台消除警告：`docker run --platform linux/amd64 ...`

### 校验完整性

```bash
shasum -a 256 xgb_wav_docker_amd64.tar.gz   # macOS
sha256sum xgb_wav_docker_amd64.tar.gz       # Linux
```

### 打包机上的实测记录

```
docker load -i xgb_wav_docker_amd64.tar.gz  → Loaded image: xgb_wav:1.0（5.8 s）
docker image inspect                         → linux/amd64, 3.84 GB
8 条 wav：346 ms/条，bad 3 / good 5（与宿主机一致）
204 条留出集：R=1.0000 / P=0.9118 / 误杀 3-173 / AUC 0.9983（与训练标定一致）
容器内 doctor.py：全部检查通过
```

