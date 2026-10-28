# syntax=docker/dockerfile:1
# xgb_wav 推理镜像（CPU 版，支持 linux/amd64 与 linux/arm64）
#
# 构建：
#   docker build -t xgb_wav:1.0 .
# 运行（把 wav 目录挂到 /data，CSV 输出到宿主机 ./outputs）：
#   docker run --rm -v "$PWD/data:/data" -v "$PWD/outputs:/out" xgb_wav:1.0 \
#          --input /data --out-dir /out
#
# 说明：
#   - 模型权重已打进镜像（models/w2v_ckpt.pt），运行时**完全离线**，不下载任何东西。
#   - 只提供命令行；图形界面请用 macOS 原生或 Windows x64 的 exe。
# 基础镜像可覆盖（国内/受限网络可换成自己的镜像源）：
#   docker build --build-arg BASE_IMAGE=ghcr.io/astral-sh/uv:python3.11-bookworm-slim -t xgb_wav:1.0 .
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=8 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_HOME=/opt/torch \
    PYTHONPATH=/app/src

# libsndfile: soundfile/librosa 读 wav；libgomp1: xgboost/torch 的 OpenMP
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 依赖层（尽量利用缓存）----------
COPY requirements.txt ./
ARG TARGETARCH=amd64
# 顺序很重要：先装 requirements 里除 torch/torchaudio 之外的部分，
# 最后再装 CPU 版 torch——否则 pip 会把已装的 CPU torch 换成 PyPI 的 CUDA 版，
# 额外拖进 nvidia_* 依赖（+4 GB）。
# 注意：这一段不能写 # 注释在续行里（会被 shell 当注释吞掉后半行）。
RUN grep -viE '^(torch|torchaudio)([[:space:]]*$|[<>=!~])' requirements.txt > /tmp/req-notorch.txt \
 && ls /tmp/req-notorch.txt && cat /tmp/req-notorch.txt \
 && python -m pip install -r /tmp/req-notorch.txt \
 && case "$TARGETARCH" in \
      amd64) IDX="--index-url https://download.pytorch.org/whl/cpu" ;; \
      *)     IDX="" ;; \
    esac \
 && python -m pip install $IDX torch torchaudio \
 && python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

# ---------- 代码与模型 ----------
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY test.py run_gui.py doctor.py README.md LICENSE ./
COPY models/ ./models/
COPY README.md LICENSE ./

# 非 root 运行；输出目录给写入权限
RUN useradd -m -u 10001 appuser \
 && mkdir -p /out \
 && chown -R appuser:appuser /app /out
USER appuser

VOLUME ["/data", "/out"]

ENTRYPOINT ["python", "test.py"]
CMD ["--help"]
