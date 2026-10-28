#!/usr/bin/env bash
# 在 macOS / Linux 上构建镜像（ARM Mac 会产出 arm64 镜像）
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-xgb_wav:1.0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker 未安装"
  exit 1
fi
if [ ! -f models/w2v_ckpt.pt ]; then
  echo "[ERROR] 缺少 models/w2v_ckpt.pt（请先解压完整包）"
  exit 1
fi

echo "== 构建 $IMAGE（首次约 5-15 分钟）"
docker build -t "$IMAGE" .

echo
echo "== 完成。试跑："
echo "  docker run --rm -v \"\$PWD/data:/data\" -v \"\$PWD/outputs:/out\" $IMAGE --input /data --out-dir /out"
