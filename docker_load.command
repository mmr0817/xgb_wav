#!/bin/bash
# ============================================================
#  macOS 双击运行：导入镜像 + 环境自检
#  用法：把 xgb_wav_docker_amd64.tar.gz 放在本文件同目录，然后双击本文件
#       （若提示“无法打开，因为来自身份不明的开发者” → 右键 → 打开）
# ============================================================
set -e
cd "$(dirname "$0")"

TAR="${1:-${TAR:-xgb_wav_docker_amd64.tar.gz}}"
OUT="${OUT:-$PWD/outputs}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[错误] 未找到 docker。请先安装并启动 Docker Desktop：https://www.docker.com/products/docker-desktop/"
  read -n 1 -s -r -p "按任意键关闭…"; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[错误] Docker 引擎未运行。请先打开 Docker Desktop，等托盘图标不再转动。"
  read -n 1 -s -r -p "按任意键关闭…"; exit 1
fi
if [ ! -f "$TAR" ]; then
  echo "[错误] 找不到镜像包：$TAR"
  echo "       请把 xgb_wav_docker_amd64.tar.gz 和本文件放在同一目录。"
  read -n 1 -s -r -p "按任意键关闭…"; exit 1
fi

echo "== 1/3 导入镜像（约 1.2 GB，需要 10–60 秒）"
docker load -i "$TAR"

echo
echo "== 2/3 镜像信息"
docker image inspect xgb_wav:1.0 --format 'os/arch = {{.Os}}/{{.Architecture}} | size = {{.Size}} bytes'
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  echo "   提示：本机是 Apple Silicon，而这个镜像是 linux/amd64，会走模拟运行。"
  echo "         想要更快，可在 Docker Desktop → Settings → General 勾选"
  echo "         'Use Rosetta for x86/amd64 emulation'；或在本仓库执行 bash docker_build.sh 构建原生 arm64 镜像。"
fi

echo
echo "== 3/3 容器内自检（模型加载 + 端到端推理）"
docker run --rm --entrypoint python xgb_wav:1.0 doctor.py | tail -20

echo
echo "============================================================"
echo " 完成。批量检测示例（把 /Users/you/wavs 换成你的目录）："
echo "   mkdir -p \"$OUT\""
echo "   docker run --rm -v \"/Users/you/wavs:/data:ro\" -v \"$OUT:/out\" \\"
echo "     xgb_wav:1.0 --input /data --out-dir /out"
echo
echo " 结果：$OUT/predictions.csv"
echo "============================================================"
read -n 1 -s -r -p "按任意键关闭窗口" || true
