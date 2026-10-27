#!/usr/bin/env bash
# 一键：创建 GitHub 仓库（可选，需 GH_TOKEN）+ 推送 xgb_wav
#
# 用法：
#   1) 有 Token（具 repo 权限）时，自动建仓并推送：
#        GH_TOKEN=ghp_xxx bash scripts/push_to_github.sh
#   2) 已在网页建好仓库，只需推送：
#        bash scripts/push_to_github.sh
#
# 说明：仓库默认 Public；权重 w2v_ckpt.pt 已被 .gitignore 排除（361 MB 超 GitHub 单文件限制），
#       请用 Git LFS 或 Release 附件分发，见 docs/UPLOAD.md。

set -euo pipefail

OWNER="${OWNER:-mmr0817}"
REPO="${REPO:-xgb_wav}"
REMOTE_URL="git@github.com:${OWNER}/${REPO}.git"

cd "$(dirname "$0")/.."

echo "== 仓库: ${OWNER}/${REPO}"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE_URL"
git remote set-url origin "$REMOTE_URL"

# ---- 1) 仓库是否存在 ----
code=$(curl -s -o /dev/null -w '%{http_code}' "https://api.github.com/repos/${OWNER}/${REPO}")
if [ "$code" = "404" ]; then
  if [ -n "${GH_TOKEN:-}" ]; then
    echo "== 仓库不存在，使用 Token 创建…"
    resp=$(curl -s -X POST \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      -d "{\"name\":\"${REPO}\",\"private\":false,\"description\":\"工业音频坏件检测（XGB + wav2vec2 融合）\",\"has_issues\":true,\"has_wiki\":false}" \
      "https://api.github.com/user/repos")
    echo "$resp" | head -c 300; echo
    code=$(curl -s -o /dev/null -w '%{http_code}' "https://api.github.com/repos/${OWNER}/${REPO}")
    [ "$code" = "200" ] || { echo "[失败] 建仓失败，请检查 Token 权限(repo) 或用户名"; exit 1; }
    echo "== 仓库已创建"
  else
    echo "[需要先建仓库] 打开下面链接点 Create（不要勾选 README/gitignore/license）："
    echo "    https://github.com/new?name=${REPO}&owner=${OWNER}&visibility=public"
    echo "或者提供 Token 后重跑： GH_TOKEN=ghp_xxx bash scripts/push_to_github.sh"
    exit 2
  fi
fi

# ---- 2) 推送 ----
echo "== 推送 main…"
git branch -M main
git push -u origin main
echo
echo "== 完成：https://github.com/${OWNER}/${REPO}"
echo "   权重分发（二选一）："
echo "     git lfs install && git lfs track 'models/*.pt' && git add .gitattributes models/w2v_ckpt.pt && git commit -m weights && git push"
echo "     或 gh release create v1.0.0 models/w2v_ckpt.pt"
