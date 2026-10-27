# 上传到 GitHub（仓库名 `xgb_wav`，账号 `mmr0817`）

## 当前状态（已排查）

| 检查项 | 结果 |
|---|---|
| SSH 私钥 | `~/.ssh/id_ed25519_github_mmr0817` 存在 |
| `~/.ssh/config` | 之前缺失 → 已补上 `Host github.com / IdentityFile ...` |
| `ssh -T git@github.com` | ✅ `Hi mmr0817! You've successfully authenticated` |
| 账号名 | **`mmr0817`**（原先填的 `mmr_0817` 多了下划线，GitHub 不允许） |
| 远程地址 | 已修正为 `git@github.com:mmr0817/xgb_wav.git` |
| GitHub 仓库 | ❌ **尚不存在**（API 返回 404）→ 创建仓库需要 Token 或网页操作 |

## 只差最后一步：创建仓库

### 方式 A：网页（10 秒）

打开 → https://github.com/new?name=xgb_wav&owner=mmr0817&visibility=public

- Repository name: `xgb_wav`
- **不要**勾选 Add a README / .gitignore / license（本地已有）
- 点 Create repository

然后推送：

```bash
cd /home/user/文档/xgb_wav
bash scripts/push_to_github.sh          # 或 git push -u origin main
```

### 方式 B：一条命令（有 Token）

```bash
cd /home/user/文档/xgb_wav
GH_TOKEN=ghp_xxx bash scripts/push_to_github.sh    # 自动建仓 + 推送
```

Token 需要 `repo` 权限（fine-grained 则勾选 Administration: Read and write + Contents: Read and write）。

### 方式 C：gh CLI（若已安装并登录）

```bash
gh repo create mmr0817/xgb_wav --public --source=. --remote=origin --push
```

## 权重 `models/w2v_ckpt.pt`（361 MB）

已被 `.gitignore` 排除，不能直接 push（GitHub 单文件上限 100 MB）。二选一：

```bash
# LFS
git lfs install && git lfs track "models/*.pt"
git add .gitattributes models/w2v_ckpt.pt && git commit -m "weights" && git push

# 或 Release 附件
gh release create v1.0.0 models/w2v_ckpt.pt --title "v1.0.0 weights"
```

## 推完之后

```bash
git clone git@github.com:mmr0817/xgb_wav.git && cd xgb_wav
pip install -r requirements.txt
# 放入 models/w2v_ckpt.pt（LFS 或 Release 下载）
python -m xgb_wav.cli --models models --input <wav目录> --output out.csv
```
