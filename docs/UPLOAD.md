# 上传到 GitHub（仓库名 `xgb_wav`）

> 本机已完成 `git init` 与两次提交，但**没有你的 GitHub 凭据**，因此最后一步 push 需要你执行。

## 1. 在 GitHub 上创建空仓库

网页 → New repository：

- Repository name: **`xgb_wav`**
- 不要勾选 "Add a README / .gitignore / license"（本地已有）
- 建议 Public（开源）

## 2. 关联远程并推送

```bash
cd /home/user/文档/xgb_wav

git remote add origin git@github.com:<你的用户名>/xgb_wav.git
# 或 HTTPS： git remote add origin https://github.com/<你的用户名>/xgb_wav.git

git branch -M main
git push -u origin main
```

## 3. 处理 wav2vec2 权重（361 MB，超过 GitHub 单文件 100 MB 限制）

`models/w2v_ckpt.pt` 已被 `.gitignore` 排除。二选一：

### 方案 A：Git LFS（推荐）

```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes models/w2v_ckpt.pt
git commit -m "weights: wav2vec2 checkpoint (LFS)"
git push
```

### 方案 B：Release 附件

```bash
gh release create v1.0.0 models/w2v_ckpt.pt \
   --title "v1.0.0 model weights" --notes "wav2vec2 权重，下载后放入 models/"
```

（没有 `gh` 就网页操作：Releases → Draft a new release → 上传 `w2v_ckpt.pt`）

然后在 README 里把下载链接补上。

## 4. 验证

```bash
git clone git@github.com:<用户名>/xgb_wav.git && cd xgb_wav
pip install -r requirements.txt
# 放入 models/w2v_ckpt.pt（LFS 或 Release 下载）
python -m xgb_wav.cli --models models --input <某个wav目录> --output out.csv
```

## 5. 需要我代推时

把下列任一信息给我即可：

- 一个具 `repo` 权限的 GitHub Token（`ghp_...` / fine-grained），我用 HTTPS 推送；
- 或已配置好的 SSH key + 仓库 SSH 地址。

> 注意：Token 会出现在命令历史里，请在推送完成后立即在 GitHub 上撤销。
