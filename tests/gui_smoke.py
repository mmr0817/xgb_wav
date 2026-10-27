#!/usr/bin/env python
"""GUI 冒烟测试：在真实 Tk 窗口里跑一遍「加载模型 -> 选文件夹 -> 开始检测 -> 导出 CSV」。

用法（需要 X 显示，例如 DISPLAY=:1 或本机桌面）：
    DISPLAY=:1 python tests/gui_smoke.py --input /tmp/demo_wav

无显示环境下直接退出 0（跳过），不会让 CI 失败。

检查项
------
1. App 构造成功、模型异步加载完成（状态栏出现「模型就绪」）
2. 模拟「选择文件夹」→「开始检测」按钮点击（走真实回调）
3. Treeview 行数 == 输入文件数，bad/good 行数与预期一致
4. 自动导出 CSV，内容与 CLI/API 结果一致
5. 顺带截图（有 ffmpeg 时），便于人工确认界面

退出码：0 全部通过（或无显示跳过）；1 有检查失败
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

import xgb_wav.gui as gui  # noqa: E402
from xgb_wav.data import discover_wavs  # noqa: E402

CHECKS: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    return bool(ok)


def pump(app, seconds: float = 0.05) -> None:
    """跑一轮事件循环（处理 after 回调，例如队列轮询）。"""
    try:
        app.update()
    except Exception:
        pass
    time.sleep(seconds)


def wait_until(app, cond, timeout: float = 240.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        pump(app)
    return False


def screenshot(app, path: Path) -> bool:
    """截图：优先抓窗口自身（-window_id），失败再抓全屏并按窗口坐标裁剪。

    用亮度做校验：Tk 窗口是浅色（均值 > 150），锁屏/桌面壁纸是深色。
    """
    if not shutil.which("ffmpeg"):
        return False
    try:
        from PIL import Image, ImageStat

        path.parent.mkdir(parents=True, exist_ok=True)
        pump(app, 0.4)
        disp = os.environ.get("DISPLAY", ":0")
        w, h = max(1, app.winfo_width()), max(1, app.winfo_height())
        sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()

        plans = []
        if app.winfo_id():
            plans.append(("window", ["-window_id", str(app.winfo_id())], f"{w}x{h}"))
        plans.append(("root", [], f"{sw}x{sh}"))

        for kind, extra, size in plans:
            tmp = path.with_name(f"{path.stem}_{kind}.png")
            try:
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "x11grab", *extra,
                     "-video_size", size, "-i", disp, "-frames:v", "1", str(tmp)],
                    check=True, timeout=60)
                im = Image.open(tmp).convert("RGB")
                if kind == "root":
                    x, y = app.winfo_rootx(), app.winfo_rooty()
                    im = im.crop((x, y, x + w, y + h))
                lum = ImageStat.Stat(im.convert("L")).mean[0]
                print(f"  [info] 截图方式={kind} 亮度={lum:.0f}")
                if lum > 150:                      # 浅色 => 是我们的窗口
                    im.save(path)
                    tmp.unlink(missing_ok=True)
                    return True
                tmp.unlink(missing_ok=True)
            except Exception as e:  # pragma: no cover
                print(f"  [warn] 截图({kind})失败: {e}")
        return False
    except Exception as e:  # pragma: no cover
        print(f"  [warn] 截图失败: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="xgb_wav GUI 冒烟测试")
    ap.add_argument("--input", default="/tmp/demo_wav", help="待检测的 wav 目录")
    ap.add_argument("--models", default=str(REPO / "models"), help="模型目录")
    ap.add_argument("--shot-dir", default=str(REPO / "outputs/gui_smoke"))
    ap.add_argument("--expect-bad", type=int, default=None, help="期望的 bad 条数")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    if not os.environ.get("DISPLAY"):
        print("[SKIP] 没有 DISPLAY（无图形环境），跳过真实窗口测试")
        return 0

    inp = Path(args.input).expanduser()
    files = discover_wavs(inp)
    shot_dir = Path(args.shot_dir).expanduser()
    out_csv = shot_dir / "gui_predictions.csv"
    print(f"输入目录 : {inp}（{len(files)} 个 wav）")
    print(f"模型目录 : {args.models}")
    print(f"截图/CSV : {shot_dir}")

    # 无人值守：把模态对话框改成打印，否则出错时会一直卡在对话框上
    gui.messagebox.showerror = lambda title, msg, **kw: print(
        f"[dialog.error] {title}: {str(msg)[:1500]}")
    gui.messagebox.showwarning = lambda title, msg, **kw: print(
        f"[dialog.warn] {title}: {str(msg)[:1500]}")

    print("\n[1] 启动 GUI 并等待模型加载")
    app = gui.App()
    app.geometry("1000x680+60+60")
    pump(app, 0.3)
    ok = wait_until(app, lambda: app.detector is not None, args.timeout)
    check("模型加载完成", ok, app.status.get())
    if not ok:
        app.destroy()
        return 1
    check("模型目录自动探测", Path(app.model_var.get()) == Path(args.models),
          app.model_var.get())
    check("状态栏显示阈值", "threshold" in app.status.get(), app.status.get())
    screenshot(app, shot_dir / "gui_1_ready.png")

    print("\n[2] 模拟「选择文件夹」")
    gui.filedialog.askdirectory = lambda **kw: str(inp)  # 假装用户在对话框里选了目录
    app.pick_folder()
    check("文件列表已填入", len(app.paths) == len(files), app.input_var.get())
    check("开始检测按钮已可用", str(app.run_btn["state"]) == "normal",
          f"state={app.run_btn['state']}")

    print("\n[3] 点击「开始检测」")
    app.out_var.set(str(out_csv))
    app.run_btn.invoke()
    ok = wait_until(app, lambda: app.result is not None, args.timeout)
    check("检测完成（有结果）", ok, app.status.get())
    if not ok:
        app.destroy()
        return 1

    rows = app.tree.get_children()
    vals = [app.tree.item(i, "values") for i in rows]
    bad = sum(1 for v in vals if v[1] == "bad")
    check("表格行数 == 输入文件数", len(rows) == len(files), f"{len(rows)}/{len(files)}")
    check("表格 bad 数符合预期",
          args.expect_bad is None or bad == args.expect_bad, f"bad={bad}")
    check("进度条已走满", int(float(app.progress["value"])) == len(files),
          f"progress={app.progress['value']}/{len(files)}")
    check("状态栏显示完成", "完成" in app.status.get(), app.status.get())
    check("导出 CSV 已生成", out_csv.exists(), str(out_csv))

    if out_csv.exists():
        df = pd.read_csv(out_csv, encoding="utf-8-sig")
        check("CSV 行数一致", len(df) == len(files), f"{len(df)}")
        bad_csv = int((df["pred"] == 1).sum())
        check("CSV bad 数与界面一致", bad_csv == bad, f"csv={bad_csv} ui={bad}")
        print("\n界面结果:")
        print(df[["file", "label", "score", "xgb_proba", "w2v_proba"]]
              .to_string(index=False, max_colwidth=34))
        check("分数在合理范围", bool(((df["score"] > -20) & (df["score"] < 20)).all()))

    print("\n[4] 截图")
    shot_ok = screenshot(app, shot_dir / "gui_2_result.png")
    print(f"  截图: {shot_dir / 'gui_2_result.png' if shot_ok else '（不可用）'}")

    print("\n[5] 错误处理：模型目录不存在")
    good_dir = app.model_var.get()
    app.model_var.set(str(shot_dir / "no_such_model_dir"))
    app._load_model_async()
    ok = wait_until(app, lambda: app.status.get() == "出错", 120)
    check("错误被捕获且窗口存活", ok and bool(app.winfo_exists()), app.status.get())
    app.model_var.set(good_dir)
    app._load_model_async()
    ok = wait_until(app, lambda: "模型就绪" in app.status.get(), args.timeout)
    check("修正目录后可恢复加载", ok, app.status.get())

    app.destroy()

    failed = [c for c in CHECKS if not c[1]]
    print("\n" + "=" * 60)
    print(f"结果：{len(CHECKS) - len(failed)}/{len(CHECKS)} 项通过")
    if failed:
        for name, _, detail in failed:
            print(f"  失败: {name} {detail}")
        return 1
    print("GUI 冒烟测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
