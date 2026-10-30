"""图形界面（tkinter）：选择文件/文件夹 -> 检测 -> 调阈值 -> 导出 CSV。

打包后可直接双击运行；未打包时：python main.py
阈值可在界面里实时调整（不重新推理，只按已有融合分重判），并保存到 settings.json；
「自动定阈…」按本批分数分布给候选阈值（未知 / 全坏 / 全好三种假设，见 autothr.py）。
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ._console import setup_console
from .autothr import suggest as suggest_thresholds
from .data import discover_wavs
from .pipeline import FusionDetector
from .perf import apply_low_power
from .settings import app_dir, load_settings, save_settings  # noqa: F401  (GUI 也在用)


def default_model_dir() -> Path:
    """优先 exe 同级的 models/，其次包内 models/，最后仓库 models/。"""
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates += [exe_dir / "models", Path(getattr(sys, "_MEIPASS", exe_dir)) / "models"]
    here = Path(__file__).resolve()
    candidates += [here.parents[2] / "models", here.parents[1] / "models", Path.cwd() / "models"]
    for c in candidates:
        if (c / "detector.json").exists():
            return c
    return candidates[0]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("xgb_wav · 工业音频坏件检测")
        self.geometry("1000x680")
        self.minsize(880, 600)

        self.detector: Optional[FusionDetector] = None
        self.paths: List[Path] = []
        self.result: Optional[pd.DataFrame] = None
        self.queue: "queue.Queue" = queue.Queue()
        self.settings: Dict[str, object] = load_settings()
        self.model_threshold: float = 0.0      # 模型自带阈值（恢复默认用）
        self._relabel_job = None

        self._build_ui()
        self.after(100, self._poll_queue)
        self._load_model_async()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        top = ttk.LabelFrame(self, text="模型与输入")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="模型目录:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.model_var = tk.StringVar(value=str(default_model_dir()))
        ttk.Entry(top, textvariable=self.model_var, width=72).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="浏览…", command=self.browse_model).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="加载模型", command=self._load_model_async).grid(row=0, column=3, padx=4)

        ttk.Label(top, text="输入:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.input_var = tk.StringVar(value="（未选择）")
        ttk.Entry(top, textvariable=self.input_var, width=72).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(top, text="选择文件", command=self.pick_files).grid(row=1, column=2, padx=4)
        ttk.Button(top, text="选择文件夹", command=self.pick_folder).grid(row=1, column=3, padx=4)

        ttk.Label(top, text="输出 CSV:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.out_var = tk.StringVar(value=str(Path.cwd() / "predictions.csv"))
        ttk.Entry(top, textvariable=self.out_var, width=72).grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(top, text="另存为…", command=self.pick_output).grid(row=2, column=2, padx=4)

        thr_row = ttk.Frame(top)
        thr_row.grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=2)
        ttk.Label(thr_row, text="判定阈值:").pack(side="left")
        self.thr_var = tk.DoubleVar(value=0.0)
        self.thr_spin = ttk.Spinbox(thr_row, from_=-2.0, to=6.0, increment=0.01,
                                    textvariable=self.thr_var, width=8,
                                    command=self._on_thr_change)
        self.thr_spin.pack(side="left", padx=4)
        self.thr_scale = ttk.Scale(thr_row, from_=-1.0, to=3.0,
                                   variable=self.thr_var, command=self._on_thr_change)
        self.thr_scale.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(thr_row, text="应用", command=self.apply_threshold).pack(side="left", padx=4)
        ttk.Button(thr_row, text="恢复模型默认",
                   command=self.reset_threshold).pack(side="left", padx=4)
        ttk.Button(thr_row, text="自动定阈…",
                   command=self.auto_threshold_dialog).pack(side="left", padx=4)
        self.thr_info = tk.StringVar(value="阈值越大：漏检↑ 误杀↓")
        ttk.Label(thr_row, textvariable=self.thr_info).pack(side="left", padx=8)

        self.low_var = tk.BooleanVar(value=bool(load_settings().get("low_power", False)))
        ttk.Checkbutton(thr_row, text="省电模式(≈1/4速度)", variable=self.low_var,
                        command=self._on_low_power).pack(side="left", padx=8)

        row = ttk.Frame(top)
        row.grid(row=4, column=0, columnspan=4, sticky="we", pady=6)
        self.run_btn = ttk.Button(row, text="开始检测", command=self.start, state="disabled")
        self.run_btn.pack(side="left", padx=8)
        self.export_btn = ttk.Button(row, text="导出 CSV", command=self.export, state="disabled")
        self.export_btn.pack(side="left", padx=4)
        self.status = tk.StringVar(value="正在加载模型…")
        ttk.Label(row, textvariable=self.status).pack(side="left", padx=12)
        top.columnconfigure(1, weight=1)

        mid = ttk.LabelFrame(self, text="检测结果")
        mid.pack(fill="both", expand=True, **pad)
        cols = ("file", "label", "score", "xgb", "w2v", "channel", "batch", "path")
        titles = ("文件", "判定", "融合分", "XGB", "wav2vec2", "通道", "批次", "路径")
        widths = (220, 60, 80, 80, 90, 60, 90, 320)
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=18)
        for c, t, w in zip(cols, titles, widths):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("bad", background="#ffe2e2")
        self.tree.tag_configure("good", background="#e8f6e8")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=(0, 8))
        self.log = tk.Text(self, height=6)
        self.log.pack(fill="x", padx=8, pady=(0, 8))

    # ---------------- 交互 ----------------
    def browse_model(self) -> None:
        d = filedialog.askdirectory(title="选择模型目录（含 detector.json）")
        if d:
            self.model_var.set(d)
            self._load_model_async()

    def pick_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择 wav 文件", filetypes=[("WAV", "*.wav *.WAV")])
        if files:
            self.paths = [Path(f) for f in files]
            self.input_var.set(f"{len(self.paths)} 个文件")
            self.run_btn.configure(state="normal" if self.detector else "disabled")

    def pick_folder(self) -> None:
        d = filedialog.askdirectory(title="选择包含 wav 的文件夹")
        if not d:
            return
        self.paths = discover_wavs(Path(d))
        self.input_var.set(f"{d}（{len(self.paths)} 个 wav）")
        self.run_btn.configure(state="normal" if (self.paths and self.detector) else "disabled")
        if not self.paths:
            messagebox.showwarning("提示", "该文件夹下没有 wav 文件")

    def pick_output(self) -> None:
        f = filedialog.asksaveasfilename(title="保存 CSV", defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if f:
            self.out_var.set(f)

    # ---------------- 阈值 ----------------
    def _on_thr_change(self, _evt=None) -> None:
        """拖动滑块/改数字后稍作防抖，再重新判定（不重新推理）。"""
        if self._relabel_job is not None:
            try:
                self.after_cancel(self._relabel_job)
            except Exception:
                pass
        self._relabel_job = self.after(250, self.apply_threshold)

    def _on_low_power(self) -> None:
        """省电模式开关：立即生效（下次推理起限速），并写入 settings.json。"""
        flag = bool(self.low_var.get())
        apply_low_power(flag)
        data = load_settings()
        data["low_power"] = flag
        save_settings(data)
        self.status.set("省电模式已开启（约 1/4 速度，占用/发热更低）" if flag else "省电模式已关闭（全速）")

    def apply_threshold(self) -> None:
        self._relabel_job = None
        if self.detector is None:
            return
        try:
            thr = round(float(self.thr_var.get()), 4)
        except (tk.TclError, ValueError):
            return
        self.thr_var.set(thr)
        self.detector.set_threshold(thr)
        if self.result is not None and len(self.result):
            self.result["threshold"] = thr
            self.result["pred"] = (self.result["score"].to_numpy() >= thr).astype(int)
            self.result["label"] = [self.detector.rule.label_of(v)
                                    for v in self.result["pred"]]
            self._render_results()
        self._update_thr_info()
        self.settings["threshold"] = thr
        save_settings(self.settings)

    def reset_threshold(self) -> None:
        self.thr_var.set(round(float(self.model_threshold), 4))
        self.apply_threshold()

    def _update_thr_info(self) -> None:
        if self.detector is None:
            self.thr_info.set("")
            return
        thr = float(self.thr_var.get())
        if self.result is not None and len(self.result):
            n_bad = int((self.result["pred"] == 1).sum())
            self.thr_info.set(f"阈值 {thr:.4f}｜bad {n_bad} / good {len(self.result) - n_bad}")
        else:
            self.thr_info.set(f"阈值 {thr:.4f}（模型默认 {self.model_threshold:.4f}）")

    # ---------------- 按批自优化阈值 ----------------
    def auto_threshold_dialog(self) -> None:
        """按当前这一批的分数分布给候选阈值（三种假设：未知 / 全坏 / 全好）。"""
        if self.result is None or not len(self.result):
            messagebox.showinfo("自动定阈", "请先检测一批数据（点「开始检测」），再按本批分数定阈。")
            return
        AutoThresholdDialog(self)

    def _log_thr_hint(self, df: pd.DataFrame) -> None:
        """跑完一批后，在日志里提示本批分数分界在哪（不自动改阈值）。"""
        try:
            base = float(self.model_threshold or self.thr_var.get())
            cands = suggest_thresholds(df["score"].to_numpy(), base, "auto")
        except Exception:
            return
        rec = next((c for c in cands if c.recommended), None)
        if rec is None or rec.key == "model":
            return
        try:
            cur = float(self.thr_var.get())
        except (tk.TclError, ValueError):
            cur = base
        if abs(rec.threshold - cur) < 1e-6:
            return
        self.log.insert("end", f"[提示] 本批分数{rec.title}在 {rec.threshold:+.3f}"
                               f"（bad {rec.n_bad} / good {rec.n_good}）；"
                               f"点「自动定阈…」可套用\n")
        self.log.see("end")

    # ---------------- 模型 ----------------
    def _load_model_async(self) -> None:
        self.status.set("正在加载模型…")
        self.run_btn.configure(state="disabled")
        model_dir = self.model_var.get()      # Tk 变量只能在主线程读取

        def work():
            try:
                det = FusionDetector.from_dir(model_dir, device="auto")
                self.queue.put(("model", det))
            except Exception as e:
                self.queue.put(("error", f"模型加载失败: {e}"))

        threading.Thread(target=work, daemon=True).start()

    # ---------------- 检测 ----------------
    def start(self) -> None:
        if not self.paths or self.detector is None:
            return
        self.run_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.progress.configure(value=0, maximum=len(self.paths))
        self.status.set(f"检测中… 0/{len(self.paths)}")

        def work():
            try:
                df = self.detector.predict_paths(self.paths)
                self.queue.put(("result", df))
            except Exception:
                self.queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "model":
                    self.detector = payload
                    self.model_threshold = float(payload.default_threshold)
                    try:
                        thr = float(self.settings.get("threshold", self.model_threshold))
                    except (TypeError, ValueError):
                        thr = self.model_threshold
                    self.thr_var.set(round(thr, 4))
                    payload.set_threshold(thr)
                    self._update_thr_info()
                    self.status.set(f"模型就绪（{payload.device}）· {payload.rule.summary()}")
                    self.run_btn.configure(state="normal" if self.paths else "disabled")
                elif kind == "result":
                    self._show_result(payload)
                elif kind == "error":
                    self.status.set("出错")
                    self.log.insert("end", str(payload) + "\n")
                    messagebox.showerror("错误", str(payload)[:800])
                    self.run_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _show_result(self, df: pd.DataFrame) -> None:
        self.result = df
        self._render_results()
        self.progress.configure(value=len(df))
        n_bad = int((df["pred"] == 1).sum())
        self.status.set(f"完成：{len(df)} 条，bad {n_bad}，good {len(df) - n_bad}，"
                        f"耗时 {df.attrs.get('elapsed', 0):.1f}s（阈值可调）")
        self.export_btn.configure(state="normal")
        self.run_btn.configure(state="normal")
        self._log_thr_hint(df)
        self.export()

    def _render_results(self) -> None:
        """按当前阈值重绘表格（阈值调整后调用，不重新推理）。"""
        df = self.result
        if df is None:
            return
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            self.tree.insert("", "end",
                             values=(r["file"], r["label"], f"{r['score']:.3f}",
                                     f"{r['xgb_proba']:.4f}", f"{r['w2v_proba']:.4f}",
                                     r["channel"], r["batch"], r["path"]),
                             tags=(r["label"],))
        self._update_thr_info()

    def export(self) -> None:
        if self.result is None:
            return
        try:
            thr = round(float(self.thr_var.get()), 4)
        except (tk.TclError, ValueError):
            thr = float(self.result["threshold"].iloc[0])
        df = self.result.copy()
        df["threshold"] = thr
        df["pred"] = (df["score"].to_numpy() >= thr).astype(int)
        if self.detector is not None:
            df["label"] = [self.detector.rule.label_of(v) for v in df["pred"]]
        out = Path(self.out_var.get()).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        n_bad = int((df["pred"] == 1).sum())
        self.log.insert("end", f"已导出: {out}（阈值 {thr:.4f}，"
                               f"bad {n_bad} / good {len(df) - n_bad}）\n")
        self.log.see("end")


class AutoThresholdDialog(tk.Toplevel):
    """按当前批次分数分布，列出候选阈值供用户选择（不自动改，点「应用」才生效）。"""

    MODE_LABELS = (
        ("auto", "未知（自动找分界）"),
        ("all_bad", "本批全是坏件（防漏检）"),
        ("all_good", "本批全是好件（防误杀）"),
    )

    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self.title("自动定阈 · 按本批分数分布")
        self.geometry("760x430")
        self.transient(app)
        self.grab_set()
        self.cands: List = []

        df = app.result
        self.scores = df["score"].to_numpy()
        cur = float(app.thr_var.get())
        info = (f"本批 {len(self.scores)} 条｜分数 min {self.scores.min():+.3f}"
                f" / 中位 {float(df['score'].median()):+.3f}"
                f" / max {self.scores.max():+.3f}｜当前阈值 {cur:.4f}")
        ttk.Label(self, text=info).pack(anchor="w", padx=10, pady=(8, 0))

        mode_row = ttk.Frame(self)
        mode_row.pack(anchor="w", padx=10, pady=6)
        ttk.Label(mode_row, text="这批数据是：").pack(side="left")
        self.mode_var = tk.StringVar(value="auto")
        for value, label in self.MODE_LABELS:
            ttk.Radiobutton(mode_row, text=label, value=value,
                            variable=self.mode_var,
                            command=self.refresh).pack(side="left", padx=6)

        cols = ("title", "thr", "bad", "good", "note")
        titles = ("策略", "阈值", "bad", "good", "说明")
        widths = (175, 80, 60, 60, 380)
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, padx=10)
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings", height=7)
        for c, t, w in zip(cols, titles, widths):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("warn", foreground="#c0392b")
        self.tree.tag_configure("rec", background="#e8f0fe")
        self.tree.bind("<Double-1>", lambda _e: self.apply_selected())

        self.note = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.note, foreground="#555").pack(anchor="w", padx=10)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=8)
        ttk.Button(btns, text="应用所选阈值", command=self.apply_selected).pack(side="right", padx=4)
        ttk.Button(btns, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="选中推荐值", command=self.select_recommended).pack(side="left", padx=4)

        self.refresh()

    def refresh(self) -> None:
        base = float(self.app.model_threshold or self.app.thr_var.get())
        self.cands = suggest_thresholds(self.scores, base, self.mode_var.get())
        self.tree.delete(*self.tree.get_children())
        for c in self.cands:
            tags = ["rec"] if c.recommended else (["warn"] if c.warn else [])
            self.tree.insert("", "end", tags=tags,
                             values=(c.title, f"{c.threshold:.4f}", c.n_bad, c.n_good, c.note))
        self.select_recommended()
        if self.mode_var.get() == "all_bad":
            self.note.set("全坏批次：阈值取到「最高且不漏」；若远低于模型标定值（红色），"
                          "说明本批有低分坏件，产线上误杀会升高。")
        elif self.mode_var.get() == "all_good":
            self.note.set("全好批次：阈值取到「最低且全部放行」；若高于标定值（红色），"
                          "说明本批有高分好件，需确认是否值得为此提高漏检风险。")
        else:
            self.note.set("未知批次：按分数分布的自然分界（最大间隙 / Otsu）二分，适合好坏混杂时快速定阈。")

    def select_recommended(self) -> None:
        items = self.tree.get_children()
        if not items:
            return
        idx = next((i for i, c in enumerate(self.cands) if c.recommended), 0)
        self.tree.selection_set(items[idx])
        self.tree.focus(items[idx])

    def apply_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("自动定阈", "请先选中一行。", parent=self)
            return
        idx = self.tree.index(sel[0])
        c = self.cands[idx]
        self.app.thr_var.set(round(float(c.threshold), 4))
        self.app.apply_threshold()
        self.app.log.insert("end", f"[自动定阈] {c.title} → 阈值 {c.threshold:.4f}"
                                   f"（bad {c.n_bad} / good {c.n_good}）\n")
        self.app.log.see("end")
        self.destroy()


def main() -> None:
    setup_console()
    App().mainloop()


if __name__ == "__main__":
    main()

