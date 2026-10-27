"""图形界面（tkinter）：选择文件/文件夹 -> 检测 -> 导出 CSV。

打包后可直接双击运行；未打包时：
    python -m xgb_wav.gui
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .data import discover_wavs
from .pipeline import FusionDetector


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

        row = ttk.Frame(top)
        row.grid(row=3, column=0, columnspan=4, sticky="we", pady=6)
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

    # ---------------- 模型 ----------------
    def _load_model_async(self) -> None:
        self.status.set("正在加载模型…")
        self.run_btn.configure(state="disabled")

        def work():
            try:
                det = FusionDetector.from_dir(self.model_var.get(), device="auto")
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
        for _, r in df.iterrows():
            self.tree.insert("", "end",
                             values=(r["file"], r["label"], f"{r['score']:.3f}",
                                     f"{r['xgb_proba']:.4f}", f"{r['w2v_proba']:.4f}",
                                     r["channel"], r["batch"], r["path"]),
                             tags=(r["label"],))
        self.progress.configure(value=len(df))
        n_bad = int((df["pred"] == 1).sum())
        self.status.set(f"完成：{len(df)} 条，bad {n_bad}，good {len(df) - n_bad}，"
                        f"耗时 {df.attrs.get('elapsed', 0):.1f}s")
        self.export_btn.configure(state="normal")
        self.run_btn.configure(state="normal")
        self.export()

    def export(self) -> None:
        if self.result is None:
            return
        out = Path(self.out_var.get()).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.result.to_csv(out, index=False, encoding="utf-8-sig")
        self.log.insert("end", f"已导出: {out}\n")
        self.log.see("end")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()

