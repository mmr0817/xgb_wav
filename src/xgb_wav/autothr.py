"""按「当前这一批」的分数分布自优化判定阈值。

判定规则（与 fusion.py 一致）：bad ⟺ score >= threshold

三种使用场景（GUI 里单选）：
  auto     不知道这批是好是坏 —— 找分数分布的自然分界（最大间隙 / Otsu）
  all_bad  这批全是坏件       —— 在「至少命中 r%」的前提下取**最高**阈值（阈值越高越不容易误杀好件）
  all_good 这批全是好件       —— 取**最低**的「全部放行」阈值（用来消除误杀）

注意：all_bad 得到的阈值可能远低于模型标定值，那会让产线上的误杀率飙升；
所以每个候选都带 warn 标记，GUI 会用红字提示。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

MODES = ("auto", "all_bad", "all_good")
EPS = 1e-4          # 阈值取整到 4 位小数时的安全余量
WARN_DELTA = 0.30   # 与模型标定值相差超过这个量，认为风险明显
MIN_SIDE = 0.10     # auto 模式下，分界两侧各需占本批比例，否则视为整批同类
VALLEY = 0.35       # 分界处直方密度需低于两侧峰值的这个比例，才算「真有谷」
NEAR_MODEL = WARN_DELTA   # 数据分界离标定值不超过它，才认为是在「按批微调」


@dataclass
class Candidate:
    key: str
    title: str
    threshold: float
    n_bad: int
    n_good: int
    note: str = ""
    warn: bool = False
    recommended: bool = False

    def line(self) -> str:
        flag = "!" if self.warn else " "
        return (f"{flag}{self.title:<22s} θ={self.threshold:+.4f}  "
                f"bad={self.n_bad:<4d} good={self.n_good:<4d} {self.note}")


def _counts(scores: np.ndarray, thr: float) -> tuple[int, int]:
    n_bad = int(np.count_nonzero(scores >= thr))
    return n_bad, int(scores.size - n_bad)


def _both_sides_big_enough(n_bad: int, n_good: int) -> bool:
    """两侧都够大才算「真的分成两类」，否则说明这批基本是同一类。"""
    need = max(3, int(round(MIN_SIDE * (n_bad + n_good))))
    return min(n_bad, n_good) >= need


def valley_ratio(scores: Sequence[float], thr: float, bins: int = 18) -> float:
    """阈值所在直方格子的密度 / 两侧峰值密度的较小者；越接近 0 越像两个簇之间的谷。"""
    x = np.asarray(scores, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if x.size < 4 or hi - lo < 1e-9:
        return 1.0
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    i = int(np.clip(np.searchsorted(centers, thr), 1, bins - 2))
    left, right = int(hist[:i].max()), int(hist[i:].max())
    if left <= 0 or right <= 0:
        return 1.0
    return float(hist[i]) / float(min(left, right))


def largest_gap(scores: Sequence[float]) -> tuple[float, float] | None:
    """最大间隙：把分数二分得最开的那个位置（中点作为阈值）。"""
    x = np.unique(np.sort(np.asarray(scores, dtype=float)))
    if x.size < 2:
        return None
    gaps = np.diff(x)
    i = int(np.argmax(gaps))
    return float((x[i] + x[i + 1]) / 2.0), float(gaps[i])


def otsu(scores: Sequence[float]) -> float | None:
    """一维 Otsu（类间方差最大），适合双峰分布。"""
    x = np.asarray(scores, dtype=float)
    if x.size < 8 or float(x.max() - x.min()) < 1e-9:
        return None
    hist, edges = np.histogram(x, bins=min(256, max(8, x.size // 2)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(float)
    total = w.sum()
    if total <= 0:
        return None
    p = w / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    if np.all(np.isnan(sigma_b)):
        return None
    i = int(np.nanargmax(sigma_b))
    return float(centers[i])


def recall_threshold(scores: Sequence[float], recall: float) -> float:
    """在「这批全是坏件、要求命中 recall 比例」的前提下，取最高的可行阈值。"""
    x = np.sort(np.asarray(scores, dtype=float))
    n = x.size
    allowed_miss = int(np.floor((1.0 - recall) * n + 1e-9))   # 允许漏掉几条
    idx = min(allowed_miss, n - 1)                            # 第 (k+1) 小的分数
    return float(x[idx])


def precision_threshold(scores: Sequence[float], keep: float = 1.0) -> float:
    """在「这批全是好件、要求放行 keep 比例」的前提下，取最低的可行阈值。"""
    x = np.sort(np.asarray(scores, dtype=float))
    n = x.size
    allowed = int(np.floor((1.0 - keep) * n + 1e-9))
    idx = n - 1 - min(allowed, n - 1)
    return float(x[idx]) + EPS


def suggest(
    scores: Sequence[float],
    model_threshold: float,
    mode: str = "auto",
    recall_targets: Sequence[float] = (1.0, 0.995, 0.95),
) -> List[Candidate]:
    """生成候选阈值（含模型标定值作对照），第一个 recommended 为建议值。"""
    arr = np.asarray(list(scores), dtype=float)
    out: List[Candidate] = []

    if arr.size == 0:
        return out

    n_bad, n_good = _counts(arr, model_threshold)
    out.append(Candidate(
        key="model", title="模型标定值",
        threshold=round(float(model_threshold), 4),
        n_bad=n_bad, n_good=n_good,
        note="models/fusion.json 里验证集标定的值",
    ))

    if mode == "all_bad":
        for r in recall_targets:
            thr = round(recall_threshold(arr, r), 4)
            if any(abs(thr - c.threshold) < 1e-9 for c in out):
                continue                       # 与前面某行阈值相同，不重复列
            b, g = _counts(arr, thr)
            miss = int(arr.size - b)
            pct = (1.0 - r) * 100.0
            title = "本批全坏·全部命中" if r >= 1.0 else f"本批全坏·命中{100 - pct:g}%"
            out.append(Candidate(
                key=f"bad_{r}", title=title, threshold=thr, n_bad=b, n_good=g,
                note=f"漏 {miss} 条" + ("（最高可行阈值）" if r >= 1.0 else ""),
                warn=thr < model_threshold - WARN_DELTA,
            ))
        rec_key = "bad_1.0"
    elif mode == "all_good":
        for keep in (1.0, 0.995, 0.95):
            thr = round(precision_threshold(arr, keep), 4)
            if any(abs(thr - c.threshold) < 1e-9 for c in out):
                continue
            b, g = _counts(arr, thr)
            kill = int(arr.size - g)
            title = "本批全好·全部放行" if keep >= 1.0 else f"本批全好·放行{keep * 100:g}%"
            out.append(Candidate(
                key=f"good_{keep}", title=title, threshold=thr, n_bad=b, n_good=g,
                note=f"误杀 {kill} 条" + ("（零误杀的最低阈值）" if keep >= 1.0 else ""),
                warn=thr > model_threshold + WARN_DELTA,
            ))
        rec_key = "good_1.0"
    else:
        def judge(thr: float) -> tuple[int, int, bool, bool]:
            """返回 (bad, good, 分界是否可信, 是否贴近模型标定值)。"""
            b, g = _counts(arr, thr)
            ok = _both_sides_big_enough(b, g) and valley_ratio(arr, thr) <= VALLEY
            near = abs(thr - model_threshold) <= NEAR_MODEL
            return b, g, ok, near

        chosen = None
        gap = largest_gap(arr)
        if gap is not None:
            thr = round(gap[0], 4)
            b, g, ok, near = judge(thr)
            notes = [f"两簇间隔 {gap[1]:.3f}"]
            if not _both_sides_big_enough(b, g):
                notes.append(f"小的一侧仅 {min(b, g)} 条，像整批同类")
            if not near:
                notes.append(f"离标定值 {abs(thr - model_threshold):.2f}，可能是类别内部的分界")
            out.append(Candidate(
                key="gap", title="最大间隙分界", threshold=thr, n_bad=b, n_good=g,
                note="；".join(notes), warn=not (ok and near),
            ))
            if ok and near:
                chosen = "gap"
        thr_o = otsu(arr)
        if thr_o is not None and not any(
                abs(round(thr_o, 4) - c.threshold) < 1e-9 for c in out):
            thr = round(thr_o, 4)
            b, g, ok, near = judge(thr)
            notes = ["类间方差最大"]
            if valley_ratio(arr, thr) > VALLEY:
                notes.append("谷不明显，分布不像双峰")
            if not near:
                notes.append(f"离标定值 {abs(thr - model_threshold):.2f}，可能是类别内部的分界")
            out.append(Candidate(
                key="otsu", title="Otsu 双峰", threshold=thr, n_bad=b, n_good=g,
                note="；".join(notes), warn=not (ok and near),
            ))
            if chosen is None and ok and near:
                chosen = "otsu"
        # 没有「可信且贴近标定值」的数据分界时，建议沿用模型标定值
        rec_key = chosen or "model"

    for c in out:
        if c.key == rec_key:
            c.recommended = True
    if not any(c.recommended for c in out) and len(out) > 1:
        out[1].recommended = True      # 首选被去重掉时，退到第一条数据候选
    return out


def apply_mode(scores: Sequence[float], mode: str = "auto",
               model_threshold: float | None = None) -> float:
    """命令行用：直接给出「推荐候选」的阈值（与 GUI 完全同一套规则）。"""
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return float(model_threshold or 0.0)
    base = float(model_threshold) if model_threshold is not None else float(np.median(arr))
    cands = suggest(arr, base, mode)
    rec = next((c for c in cands if c.recommended), None)
    return round(float(rec.threshold), 4) if rec is not None else round(base, 4)
