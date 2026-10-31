# -*- coding: utf-8 -*-
"""CEG 和弦音频的 DFT 频谱、采样率变换与 2 倍插值升采样分析。

运行:
    python analyse_ceg.py

输出:
    figures/   报告用图
    audio/     各采样率下的 5 s 音频（原始采样率）
    audio/preview/  供网页试听的 16 kHz 转码版本
    analysis_summary.json  全部定量结果
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy import signal
from scipy.io import wavfile

# --------------------------------------------------------------------------- 基本设置
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
AUDIO = ROOT / "audio"
PREVIEW = AUDIO / "preview"
for d in (FIG, AUDIO, PREVIEW):
    d.mkdir(parents=True, exist_ok=True)

FS0 = 44100
DUR = 5.0
NOTE_F = {"C4": 261.625565, "E4": 329.627557, "G4": 391.995436}
NOTE_CN = {"C4": "C4", "E4": "E4", "G4": "G4"}
COL = {"C4": "#2F5597", "E4": "#548235", "G4": "#C00000"}
INK = "#1F2937"
GRID = "#D9DEE7"
ACCENT = "#C55A11"
PURPLE = "#7030A0"
TEAL = "#0F766E"
GRAY = "#6B7280"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.65,
    "axes.edgecolor": "#9AA3AF",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# --------------------------------------------------------------------------- 信号 I/O
def load_wav(path: Path) -> tuple[np.ndarray, int]:
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind == "i":
        x = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    elif data.dtype.kind == "u":
        x = (data.astype(np.float64) - np.iinfo(data.dtype).max / 2) / (np.iinfo(data.dtype).max / 2)
    else:
        x = data.astype(np.float64)
    return x, int(fs)


def write_wav(path: Path, x: np.ndarray, fs: int) -> None:
    """写 16 bit PCM；不做响度归一化，保留各方案之间的真实电平关系。"""
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    wavfile.write(path, int(fs), (y * 32767.0).astype(np.int16))


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --------------------------------------------------------------------------- 频谱工具
def dft_spectrum(x: np.ndarray, fs: int, pad: int = 4):
    """Hann 窗单边幅度谱；pad 倍补零只用于显示平滑，不提高真实分辨率。"""
    w = signal.windows.hann(x.size, sym=False)
    nfft = 1 << int(math.ceil(math.log2(max(2, x.size * pad))))
    X = np.fft.rfft(x * w, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    amp = 2.0 * np.abs(X) / np.sum(w)
    return f, amp


def welch_spectrum(x: np.ndarray, fs: int, nperseg: int = 8192):
    nper = int(min(len(x), nperseg))
    if nper < 256:
        nper = len(x)
    f, P = signal.welch(x, fs, window="hann", nperseg=nper,
                        noverlap=nper * 3 // 4, detrend=False,
                        scaling="spectrum", return_onesided=True)
    return f, P


def db(x, floor=1e-14):
    return 10.0 * np.log10(np.maximum(x, floor))


def parabolic_peak(f: np.ndarray, y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(y) - 1:
        return float(f[i])
    a, b, c = np.log(np.maximum(y[i - 1:i + 2], 1e-20))
    den = a - 2 * b + c
    if abs(den) < 1e-15:
        return float(f[i])
    d = 0.5 * (a - c) / den
    return float(f[i] + d * (f[1] - f[0]))


def peak_near(f, P, target, half=18.0):
    m = (f >= target - half) & (f <= target + half) & (f > 0)
    if not np.any(m):
        return np.nan, np.nan
    idx = np.where(m)[0]
    j = idx[np.argmax(P[idx])]
    return parabolic_peak(f, P, j), float(P[j])


def peak_closest(f, P, target, half=7.0):
    """在目标附近选“离目标最近的局部极大值”，用于判断基频峰是否仍在原位。

    欠采样时混叠分量可能比基频更强（1 kHz 情形），若直接取最大值会把
    混叠峰误当成基频；这里保留“基频是否还在”这一判据。
    """
    m = (f >= target - half) & (f <= target + half) & (f > 0)
    if not np.any(m):
        return np.nan, np.nan
    idx = np.where(m)[0]
    loc = idx[signal.find_peaks(P[idx])[0]] if idx.size >= 3 else idx
    if loc.size == 0:
        loc = idx
    j = loc[np.argmin(np.abs(f[loc] - target))]
    return parabolic_peak(f, P, j), float(P[j])


def alias_of(f0: float, fs: int) -> float:
    """连续频率 f0 在采样率 fs 下的折叠频率，范围 [0, fs/2]。"""
    return float(abs(((f0 + fs / 2.0) % fs) - fs / 2.0))


def cents(f: float, f_ref: float) -> float:
    return float(1200.0 * math.log2(max(f, 1e-12) / f_ref))


def band_energy(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.sum(P[m]))


def centroid(f, P):
    s = np.sum(P)
    return float(np.sum(f * P) / s) if s > 0 else 0.0


def rolloff(f, P, q=0.95):
    s = np.sum(P)
    if s <= 0:
        return 0.0
    c = np.cumsum(P) / s
    return float(f[np.searchsorted(c, q)])


# --------------------------------------------------------------------------- 采样率变换
def downsample_naive(x: np.ndarray, fs_in: int, fs_out: int, seconds: float = DUR):
    """直接抽取（不预低通），模拟“只改采样率、不做抗混叠”的错误做法。"""
    m = int(round(seconds * fs_out))
    idx = np.round(np.arange(m) * fs_in / fs_out).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx]


def resample_proper(x: np.ndarray, fs_in: int, fs_out: int, seconds: float = DUR):
    """多相滤波重采样，先抗混叠再抽取；采样定理满足时使用。"""
    g = math.gcd(fs_in, fs_out)
    y = signal.resample_poly(x, fs_out // g, fs_in // g)
    m = int(round(seconds * fs_out))
    return y[:m]


def upsample_zeros(x: np.ndarray, L: int = 2):
    y = np.zeros(len(x) * L)
    y[::L] = x * L          # 补零并补偿 1/L 增益
    return y


def upsample_fir(x: np.ndarray, fs_in: int, L: int = 2,
                 taps: int = 1025, cutoff: float | None = None):
    """零插值 + FIR 低通内插；补偿线性相位延迟，使输出与输入时间对齐。"""
    y = upsample_zeros(x, L)
    fs_up = fs_in * L
    fc = (fs_in / 2.0 - 0.2) if cutoff is None else cutoff
    h = signal.firwin(taps, fc, fs=fs_up, window=("kaiser", 8.0))
    z = signal.upfirdn(h, y)
    delay = (taps - 1) // 2
    z = z[delay:delay + len(y)]
    return z


def upsample_linear(x: np.ndarray, L: int = 2):
    n = len(x)
    return np.interp(np.arange(n * L) / L, np.arange(n), x)


# --------------------------------------------------------------------------- 图 1 时域
def fig_time(x, fs, path):
    t = np.arange(len(x)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.4))
    ax = axes[0]
    ax.plot(t, x, lw=0.35, color=COL["C4"])
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("时间 $t$ / s")
    ax.set_ylabel("幅度")
    ax.set_title(f"原始 CEG 和弦时域波形（$f_s$ = {fs} Hz，{len(x)/fs:.2f} s，{len(x):,} 个采样点）", loc="left")
    ax.text(0.99, 0.95, f"峰值 {np.max(np.abs(x)):.3f}   RMS {np.sqrt(np.mean(x**2)):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))

    ax = axes[1]
    start = int(1.0 * fs)          # 避开起振段，截取稳态的 20 ms
    n = int(0.020 * fs)
    tt = (np.arange(n) + start) / fs * 1000
    ax.plot(tt, x[start:start + n], lw=0.8, color=COL["C4"], alpha=0.55, label="采样点连线")
    ax.plot(tt, x[start:start + n], "o", ms=2.4, mfc="white", mec=COL["G4"], mew=0.7,
            markevery=4, label="采样点（每 4 点标 1 个）")
    ax.set_xlim(1000, 1020)
    ax.set_xlabel("时间 $t$ / ms")
    ax.set_ylabel("幅度")
    ax.set_title("1.000–1.020 s 局部放大：$T_s$ = 22.68 μs，最高可表示频率 22.05 kHz", loc="left")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 2 原始频谱
def fig_spectrum_original(x, fs, path):
    # 5 s 单帧 DFT 的真实分辨率是 0.2 Hz，5 Hz 颤音会产生密集边带；
    # 用 4096 点 Welch 平均把边带平滑成“谱包络”，更适合读基频和谐波。
    f, P = welch_spectrum(x, fs, nperseg=4096)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.0),
                             gridspec_kw={"height_ratios": [1.0, 1.15]})
    ax = axes[0]
    m = f <= 1200
    ax.plot(f[m], P[m], lw=0.9, color=COL["C4"])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, P[m].max() * 1.22)
    ax.set_xlabel("频率 $f$ / Hz")
    ax.set_ylabel("功率谱 $P(f)$")
    ax.set_title("0–1.2 kHz 功率谱：CEG 三个基频及低次谐波（Welch，4096 点 Hann）", loc="left", fontsize=10.5)
    fhi, Phi = welch_spectrum(x, fs, nperseg=16384)   # 标注用高分辨率峰位
    for name, f0 in NOTE_F.items():
        fp, _ = peak_near(fhi, Phi, f0, 15)
        pp = P[np.argmin(np.abs(f - fp))]
        ax.axvline(fp, color=COL[name], lw=0.9, ls="--", alpha=0.85)
        ax.text(fp, pp * 1.04, f"{name}\n{fp:.1f} Hz", rotation=90, color=COL[name],
                fontsize=7.6, ha="center", va="bottom")

    ax = axes[1]
    m = f <= 5000
    pdb = 10 * np.log10(np.maximum(P[m], 1e-14))
    ax.plot(f[m], pdb, lw=0.9, color=COL["C4"])
    ax.set_xlim(0, 5000)
    ax.set_ylim(-110, max(-20, pdb.max() + 3))
    ax.set_xlabel("频率 $f$ / Hz")
    ax.set_ylabel("功率 / dB")
    ax.set_title("0–5 kHz 对数功率谱：谐波衰减与宽带噪声底", loc="left", fontsize=10.5)
    for name, f0 in NOTE_F.items():
        for h in range(1, 7):
            fh = h * f0
            if fh <= 5000:
                ax.axvline(fh, color=COL[name], lw=0.55, ls=":", alpha=0.5)
    ax.text(0.985, 0.06, "点线：各音符基频的整数倍谐波", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=GRAY)
    ax.text(0.985, 0.12, "高频段约为 −60 ~ −80 dB 的宽带噪声/混响底", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=GRAY)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f, P


# --------------------------------------------------------------------------- 图 3 时频图
def fig_spectrogram(x, fs, path):
    f, t, S = signal.spectrogram(x, fs, window="hann", nperseg=4096,
                                 noverlap=3584, scaling="spectrum", mode="magnitude")
    Sdb = 20 * np.log10(np.maximum(S, 1e-10))
    m = f <= 5000
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    mesh = ax.pcolormesh(t, f[m], Sdb[m], shading="auto", cmap="magma",
                         vmin=-100, vmax=-20, rasterized=True)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 5000)
    ax.set_xlabel("时间 $t$ / s")
    ax.set_ylabel("频率 $f$ / Hz")
    ax.set_title("原始信号时频图（STFT，Hann 窗 4096 点，重叠 87.5%）", loc="left")
    cb = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.046)
    cb.set_label("幅度 / dB", fontsize=9)
    for name, f0 in NOTE_F.items():
        ax.axhline(f0, color="white", lw=0.6, ls="--", alpha=0.55)
        ax.text(t[-1] * 1.004, f0, name, color="white", fontsize=8, va="center", clip_on=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 4 采样网格
def fig_sampling_grid(x0, fs0, cases, path):
    t_zoom = 0.012
    start = 1.0
    nref = int(t_zoom * fs0)
    sref = int(start * fs0)
    tref = (np.arange(nref) + sref) / fs0 * 1000
    fig, axes = plt.subplots(4, 1, figsize=(10.2, 7.0), sharex=True)
    for ax, (label, y, fs, note) in zip(axes, cases):
        ax.plot(tref, x0[sref:sref + nref], color="#9CA3AF", lw=0.9, alpha=0.8)
        n = int(t_zoom * fs)
        s = int(start * fs)
        ts = (np.arange(n) + s) / fs * 1000
        ys = y[s:s + n]
        ax.vlines(ts, 0, ys, color=COL["G4"], lw=0.55, alpha=0.55)
        ax.plot(ts, ys, "o", ms=4.2, mfc="white", mec=COL["G4"], mew=1.0)
        ax.axhline(0, color="#D1D5DB", lw=0.6)
        ax.set_ylabel("幅度")
        ax.set_title(f"{label}：$f_s$ = {fs} Hz，$T_s$ = {1000/fs:.3f} ms，{note}", loc="left", fontsize=9.6)
        ax.set_xlim(1000, 1012)
    axes[-1].set_xlabel("时间 $t$ / ms（1.000 s 起的 12 ms）")
    fig.suptitle("同一段和弦在不同采样率下的离散化\n灰线：44.1 kHz 连续波形；红圈：当前采样率下的采样点",
                 y=0.998, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 5 全带频谱
def fig_spectra_full(entries, path):
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    for ax, (label, y, fs, kind, color) in zip(axes.ravel(), entries):
        f, P = welch_spectrum(y, fs, nperseg=max(256, int(round(0.186 * fs))))
        ax.plot(f, db(P), lw=0.65, color=color)
        ax.set_xlim(0, fs / 2)
        ax.set_ylim(-105, -10)
        ax.set_xlabel("频率 $f$ / Hz")
        ax.set_ylabel("功率 / dB")
        ax.set_title(f"{label}\n$f_s$={fs} Hz，$f_N$={fs/2:.0f} Hz，{kind}", fontsize=9.0)
        ax.axvline(fs / 2, color=COL["G4"], lw=1.0, ls="--", alpha=0.8)
        ax.text(fs / 2, -101, " $f_N$", color=COL["G4"], fontsize=7.5, ha="right", va="bottom")
        if fs <= 2000:
            for name, f0 in NOTE_F.items():
                fa = alias_of(f0, fs)
                ax.axvline(fa, color=COL[name], lw=0.9, ls=":", alpha=0.9)
                ax.text(fa, -20, name, color=COL[name], fontsize=7.2, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    fig.suptitle("不同采样率下的功率谱（Welch，Hann，5 s）：带宽、Nyquist 频率与基频位置",
                 y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 6 低频放大
def fig_spectra_zoom(entries, path):
    data = {label: (y, fs) for label, y, fs, _kind, _color in entries}
    styles = {
        "44.1 kHz（原始）": ("#111827", 1.15, "-"),
        "22.05 kHz": (COL["C4"], 0.9, "--"),
        "8 kHz": (COL["E4"], 0.9, "-."),
        "1 kHz（直接抽取）": (ACCENT, 1.0, "-"),
        "512 Hz（直接抽取）": (COL["G4"], 1.15, "-"),
    }
    groups = [
        (["44.1 kHz（原始）", "22.05 kHz", "8 kHz"],
         "合法采样（44.1 / 22.05 / 8 kHz）：0–560 Hz 几乎重合"),
        (["1 kHz（直接抽取）"],
         "1 kHz 直接抽取：基频仍在，谐波折叠成密集谱线"),
        (["512 Hz（直接抽取）"],
         "512 Hz 直接抽取：基频本身已被折叠"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), sharey=True)
    for ax, (labels, title) in zip(axes, groups):
        for label in labels:
            y, fs = data[label]
            # 所有面板统一用 0.5 s 窗，Δf≈2 Hz，保证跨采样率可比
            f, P = welch_spectrum(y, fs, nperseg=max(256, int(round(0.5 * fs))))
            c, lw, ls = styles[label]
            m = f <= 560
            ax.plot(f[m], db(P[m]), lw=lw, ls=ls, color=c, label=label, alpha=0.95)
        ax.set_xlim(0, 560)
        ax.set_ylim(-105, -14)
        ax.set_xlabel("频率 $f$ / Hz")
        ax.set_title(title, loc="left", fontsize=9.4)
    axes[0].set_ylabel("功率 / dB")
    axes[0].legend(loc="lower left", fontsize=7.6)

    # 1 kHz：基频位置仍在，但谐波折叠成密集谱线
    ax = axes[1]
    for name, f0 in NOTE_F.items():
        ax.axvline(f0, color=COL[name], lw=0.8, ls=":", alpha=0.65)
        ax.text(f0, -17, name, color=COL[name], fontsize=8, ha="center", va="top")
    ax.text(0.97, 0.05, "基频峰位基本不变\n但 500–4000 Hz 谐波大量折回",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6, color=GRAY)

    # 512 Hz：基频本身折叠
    ax = axes[2]
    for name, f0 in NOTE_F.items():
        fa = alias_of(f0, 512)
        ax.axvline(f0, color="#9CA3AF", lw=0.7, ls=":", alpha=0.7)
        ax.axvline(fa, color=COL[name], lw=0.8, ls="--", alpha=0.55)
        ax.text(fa, -17, f"{name}\n{fa:.0f}", color=COL[name], fontsize=7.4,
                ha="center", va="top")
    lines = []
    for name, f0 in NOTE_F.items():
        lines.append(f"{name} {f0:.0f} → {alias_of(f0, 512):.0f} Hz")
    ax.text(0.04, 0.05, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.4, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", lw=0.6, alpha=0.9))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 7 混叠音高图
def fig_alias_notes(alias_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ypos = {"原始": 2.0, "1 kHz": 1.0, "512 Hz": 0.0}
    for name, f0 in NOTE_F.items():
        ax.scatter([f0], [ypos["原始"]], s=70, color=COL[name], zorder=3)
        ax.text(f0 + 8, ypos["原始"] + 0.10, f"{name}  {f0:.0f}", color=COL[name],
                ha="left", va="bottom", fontsize=7.8)
        f1 = alias_of(f0, 1000)
        ax.scatter([f1], [ypos["1 kHz"]], s=55, color=COL[name], marker="s", zorder=3)
        f2 = alias_of(f0, 512)
        ax.scatter([f2], [ypos["512 Hz"]], s=55, color=COL[name], marker="D", zorder=3)
        ax.annotate("", xy=(f2, ypos["512 Hz"] + 0.05), xytext=(f0, ypos["原始"] - 0.08),
                    arrowprops=dict(arrowstyle="->", color=COL[name], lw=1.0, alpha=0.5))
        ax.text(f2, ypos["512 Hz"] - 0.14, f"{f2:.0f}", color=COL[name],
                ha="center", va="top", fontsize=7.6)
    ax.axvline(256, color=COL["G4"], lw=1.0, ls="--", alpha=0.8)
    ax.axvline(500, color=ACCENT, lw=1.0, ls="--", alpha=0.8)
    ax.text(252, 2.46, "$f_N$=256", color=COL["G4"], fontsize=7.6, ha="right")
    ax.text(496, 2.46, "$f_N$=500", color=ACCENT, fontsize=7.6, ha="right")
    ax.text(440, 1.10, "基频位置不变", color=ACCENT, fontsize=7.6, ha="left")
    ax.set_xlim(0, 560)
    ax.set_ylim(-0.55, 2.62)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["512 Hz 采样", "1 kHz 采样", "原始"])
    ax.set_xlabel("频率 $f$ / Hz")
    ax.set_title("基频在采样后的位置：512 Hz 时三个音全部偏离", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    x = np.arange(3)
    w = 0.34
    err1 = [alias_rows["1 kHz（直接抽取）"][n]["cents"] for n in NOTE_F]
    err2 = [alias_rows["512 Hz（直接抽取）"][n]["cents"] for n in NOTE_F]
    ax.axhspan(-50, 50, color=COL["E4"], alpha=0.10)
    ax.text(2.45, 62, "±50 cent 可察觉带", color=COL["E4"], fontsize=7.2, ha="right")
    b1 = ax.bar(x - w / 2, err1, w, color=ACCENT, label="1 kHz（基频仍在带内）")
    b2 = ax.bar(x + w / 2, err2, w, color=COL["G4"], label="512 Hz（基频混叠）")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + (35 if h >= 0 else -35),
                    f"{h:+.0f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=7.4)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(NOTE_F))
    ax.set_ylabel("音分误差 / cent")
    ax.set_title(f"基频峰位偏差：512 Hz 达 {min(err2):.0f} ~ {max(err2):.0f} cent", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=7.6)
    ax.set_ylim(-2250, 290)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 8 升采样时域
def fig_upsample_time(x512, zero, fir, lin, path):
    start = 1.0                     # 取稳态段，避免起振段幅度接近 0
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.8))
    ax = axes[0]
    n = int(0.010 * 512)
    s5 = int(start * 512)
    s10 = int(start * 1024)
    t5 = (np.arange(n) + s5) / 512 * 1000
    t10 = (np.arange(2 * n) + s10) / 1024 * 1000
    ax.plot(t10, fir[s10:s10 + 2 * n], color=COL["E4"], lw=1.4, label="FIR 内插（1024 Hz）")
    ax.plot(t10, lin[s10:s10 + 2 * n], color=PURPLE, lw=1.2, ls="--", label="线性内插（1024 Hz）")
    ax.plot(t5, x512[s5:s5 + n], "o", ms=4.5, mfc="white", mec=COL["G4"], mew=1.0,
            label="512 Hz 原采样点")
    ax.set_xlim(1000, 1010)
    ax.set_xlabel("时间 $t$ / ms")
    ax.set_ylabel("幅度")
    ax.set_title("512 Hz → 1024 Hz：FIR 与线性内插在采样点之间补值", loc="left")
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax = axes[1]
    n = int(0.008 * 512)
    s5 = int(start * 512)
    s10 = int(start * 1024)
    t5 = (np.arange(n) + s5) / 512 * 1000
    t10 = (np.arange(2 * n) + s10) / 1024 * 1000
    z = zero[s10:s10 + 2 * n]
    ax.vlines(t10, 0, z, color="#F59E0B", lw=1.0, alpha=0.9, label="零插值序列（含零值）")
    ax.plot(t10, z, "o", ms=3.0, color="#F59E0B", mfc="white", mew=0.8)
    ax.plot(t10, fir[s10:s10 + 2 * n], color=COL["E4"], lw=1.6, label="FIR 低通后")
    ax.plot(t5, x512[s5:s5 + n], "o", ms=5.0, mfc="white", mec=COL["G4"], mew=1.2,
            label="原 512 Hz 样本")
    ax.set_xlim(1000, 1008)
    ax.set_xlabel("时间 $t$ / ms")
    ax.set_ylabel("幅度")
    ax.set_title("局部：零插值在 1024 Hz 网格上产生零值，低通滤波器把冲激“铺开”为连续波形",
                 loc="left", fontsize=9.6)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 图 9 升采样频谱
def fig_upsample_spectrum(series, metrics, path):
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.2),
                             gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax = axes[0]
    styles = {
        "512 Hz 原信号": (COL["G4"], 1.4, "-"),
        "零插值（1024 Hz）": ("#F59E0B", 0.9, "-"),
        "FIR 内插（1024 Hz）": (COL["E4"], 1.1, "-"),
        "线性内插（1024 Hz）": (PURPLE, 1.0, "--"),
    }
    for key, (f, P) in series.items():
        c, lw, ls = styles[key]
        m = f <= 512
        ax.plot(f[m], db(P[m]), color=c, lw=lw, ls=ls, label=key)
    ax.axvspan(256, 512, color=COL["G4"], alpha=0.07)
    ax.text(384, -99, "镜像频带 256–512 Hz", color=COL["G4"], fontsize=8.5, ha="center")
    ax.axvline(256, color=COL["G4"], lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(0, 512)
    ax.set_ylim(-105, -10)
    ax.set_xlabel("频率 $f$ / Hz")
    ax.set_ylabel("功率 / dB")
    ax.set_title("512 Hz → 1024 Hz 的频谱：零插值产生镜像，FIR 低通抑制镜像", loc="left")
    ax.legend(loc="lower left", ncol=2, fontsize=8)

    ax = axes[1]
    for key, (f, P) in series.items():
        c, lw, ls = styles[key]
        m = f <= 300
        ax.plot(f[m], db(P[m]), color=c, lw=lw, ls=ls, label=key)
    ax.axvline(256, color=COL["G4"], lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(0, 300)
    ax.set_ylim(-100, -15)
    ax.set_xlabel("频率 $f$ / Hz")
    ax.set_ylabel("功率 / dB")
    ax.set_title("通带放大：三种内插都只能保留 0–256 Hz 的旧信息，不能恢复 44.1 kHz 中被折叠的高频", loc="left")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.text(0.99, 0.05, f"镜像抑制：FIR {metrics['image_suppression_fir_db']:.1f} dB，线性 {metrics['image_suppression_linear_db']:.1f} dB",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, color=GRAY)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 主流程
def main():
    # 清理上一次运行生成的图与音频，避免旧文件混入清单
    for p in list(AUDIO.glob("*.wav")) + list(PREVIEW.glob("*.wav")) + list(FIG.glob("fig*.png")):
        try:
            p.unlink()
        except OSError:
            pass

    wav_path = ROOT / "CEG.wav"
    x_all, fs0 = load_wav(wav_path)
    if fs0 != FS0:
        raise SystemExit(f"本实验按 44.1 kHz 设计，读到的 f_s={fs0} Hz，请先确认 CEG.wav。")
    seg = x_all[: int(round(DUR * fs0))]
    gain = 0.85 / (np.max(np.abs(seg)) + 1e-12)
    seg = seg * gain

    print(f"[i] CEG.wav: {len(x_all)/fs0:.3f} s, fs={fs0} Hz, 使用前 {len(seg)/fs0:.3f} s")
    write_wav(AUDIO / "01_original_44100.wav", seg, fs0)

    # ---- 原始信号分析
    fig_time(seg, fs0, FIG / "fig01_time_original.png")
    f0, a0 = fig_spectrum_original(seg, fs0, FIG / "fig02_spectrum_original.png")
    fig_spectrogram(seg, fs0, FIG / "fig03_spectrogram_original.png")

    fw, Pw = welch_spectrum(seg, fs0, nperseg=16384)
    notes = {}
    for name, f0n in NOTE_F.items():
        fp, pp = peak_near(fw, Pw, f0n, 15)
        harmonics = []
        for h in (1, 2, 3):
            fh, ph = peak_near(fw, Pw, h * fp, 8)
            harmonics.append(round(fh, 2))
        notes[name] = dict(f0_theory=round(f0n, 2), f0_measured=round(fp, 2),
                           cents=round(cents(fp, f0n), 1),
                           harmonics_hz=harmonics,
                           level_db=round(10 * math.log10(pp / np.max(Pw)), 2))

    # ---- 采样率实验
    # 满足采样定理：多相滤波抗混叠；不满足：故意直接抽取，以暴露混叠。
    cases = [
        ("44.1 kHz（原始）", seg, fs0, "原始", "全带"),
        ("22.05 kHz", resample_proper(seg, fs0, 22050), 22050, "多相抗混叠", "合法"),
        ("8 kHz", resample_proper(seg, fs0, 8000), 8000, "多相抗混叠", "合法"),
        ("1 kHz（直接抽取）", downsample_naive(seg, fs0, 1000), 1000, "直接抽取", "基频保留/谐波混叠"),
        ("512 Hz（直接抽取）", downsample_naive(seg, fs0, 512), 512, "直接抽取", "基频混叠"),
    ]
    audio_names = {
        "22.05 kHz": "02_22050_resampled.wav",
        "8 kHz": "03_8000_resampled.wav",
        "1 kHz（直接抽取）": "04_1000_naive.wav",
        "512 Hz（直接抽取）": "05_512_naive.wav",
    }
    for label, y, fs, kind, _ in cases[1:]:
        write_wav(AUDIO / audio_names[label], y, fs)

    y_aa = resample_proper(seg, fs0, 512)
    write_wav(AUDIO / "06_512_antialias.wav", y_aa, 512)

    # 频谱图用的 entry: (label, y, fs, kind, color)
    entries = [
        (cases[0][0], cases[0][1], cases[0][2], "原始", "#111827"),
        (cases[1][0], cases[1][1], cases[1][2], "合法，带宽 11.025 kHz", COL["C4"]),
        (cases[2][0], cases[2][1], cases[2][2], "合法，带宽 4 kHz", COL["E4"]),
        (cases[3][0], cases[3][1], cases[3][2], "基频保留，谐波混叠", ACCENT),
        (cases[4][0], cases[4][1], cases[4][2], "基频混叠", COL["G4"]),
        ("512 Hz（抗混叠重采样）", y_aa, 512, "基频大部分被滤除", TEAL),
    ]
    fig_spectra_full(entries, FIG / "fig05_spectra_full.png")
    fig_spectra_zoom(entries[:5], FIG / "fig06_spectra_zoom.png")
    fig_sampling_grid(seg, fs0,
                      [(cases[0][0], cases[0][1], cases[0][2], "全带"),
                       (cases[2][0], cases[2][1], cases[2][2], "Nyquist 4 kHz，前 12 个谐波基本保留"),
                       (cases[3][0], cases[3][1], cases[3][2], "Nyquist 500 Hz，基频在带内"),
                       (cases[4][0], cases[4][1], cases[4][2], "Nyquist 256 Hz，基频已越过 f_N")],
                      FIG / "fig04_sampling_grid.png")

    # 采样率实验的定量指标
    rate_rows = []
    for label, y, fs, kind, status in cases + [("512 Hz（抗混叠重采样）", y_aa, 512, "多相抗混叠", "基频大部分被滤除")]:
        f, P = welch_spectrum(y, fs)
        rate_rows.append(dict(
            label=label, fs=fs, nyquist=fs / 2, samples=len(y),
            duration=round(len(y) / fs, 4), kind=kind, status=status,
            rms=round(float(np.sqrt(np.mean(y ** 2))), 5),
            peak=round(float(np.max(np.abs(y))), 5),
            centroid_hz=round(centroid(f, P), 1),
            rolloff95_hz=round(rolloff(f, P), 1),
        ))
    for row in rate_rows:
        if row["fs"] <= 1000 and row["label"] != "512 Hz（抗混叠重采样）":
            row["alias_notes"] = {name: round(alias_of(f0n, row["fs"]), 2)
                                  for name, f0n in NOTE_F.items()}

    # 1 kHz / 512 Hz 基频实测与音分误差
    alias_rows = {}
    case_map = {label: (y, fs) for label, y, fs, _, _ in cases}
    for label in ("1 kHz（直接抽取）", "512 Hz（直接抽取）"):
        y, fs = case_map[label]
        f, P = welch_spectrum(y, fs)
        per_note = {}
        for name, f0n in NOTE_F.items():
            target = f0n if fs >= 1000 else alias_of(f0n, fs)
            fp, _ = peak_closest(f, P, target, 7 if fs >= 1000 else 7)
            per_note[name] = dict(
                f_theory=round(target, 2), f_measured=round(fp, 2),
                cents=round(cents(fp, f0n), 1),
            )
        alias_rows[label] = per_note

    # ---- 2 倍插值：512 Hz -> 1024 Hz
    x512 = cases[4][1]
    zero = upsample_zeros(x512, 2)
    fir = upsample_fir(x512, 512, 2)
    lin = upsample_linear(x512, 2)
    write_wav(AUDIO / "07_512_to_1024_fir.wav", fir, 1024)
    write_wav(AUDIO / "08_512_to_1024_linear.wav", lin, 1024)

    series = {}
    for key, y, fs in (("512 Hz 原信号", x512, 512),
                       ("零插值（1024 Hz）", zero, 1024),
                       ("FIR 内插（1024 Hz）", fir, 1024),
                       ("线性内插（1024 Hz）", lin, 1024)):
        series[key] = welch_spectrum(y, fs)

    e_base = band_energy(*series["512 Hz 原信号"], 0, 256)
    e_zero_base = band_energy(*series["零插值（1024 Hz）"], 0, 256)
    e_zero_img = band_energy(*series["零插值（1024 Hz）"], 256, 512)
    e_fir_base = band_energy(*series["FIR 内插（1024 Hz）"], 0, 256)
    e_fir_img = band_energy(*series["FIR 内插（1024 Hz）"], 256, 512)
    e_lin_base = band_energy(*series["线性内插（1024 Hz）"], 0, 256)
    e_lin_img = band_energy(*series["线性内插（1024 Hz）"], 256, 512)
    up_metrics = dict(
        fs_in=512, fs_out=1024, L=2,
        base_energy_512=round(e_base, 6),
        base_energy_zero=round(e_zero_base, 6),
        image_energy_zero=round(e_zero_img, 6),
        image_to_base_zero_db=round(10 * math.log10(e_zero_img / e_zero_base), 2),
        image_energy_fir=round(e_fir_img, 6),
        image_to_base_fir_db=round(10 * math.log10(max(e_fir_img, 1e-18) / e_fir_base), 2),
        image_suppression_fir_db=round(10 * math.log10(e_zero_img / max(e_fir_img, 1e-18)), 2),
        image_energy_linear=round(e_lin_img, 6),
        image_to_base_linear_db=round(10 * math.log10(max(e_lin_img, 1e-18) / e_lin_base), 2),
        image_suppression_linear_db=round(10 * math.log10(e_zero_img / max(e_lin_img, 1e-18)), 2),
        base_change_fir_db=round(10 * math.log10(e_fir_base / e_base), 3),
        base_change_linear_db=round(10 * math.log10(e_lin_base / e_base), 3),
        centroid_512=round(centroid(*series["512 Hz 原信号"]), 1),
        centroid_fir=round(centroid(*series["FIR 内插（1024 Hz）"]), 1),
        centroid_linear=round(centroid(*series["线性内插（1024 Hz）"]), 1),
    )
    fig_upsample_time(x512, zero, fir, lin, FIG / "fig08_upsample_time.png")
    fig_upsample_spectrum(series, up_metrics, FIG / "fig09_upsample_spectrum.png")

    # ---- 混叠音高图（放在频谱图之后）
    fig_alias_notes(alias_rows, FIG / "fig07_alias_notes.png")

    # ---- 网页试听版：统一 16 kHz，仅用于浏览器兼容
    preview_rows = []
    for p in sorted(AUDIO.glob("*.wav")):
        fs, d = wavfile.read(p)
        y = d.astype(np.float64) / 32768.0
        if fs != 16000:
            g = math.gcd(fs, 16000)
            y16 = signal.resample_poly(y, 16000 // g, fs // g)
            y16 = y16[: int(round(len(y) / fs * 16000))]
        else:
            y16 = y
        prev = PREVIEW / f"{p.stem}_preview16k.wav"
        write_wav(prev, y16, 16000)
        preview_rows.append(dict(source=p.name, preview=prev.name,
                                 source_fs=fs, preview_fs=16000,
                                 duration=round(len(y) / fs, 3)))

    # ---- 汇总
    summary = dict(
        source=str(wav_path), fs=fs0, source_duration_s=round(len(x_all) / fs0, 3),
        used_duration_s=round(len(seg) / fs0, 3), used_samples=len(seg),
        df_hz=round(fs0 / len(seg), 4), gain_applied=round(gain, 4),
        notes=notes, sampling_rate_experiments=rate_rows,
        undersampled_notes=alias_rows, upsample=up_metrics,
        figures=sorted(p.name for p in FIG.glob("fig*.png")),
        audio=[dict(file=p.name, fs=int(wavfile.read(p)[0]), duration=wav_duration(p))
               for p in sorted(AUDIO.glob("*.wav"))],
        preview=preview_rows,
    )
    (ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[done] figures:", len(summary["figures"]), " audio:", len(summary["audio"]))


if __name__ == "__main__":
    main()
