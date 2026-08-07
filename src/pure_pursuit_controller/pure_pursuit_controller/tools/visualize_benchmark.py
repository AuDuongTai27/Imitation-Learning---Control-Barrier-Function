#!/usr/bin/env python3
"""
visualize_benchmark.py
──────────────────────
Visualize CSV sinh ra bởi benchmark_frequency.py

Cách dùng:
  python3 visualize_benchmark.py                          # tự tìm file mới nhất
  python3 visualize_benchmark.py benchmark_frequency_xxx.csv
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ─── Tự tìm file CSV mới nhất nếu không truyền argument ───────────────────────
CSV_DIR = os.path.expanduser("~/f1_ws/benchmark_results")

if len(sys.argv) >= 2:
    csv_path = sys.argv[1]
else:
    files = sorted(glob.glob(os.path.join(CSV_DIR, "benchmark_frequency_*.csv")))
    if not files:
        print(f"[ERROR] Không tìm thấy CSV nào trong {CSV_DIR}")
        sys.exit(1)
    csv_path = files[-1]
    print(f"[INFO] Tự động chọn file: {csv_path}")

df = pd.read_csv(csv_path)
df["time_s"] = df["wall_time_s"] - df["wall_time_s"].iloc[0]   # relative time
t = df["time_s"].to_numpy()

# ─── Layout: 3 hàng ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0f0f17")
fig.suptitle(
    f"Real-Time Frequency Benchmark\n{os.path.basename(csv_path)}",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 1, hspace=0.55, top=0.91, bottom=0.07,
                       left=0.08, right=0.97)

COLORS = {
    "ai_avg":  "#00e5ff",   # cyan
    "ai_band": "#00e5ff33",
    "cbf_avg": "#ff6f61",   # coral
    "cbf_band":"#ff6f6133",
    "lat":     "#b9f542",   # lime
    "grid":    "#2a2a3d",
    "text":    "#ccccdd",
}

STYLE = dict(facecolor="#14141f", framealpha=1)

def style_ax(ax, ylabel, title):
    ax.set_facecolor("#14141f")
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color("white")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(color=COLORS["grid"], linewidth=0.7, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ─── Plot 1: AI Hz ────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(t, df["ai_hz_avg"].to_numpy(), color=COLORS["ai_avg"],
         linewidth=2, label="AI avg (Hz)", zorder=3)
ax1.fill_between(t, df["ai_hz_min"].to_numpy(), df["ai_hz_max"].to_numpy(),
                 color=COLORS["ai_band"], label="AI min-max range")
ax1.axhline(df["ai_hz_avg"].mean(), color=COLORS["ai_avg"],
            linestyle=":", linewidth=1, alpha=0.7,
            label=f"Mean = {df['ai_hz_avg'].mean():.1f} Hz")
ax1.legend(fontsize=8, facecolor="#1e1e2f", labelcolor=COLORS["text"],
           framealpha=0.8, loc="upper right")
style_ax(ax1, "Frequency (Hz)", "AI Inference  (/drive_raw)")

# ─── Plot 2: CBF Hz ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, df["cbf_hz_avg"].to_numpy(), color=COLORS["cbf_avg"],
         linewidth=2, label="CBF avg (Hz)", zorder=3)
ax2.fill_between(t, df["cbf_hz_min"].to_numpy(), df["cbf_hz_max"].to_numpy(),
                 color=COLORS["cbf_band"], label="CBF min-max range")
ax2.axhline(df["cbf_hz_avg"].mean(), color=COLORS["cbf_avg"],
            linestyle=":", linewidth=1, alpha=0.7,
            label=f"Mean = {df['cbf_hz_avg'].mean():.1f} Hz")
ax2.legend(fontsize=8, facecolor="#1e1e2f", labelcolor=COLORS["text"],
           framealpha=0.8, loc="upper right")
style_ax(ax2, "Frequency (Hz)", "CBF Safety Filter  (/drive)")

# ─── Plot 3: Latency AI → CBF ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
lat_series = df["latency_ai2cbf_ms_avg"].dropna()
t_lat = t[lat_series.index.to_numpy()]
lat = lat_series.to_numpy()
ax3.plot(t_lat, lat, color=COLORS["lat"], linewidth=2,
         label="Latency AI->CBF (ms)", zorder=3)
ax3.fill_between(t_lat, 0, lat, color=COLORS["lat"] + "33")
mean_lat = lat.mean()
ax3.axhline(mean_lat, color=COLORS["lat"], linestyle=":",
            linewidth=1, alpha=0.8,
            label=f"Mean = {mean_lat:.2f} ms")
ax3.legend(fontsize=8, facecolor="#1e1e2f", labelcolor=COLORS["text"],
           framealpha=0.8, loc="upper right")
style_ax(ax3, "Latency (ms)", "Pipeline Latency  AI -> CBF")

# ─── Summary box ──────────────────────────────────────────────────────────────
summary = (
    f"AI:   avg={df['ai_hz_avg'].mean():.1f} Hz  "
    f"min={df['ai_hz_min'].min():.1f}  max={df['ai_hz_max'].max():.1f}\n"
    f"CBF:  avg={df['cbf_hz_avg'].mean():.1f} Hz  "
    f"min={df['cbf_hz_min'].min():.1f}  max={df['cbf_hz_max'].max():.1f}\n"
    f"Lat:  avg={mean_lat:.2f} ms  "
    f"max={lat.max():.2f} ms  samples={len(lat)}"
)
fig.text(0.50, 0.005, summary,
         ha="center", va="bottom", fontsize=8.5,
         color=COLORS["text"], fontfamily="monospace",
         bbox=dict(facecolor="#1a1a2e", edgecolor="#333355",
                   boxstyle="round,pad=0.4"))

# ─── Save & Show ──────────────────────────────────────────────────────────────
out_path = csv_path.replace(".csv", "_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"[SAVED] {out_path}")
plt.show()
