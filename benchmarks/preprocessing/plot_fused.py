"""Throughput and peak RAM of the preprocessing chain variants, from the 1 s rows of bench_fused.py."""

import sys

import matplotlib.pyplot as plt

PROBE_MS_PER_S = 30_000 * 384 / 1e6
GREY, ACCENT = "#b0b0b0", "#c8551b"
rows = [  # label, MS/s, peak RAM above chunk (% of chunk), fused
    ("Default chain", 22, 1826, False),
    ("FIR steps, 1 thread", 83, 569, False),
    ("FIR steps, 8 threads", 422, 628, False),
    ("Fused, 1 thread", 113, 376, True),
    ("Fused, 8 threads", 644, 404, True),
]
labels = [r[0] for r in rows]
colors = [ACCENT if r[3] else GREY for r in rows]
probes = [r[1] / PROBE_MS_PER_S for r in rows]
ram = [r[2] / 100 for r in rows]

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
for ax, values, fmt, heading in (
    (ax1, probes, None, "Probes processed in real time"),
    (ax2, ram, "{:.1f}×", "Peak RAM, multiples of the chunk"),
):
    ax.barh(labels, values, color=colors, height=0.6)
    for y, v in enumerate(values):
        text = (f"{v:.1f}" if v < 10 else f"{v:.0f}") if fmt is None else fmt.format(v)
        ax.text(v, y, "  " + text, va="center", fontsize=9)
    ax.set_title(heading, fontsize=10, loc="left", color="#444")
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(values) * 1.18)
ax1.invert_yaxis()
fig.tight_layout()
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "fused.png", dpi=150)
