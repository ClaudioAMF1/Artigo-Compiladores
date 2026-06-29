"""Generate a single combined scatter-panel figure (3 stacked subplots)
for the Gemini, Llama, and GPT-4o-mini back-ends, replacing the three
separate scatter figures that left large vertical gaps in the article.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

CONFIGS = [
    ("results_gemini.json", "(a) Gemini 2.5 Flash Lite", "#1f77b4"),
    ("results_groq_llama.json", "(b) Llama 3.3 70B (Groq)", "#ff7f0e"),
    ("results_github_gpt4omini.json", "(c) GPT-4o-mini (GitHub Models)", "#2ca02c"),
]

# One column wide, three rows tall but compact.
fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.6), sharex=False)

for ax, (fname, label, color) in zip(axes, CONFIGS):
    data = json.load(open(RES / fname))["results"]
    pts = [
        (x["original_time_s"] * 1000, x["optimized_time_s"] * 1000)
        for x in data
        if x["original_time_s"] > 0 and x["optimized_time_s"] > 0
    ]
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=10, alpha=0.55, color=color,
               edgecolor="black", linewidth=0.25)
    lo = min(min(xs), min(ys)) * 0.7
    hi = max(max(xs), max(ys)) * 1.4
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("optimized (ms)", fontsize=8)
    ax.set_title(label, fontsize=8.5, pad=3)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

axes[-1].set_xlabel("original time (ms)", fontsize=8)
fig.tight_layout(h_pad=1.0)
out = RES / "fig_scatter_panel.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
print("saved:", out)
