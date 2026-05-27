"""Gera figura comparativa: 4 histogramas de speedups sobrepostos."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CONFIGS = [
    ("results_rule_based.json",       "Baseline (AST)",      "#888888"),
    ("results_gemini.json",            "Gemini 2.5 FL",       "#1f77b4"),
    ("results_groq_llama.json",        "Llama 3.3 70B",       "#ff7f0e"),
    ("results_github_gpt4omini.json",  "GPT-4o-mini",         "#2ca02c"),
]

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bins = [-1.5 + i*0.1 for i in range(31)]  # de -1.5 a 1.5 em log10

for fname, label, color in CONFIGS:
    data = json.load(open(RESULTS / fname))
    sps = [x["speedup"] for x in data["results"] if x["speedup"] > 0]
    log_sps = [math.log10(s) for s in sps]
    ax.hist(log_sps, bins=bins, alpha=0.45, label=f"{label} (n={len(sps)})",
            color=color, edgecolor=color, linewidth=0.6)

ax.axvline(0.0, color="red", linestyle="--", linewidth=1, label=r"sp = $1\times$")
ax.set_xlabel(r"$\log_{10}(\mathrm{speedup})$")
ax.set_ylabel("frequência")
ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
fig.tight_layout()
fig.savefig(RESULTS / "fig_comparison_hist.pdf", bbox_inches="tight", pad_inches=0.05)
print("salvo:", RESULTS / "fig_comparison_hist.pdf")
