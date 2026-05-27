"""Gera figuras a partir dos resultados experimentais.

Saidas:
  experiment/results/fig_speedup_hist.pdf  - histograma de speedups (log)
  experiment/results/fig_scatter.pdf       - scatter t_orig x t_opt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def plot_speedup_hist(results: list[dict], out: Path, label: str) -> None:
    # Sem titulo interno: o \caption{} do LaTeX cumpre esse papel.
    sps = [x["speedup"] for x in results if x["speedup"] > 0]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    log_sps = [math.log10(s) for s in sps]
    ax.hist(log_sps, bins=30, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1, label=r"speedup $=1\times$")
    ax.set_xlabel(r"$\log_{10}(\mathrm{speedup})$")
    ax.set_ylabel("frequency")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_scatter(results: list[dict], out: Path, label: str) -> None:
    pts = [
        (x["original_time_s"] * 1000, x["optimized_time_s"] * 1000)
        for x in results
        if x["original_time_s"] > 0 and x["optimized_time_s"] > 0
    ]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=14, alpha=0.6, edgecolor="black", linewidth=0.3)
    lim_min = min(min(xs), min(ys)) * 0.7
    lim_max = max(max(xs), max(ys)) * 1.4
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=0.8, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("original time (ms)")
    ax.set_ylabel("optimized time (ms)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--label", required=True, type=str)
    p.add_argument("--prefix", required=True, type=str, help="prefixo dos arquivos pdf")
    args = p.parse_args()
    data = json.load(open(args.input))
    results = data["results"]
    out_dir = HERE / "results"
    plot_speedup_hist(results, out_dir / f"{args.prefix}_hist.pdf", args.label)
    plot_scatter(results, out_dir / f"{args.prefix}_scatter.pdf", args.label)
    print(f"figuras salvas em {out_dir}/{args.prefix}_*.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
