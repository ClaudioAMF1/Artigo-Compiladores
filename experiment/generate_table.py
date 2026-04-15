"""Gera snippets LaTeX das tabelas de resultados para inclusao no artigo.

Uso:
    python experiment/generate_table.py \
        --results experiment/results/results_rule_based.json \
        --out experiment/results/tables.tex
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def build_tables(data: dict) -> str:
    results = data["results"]
    n = len(results)
    valid = [r for r in results if r["optimized_valid"]]
    sp = [r["speedup"] for r in valid if r["speedup"] > 0]

    n_improved = sum(1 for s in sp if s > 1.05)
    n_regressed = sum(1 for s in sp if s < 0.95)
    n_neutral = sum(1 for s in sp if 0.95 <= s <= 1.05)

    mean_sp = statistics.mean(sp) if sp else 0.0
    median_sp = statistics.median(sp) if sp else 0.0
    hmean_sp = statistics.harmonic_mean(sp) if sp else 0.0
    max_sp = max(sp) if sp else 0.0
    min_sp = min(sp) if sp else 0.0

    rules_applied = [r for r in results if r["rules_applied"]]
    sp_rule = [r["speedup"] for r in rules_applied if r["speedup"] > 0]
    rule_counts: Counter[str] = Counter()
    for r in results:
        for k in r["rules_applied"]:
            rule_counts[k] += 1

    backend = data.get("backend", "?")
    out: list[str] = []

    # --- Tabela 1: metricas agregadas ---
    out.append(
        r"""
\begin{table}[htbp]
\caption{Resultados agregados sobre os 164 problemas do HumanEval (backend: %s).}
\label{tab:agregado}
\centering
\begin{tabular}{lr}
\hline
\textbf{Métrica} & \textbf{Valor} \\
\hline
Problemas avaliados & %d \\
Preservação semântica & %d/%d (%.1f\%%) \\
Problemas melhorados ($>$1{,}05$\times$) & %d \\
Problemas neutros & %d \\
Regressões ($<$0{,}95$\times$) & %d \\
Regras disparadas (pelo menos uma) & %d \\
\hline
Speedup médio aritmético & %.3f$\times$ \\
Speedup mediano & %.3f$\times$ \\
Speedup médio harmônico & %.3f$\times$ \\
Speedup máximo & %.3f$\times$ \\
Speedup mínimo & %.3f$\times$ \\
\hline
\end{tabular}
\end{table}
""".strip()
        % (
            _escape(backend),
            n,
            len(valid),
            n,
            len(valid) / n * 100 if n else 0.0,
            n_improved,
            n_neutral,
            n_regressed,
            len(rules_applied),
            mean_sp,
            median_sp,
            hmean_sp,
            max_sp,
            min_sp,
        )
    )

    # --- Tabela 2: detalhe dos casos onde alguma regra foi aplicada ---
    detail_rows = []
    for r in sorted(rules_applied, key=lambda x: -x["speedup"]):
        rules = ", ".join(sorted(set(r["rules_applied"])))
        detail_rows.append(
            rf"{_escape(r['task_id'])} & {_escape(r['entry_point'])} & "
            rf"{_escape(rules)} & {r['original_time_s']*1000:.2f} & "
            rf"{r['optimized_time_s']*1000:.2f} & {r['speedup']:.3f}$\times$ \\"
        )

    out.append(
        r"""
\begin{table}[htbp]
\caption{Detalhe dos problemas nos quais alguma regra do otimizador determinístico foi disparada.}
\label{tab:detalhe}
\centering
\footnotesize
\begin{tabular}{lllrrr}
\hline
\textbf{Task} & \textbf{Função} & \textbf{Regras} & \textbf{Orig. (ms)} & \textbf{Otim. (ms)} & \textbf{Speedup} \\
\hline
%s
\hline
\end{tabular}
\end{table}
""".strip()
        % ("\n".join(detail_rows))
    )

    # --- Tabela 3: contagem de regras ---
    rule_rows = "\n".join(
        rf"{_escape(k)} & {v} \\" for k, v in rule_counts.most_common()
    )
    out.append(
        r"""
\begin{table}[htbp]
\caption{Frequência de aplicação das regras do otimizador determinístico.}
\label{tab:regras}
\centering
\begin{tabular}{lr}
\hline
\textbf{Regra} & \textbf{Aplicações} \\
\hline
%s
\hline
\end{tabular}
\end{table}
""".strip()
        % rule_rows
    )

    return "\n\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    latex = build_tables(data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(latex, encoding="utf-8")
    print(latex)
    print(f"[info] snippet salvo em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
