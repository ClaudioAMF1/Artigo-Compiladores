"""Pipeline experimental: HumanEval + LLM + validacao + benchmark.

Implementacao da arquitetura descrita na Secao IV do artigo:

  Ingestao -> Analise Estatica -> Modulo LLM -> Validacao -> Benchmark

Uso:
    python experiment/pipeline.py \
        --n 20 \
        --seed 42 \
        --model gpt-4o \
        --out experiment/results/results.json

A chave OPENAI_API_KEY deve estar definida no ambiente.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_utils import benchmark, validate  # noqa: E402
from optimizer import LLMOptimizer, extract_features  # noqa: E402

DATA_PATH = HERE / "data" / "HumanEval.jsonl.gz"


@dataclass
class Sample:
    task_id: str
    entry_point: str
    prompt: str
    canonical_solution: str
    test: str

    @property
    def full_code(self) -> str:
        return self.prompt + self.canonical_solution


def load_humaneval(path: Path = DATA_PATH) -> list[Sample]:
    samples: list[Sample] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(
                Sample(
                    task_id=obj["task_id"],
                    entry_point=obj["entry_point"],
                    prompt=obj["prompt"],
                    canonical_solution=obj["canonical_solution"],
                    test=obj["test"],
                )
            )
    return samples


@dataclass
class RunResult:
    task_id: str
    entry_point: str
    features: dict[str, Any]
    original_valid: bool
    original_error: str | None
    original_time_s: float
    optimized_valid: bool
    optimized_error: str | None
    optimized_time_s: float
    speedup: float
    llm_error: str | None
    llm_latency_s: float


def run_one(sample: Sample, opt: LLMOptimizer, repeats: int, workload: int) -> RunResult:
    # 1. Analise estatica
    feats = extract_features(sample.full_code, sample.entry_point)

    # 2. Validacao do codigo original
    v_orig = validate(sample.full_code, sample.test, sample.entry_point)

    # 3. Benchmark do original
    b_orig = benchmark(
        sample.full_code,
        sample.test,
        sample.entry_point,
        repeats=repeats,
        workload_multiplier=workload,
    )

    # 4. Modulo LLM
    llm_error: str | None = None
    llm_latency = 0.0
    optimized_code: str | None = None
    try:
        t0 = time.perf_counter()
        optimized_code, _ = opt.optimize(sample.full_code, sample.entry_point)
        llm_latency = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        llm_error = f"{type(e).__name__}: {e}"

    # 5. Validacao e benchmark do otimizado
    if optimized_code is None:
        return RunResult(
            task_id=sample.task_id,
            entry_point=sample.entry_point,
            features=asdict(feats),
            original_valid=v_orig.passed,
            original_error=v_orig.error,
            original_time_s=b_orig.min_time_s,
            optimized_valid=False,
            optimized_error="LLM nao retornou codigo",
            optimized_time_s=0.0,
            speedup=0.0,
            llm_error=llm_error,
            llm_latency_s=llm_latency,
        )

    v_opt = validate(optimized_code, sample.test, sample.entry_point)
    b_opt = benchmark(
        optimized_code,
        sample.test,
        sample.entry_point,
        repeats=repeats,
        workload_multiplier=workload,
    )

    speedup = 0.0
    if v_opt.passed and b_opt.min_time_s > 0 and b_orig.min_time_s > 0:
        speedup = b_orig.min_time_s / b_opt.min_time_s

    return RunResult(
        task_id=sample.task_id,
        entry_point=sample.entry_point,
        features=asdict(feats),
        original_valid=v_orig.passed,
        original_error=v_orig.error,
        original_time_s=b_orig.min_time_s,
        optimized_valid=v_opt.passed,
        optimized_error=v_opt.error,
        optimized_time_s=b_opt.min_time_s,
        speedup=speedup,
        llm_error=llm_error,
        llm_latency_s=llm_latency,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline LLM+HumanEval")
    parser.add_argument("--n", type=int, default=20, help="numero de problemas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workload", type=int, default=200)
    parser.add_argument(
        "--out",
        type=str,
        default=str(HERE / "results" / "results.json"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    all_samples = load_humaneval()
    print(f"[info] HumanEval carregado: {len(all_samples)} problemas")

    subset = random.sample(all_samples, args.n)
    print(f"[info] subset amostrado: {args.n} problemas (seed={args.seed})")

    opt = LLMOptimizer(model=args.model)

    results: list[RunResult] = []
    for i, s in enumerate(subset, start=1):
        print(f"[{i:>3}/{len(subset)}] {s.task_id} ({s.entry_point}) ... ", end="", flush=True)
        try:
            r = run_one(s, opt, args.repeats, args.workload)
        except Exception as e:  # noqa: BLE001
            print(f"ERRO {type(e).__name__}: {e}")
            continue
        results.append(r)
        if r.optimized_valid and r.speedup > 0:
            print(f"ok speedup={r.speedup:.3f}x ({r.original_time_s*1000:.2f}ms -> {r.optimized_time_s*1000:.2f}ms)")
        elif r.llm_error:
            print(f"llm_err: {r.llm_error}")
        else:
            print(f"inval: {r.optimized_error}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n": args.n, "seed": args.seed, "results": [asdict(r) for r in results]}, f, indent=2)
    print(f"[info] resultados salvos em {out_path}")

    # Resumo
    n_total = len(results)
    n_valid = sum(1 for r in results if r.optimized_valid)
    speedups = [r.speedup for r in results if r.optimized_valid and r.speedup > 0]
    if speedups:
        mean_sp = sum(speedups) / len(speedups)
        max_sp = max(speedups)
        n_improved = sum(1 for s in speedups if s > 1.05)
        n_regression = sum(1 for s in speedups if s < 0.95)
    else:
        mean_sp = max_sp = 0.0
        n_improved = n_regression = 0

    print("\n=== RESUMO ===")
    print(f"problemas rodados       : {n_total}")
    print(f"otimizacoes validas     : {n_valid} ({(n_valid / n_total * 100) if n_total else 0:.1f}%)")
    print(f"speedup medio (validos) : {mean_sp:.3f}x")
    print(f"speedup maximo          : {max_sp:.3f}x")
    print(f"problemas melhorados    : {n_improved}")
    print(f"regressoes detectadas   : {n_regression}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
