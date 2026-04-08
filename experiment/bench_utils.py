"""Utilitarios de validacao e benchmark para funcoes Python.

Executa codigo em um namespace isolado, roda a funcao de teste fornecida
pelo HumanEval (`check(candidate)`) e mede o tempo de execucao minimo
em um numero configuravel de repeticoes.
"""
from __future__ import annotations

import re
import signal
import time
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError_("execucao excedeu o tempo limite")


class _timeout:
    """Context manager para limitar tempo de execucao via SIGALRM."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __enter__(self) -> None:
        self._old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old)


@dataclass
class ValidationResult:
    passed: bool
    error: str | None = None


def _build_namespace(code: str) -> dict[str, Any]:
    """Compila e executa o codigo em um namespace isolado."""
    ns: dict[str, Any] = {}
    exec(compile(code, "<llm-code>", "exec"), ns)
    return ns


def validate(code: str, test_code: str, entry_point: str, timeout_s: float = 10.0) -> ValidationResult:
    """Executa o codigo e roda a funcao check(candidate) do HumanEval."""
    try:
        with _timeout(timeout_s):
            ns = _build_namespace(code)
            if entry_point not in ns:
                return ValidationResult(False, f"entry_point {entry_point!r} nao definido")
            exec(compile(test_code, "<test>", "exec"), ns)
            if "check" not in ns:
                return ValidationResult(False, "funcao check(candidate) ausente nos testes")
            ns["check"](ns[entry_point])
        return ValidationResult(True)
    except TimeoutError_ as e:
        return ValidationResult(False, f"timeout: {e}")
    except AssertionError as e:
        return ValidationResult(False, f"assertion: {e}")
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"{type(e).__name__}: {e}")


@dataclass
class BenchmarkResult:
    min_time_s: float
    mean_time_s: float
    runs: int
    error: str | None = None


_CANDIDATE_CALL_RE = re.compile(r"candidate\s*\(")


def _extract_candidate_calls(test_code: str) -> list[str]:
    """Extrai todas as chamadas candidate(...) do corpo dos testes.

    Retorna uma lista de strings como "candidate(1, 2)" para serem
    reutilizadas como workload de benchmark.
    """
    calls: list[str] = []
    for m in _CANDIDATE_CALL_RE.finditer(test_code):
        start = m.start()
        depth = 0
        i = m.end() - 1
        while i < len(test_code):
            ch = test_code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    calls.append(test_code[start : i + 1])
                    break
            i += 1
    return calls


def benchmark(
    code: str,
    test_code: str,
    entry_point: str,
    repeats: int = 5,
    workload_multiplier: int = 200,
    timeout_s: float = 15.0,
) -> BenchmarkResult:
    """Mede o tempo de execucao repetindo as chamadas dos testes.

    Estrategia: constroi um workload reproduzivel a partir das chamadas
    `candidate(...)` presentes no teste oficial do HumanEval, executa-o
    `workload_multiplier` vezes e repete `repeats` medicoes, reportando
    o tempo minimo e medio.
    """
    try:
        with _timeout(timeout_s):
            ns = _build_namespace(code)
            if entry_point not in ns:
                return BenchmarkResult(0.0, 0.0, 0, f"entry_point {entry_point!r} ausente")
            ns["candidate"] = ns[entry_point]

            calls = _extract_candidate_calls(test_code)
            if not calls:
                return BenchmarkResult(0.0, 0.0, 0, "nenhuma chamada candidate(...) encontrada")

            workload_src = "\n".join(calls)
            workload = compile(workload_src, "<workload>", "exec")

            # warmup
            for _ in range(2):
                exec(workload, ns)

            times: list[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                for _ in range(workload_multiplier):
                    exec(workload, ns)
                times.append(time.perf_counter() - t0)
            return BenchmarkResult(min(times), sum(times) / len(times), repeats)
    except TimeoutError_ as e:
        return BenchmarkResult(0.0, 0.0, 0, f"timeout: {e}")
    except Exception as e:  # noqa: BLE001
        return BenchmarkResult(0.0, 0.0, 0, f"{type(e).__name__}: {e}")
