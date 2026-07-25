"""Benchmarking Agent — thin wrapper over domain/benchmark."""
from __future__ import annotations

from advisor.domain.benchmark import BenchmarkResult, run_benchmarking as _run


def run_benchmarking(model_name: str, *, allow_live: bool = True) -> BenchmarkResult:
    return _run(model_name, allow_live=allow_live)
