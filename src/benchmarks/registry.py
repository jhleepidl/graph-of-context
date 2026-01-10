from __future__ import annotations
from typing import Dict, Type
from .base import Benchmark
from .synthetic_browsecomp import SyntheticBrowseComp

BENCHMARKS: Dict[str, Type[Benchmark]] = {
    "synthetic_browsecomp": SyntheticBrowseComp,
}

def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]()
