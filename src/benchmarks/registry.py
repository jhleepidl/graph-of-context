from __future__ import annotations
from typing import Dict, Type
from .base import Benchmark
from .synthetic_browsecomp import SyntheticBrowseComp
from .lost_in_middle import LostInMiddle
from .hotpotqa import HotpotQA
from .fever_prepared import FeverPrepared

BENCHMARKS: Dict[str, Type[Benchmark]] = {
    "synthetic_browsecomp": SyntheticBrowseComp,
    "lost_in_middle": LostInMiddle,
    "hotpotqa": HotpotQA,
    "fever_prepared": FeverPrepared,
}

def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]()
