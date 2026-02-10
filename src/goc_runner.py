from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class GoCRunContext:
    task: Any
    env: Any
    client: Any
    options: Dict[str, Any]


class BaseGoCRunner(ABC):
    """Benchmark-agnostic GoC runner interface.

    Benchmarks should provide a concrete runner that overrides `run`.
    """

    @abstractmethod
    def run(self, ctx: GoCRunContext) -> Any:
        raise NotImplementedError

