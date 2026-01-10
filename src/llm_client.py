from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class LLMResponse:
    text: str
    tool_calls: List[Dict[str, Any]]  # optional, not used in this repo by default
    usage: Optional[Dict[str, Any]] = None  # e.g., {"input_tokens":..., "output_tokens":..., "total_tokens":...}

class LLMClient:
    """Adapter interface for plugging in a real LLM."""
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        raise NotImplementedError
