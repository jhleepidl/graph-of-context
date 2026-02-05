from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_event(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False))
        handle.write("\n")


def build_event(
    run_id: str,
    task_id: str,
    method: str,
    step: int,
    event_type: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "task_id": task_id,
        "method": method,
        "step": step,
        "event_type": event_type,
        "payload": payload,
    }
