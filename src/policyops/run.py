from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_real_module() -> object:
    root = Path(__file__).resolve().parents[2]
    real_src = root / "src" / "benchmarks" / "policyops_arena_v0" / "src"
    if str(real_src) not in sys.path:
        sys.path.insert(0, str(real_src))
    sys.modules.pop("policyops", None)
    return importlib.import_module("policyops.run")


def main() -> None:
    module = _load_real_module()
    if hasattr(module, "main"):
        module.main()
    else:
        raise RuntimeError("Loaded policyops run module has no main()")


if __name__ == "__main__":
    main()
