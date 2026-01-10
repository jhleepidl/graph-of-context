from __future__ import annotations
from typing import Dict, Optional
import os

def load_dotenv(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """Load a very small subset of .env semantics (KEY=VALUE, comments, quotes).

    - Lines starting with # are ignored.
    - Empty lines ignored.
    - Values may be quoted with single or double quotes.
    - By default, existing environment variables are not overwritten.
    """
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if not override and k in os.environ:
                loaded[k] = os.environ[k]
                continue
            os.environ[k] = v
            loaded[k] = v
    return loaded

def getenv_any(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default
