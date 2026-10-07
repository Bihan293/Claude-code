"""Token usage ledger (per day / model), for `opus usage`."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def record_usage(home: Path, usage: dict[str, Any], cfg) -> None:
    if not usage:
        return
    p = home / "usage.json"
    day = time.strftime("%Y-%m-%d")
    with _lock:
        try:
            data = json.loads(p.read_text("utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            data = {}
        d = data.setdefault(day, {k: 0 for k in KEYS} | {"requests": 0})
        for k in KEYS:
            v = usage.get(k) or 0
            if isinstance(v, (int, float)):
                d[k] = d.get(k, 0) + int(v)
        d["requests"] = d.get("requests", 0) + 1
        try:
            p.write_text(json.dumps(data, indent=1), "utf-8")
        except OSError:
            pass


def load(home: Path) -> dict[str, dict[str, int]]:
    p = home / "usage.json"
    try:
        return json.loads(p.read_text("utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def cost(d: dict[str, int], cfg) -> float:
    return (d.get("input_tokens", 0) * cfg.get("price_input", 5.0)
            + d.get("output_tokens", 0) * cfg.get("price_output", 25.0)
            + d.get("cache_creation_input_tokens", 0) * cfg.get("price_cache_write", 6.25)
            + d.get("cache_read_input_tokens", 0) * cfg.get("price_cache_read", 0.5)) / 1e6


def cache_hit_rate(d: dict[str, int]) -> float:
    total = d.get("input_tokens", 0) + d.get("cache_creation_input_tokens", 0) + d.get("cache_read_input_tokens", 0)
    return d.get("cache_read_input_tokens", 0) / total if total else 0.0
