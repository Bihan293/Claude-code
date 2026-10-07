"""Session persistence: every step is saved atomically, so a killed Termux
process (OOM, swipe-away, reboot) can be resumed with `opus --continue`."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import _atomic_write


@dataclass
class Session:
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4])
    cwd: str = ""
    project_key: str = ""
    title: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    status: str = "idle"           # idle | running | interrupted | done | error
    messages: list[dict[str, Any]] = field(default_factory=list)
    system: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    turn: int = 0
    compactions: int = 0
    last_prompt_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------
    @staticmethod
    def dir(home: Path) -> Path:
        d = home / "sessions"
        d.mkdir(exist_ok=True)
        return d

    def save(self, home: Path) -> None:
        self.updated = time.time()
        _atomic_write(self.dir(home) / f"{self.id}.json", json.dumps(asdict(self), ensure_ascii=False), mode=0o600)

    @classmethod
    def load(cls, home: Path, sid: str) -> "Session":
        p = cls.dir(home) / f"{sid}.json"
        if not p.exists():
            matches = sorted(cls.dir(home).glob(f"{sid}*.json"))
            if not matches:
                raise FileNotFoundError(f"session {sid} not found")
            p = matches[-1]
        data = json.loads(p.read_text("utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        s = cls(**known)
        s.repair()
        return s

    @classmethod
    def list(cls, home: Path, project_key: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for p in sorted(cls.dir(home).glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("meta", {}).get("subagent"):
                continue
            if project_key and d.get("project_key") != project_key:
                continue
            out.append({k: d.get(k) for k in ("id", "title", "status", "cwd", "updated", "turn", "usage")})
            if len(out) >= limit:
                break
        return out

    @classmethod
    def latest(cls, home: Path, project_key: str | None = None) -> "Session | None":
        lst = cls.list(home, project_key, limit=1)
        return cls.load(home, lst[0]["id"]) if lst else None

    # ---------------------------------------------------------------
    def repair(self) -> None:
        """Make message list valid for the API after an abrupt stop:
        every tool_use needs a matching tool_result in the next user message,
        roles must alternate, no empty assistant messages."""
        msgs = [m for m in self.messages if m.get("content") not in (None, "", [])]
        fixed: list[dict[str, Any]] = []
        for m in msgs:
            if fixed and fixed[-1]["role"] == m["role"]:
                # merge same-role neighbours
                a = _as_blocks(fixed[-1]["content"])
                b = _as_blocks(m["content"])
                fixed[-1] = {"role": m["role"], "content": a + b}
            else:
                fixed.append(m)
        out: list[dict[str, Any]] = []
        for i, m in enumerate(fixed):
            out.append(m)
            if m["role"] != "assistant":
                continue
            uses = [b["id"] for b in _as_blocks(m["content"]) if b.get("type") == "tool_use"]
            if not uses:
                continue
            nxt = fixed[i + 1] if i + 1 < len(fixed) else None
            have = set()
            if nxt and nxt["role"] == "user":
                have = {b.get("tool_use_id") for b in _as_blocks(nxt["content"]) if b.get("type") == "tool_result"}
            missing = [u for u in uses if u not in have]
            if missing:
                stub = [{"type": "tool_result", "tool_use_id": u, "is_error": True,
                         "content": "Interrupted: the tool did not finish (agent was stopped). Re-check state."}
                        for u in missing]
                if nxt and nxt["role"] == "user":
                    nxt["content"] = stub + _as_blocks(nxt["content"])
                else:
                    out.append({"role": "user", "content": stub})
        # results must come first in a user message following tool_use
        for m in out:
            if m["role"] == "user" and isinstance(m["content"], list):
                tr = [b for b in m["content"] if b.get("type") == "tool_result"]
                other = [b for b in m["content"] if b.get("type") != "tool_result"]
                m["content"] = tr + other
        while out and out[0]["role"] != "user":
            out.pop(0)
        self.messages = out


def _as_blocks(c: Any) -> list[dict[str, Any]]:
    if isinstance(c, str):
        return [{"type": "text", "text": c}] if c else []
    return list(c or [])
