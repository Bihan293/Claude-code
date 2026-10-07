"""Configuration, paths and secret storage.

Layout (default ``~/.opus-agent``, override with ``OPUS_HOME``)::

    config.json        non-secret settings (safe to show)
    credentials.json   secrets, chmod 600, never printed / logged / committed
    sessions/          persisted conversations (resume)
    memory/            persistent memory (global + per project)
    checkpoints/       file backups for /undo
    logs/agent.log     redacted log
"""
from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

APP_NAME = "opus-agent"


def home_dir() -> Path:
    p = Path(os.environ.get("OPUS_HOME") or (Path.home() / ".opus-agent")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    for sub in ("sessions", "memory", "checkpoints", "logs"):
        (p / sub).mkdir(exist_ok=True)
    return p


DEFAULTS: dict[str, Any] = {
    "base_url": "https://tooken.club/v1",
    "model": "claude-opus-5-5",
    # model used for cheap auxiliary work (sub-agents exploration, summaries).
    # empty string -> use main model
    "small_model": "",
    "max_tokens": 32000,
    # thinking: "adaptive" | "enabled" | "off"
    "thinking": "adaptive",
    "thinking_budget": 12000,
    # effort hint for models that support output_config.effort ("" = don't send)
    "effort": "",
    "context_window": 200000,
    # when estimated context exceeds these fractions of context_window:
    "prune_threshold": 0.55,      # stub out old large tool results (cheap)
    "compact_threshold": 0.78,    # summarize whole history (one model call)
    "keep_recent_tool_results": 8,
    "tool_output_limit": 30000,   # chars returned to model per tool call
    "max_iterations": 300,        # model calls per task
    "stream": True,
    "auth_style": "both",         # "x-api-key" | "bearer" | "both"
    "anthropic_version": "2023-06-01",
    "anthropic_beta": "",         # optional comma separated beta headers
    "prompt_caching": True,
    "cache_ttl": "5m",            # "5m" | "1h"
    "request_timeout": 600,
    "connect_timeout": 30,
    "max_retries": 8,
    "bash_timeout": 600,
    # permission mode: "auto" (fully autonomous), "ask" (confirm writes/shell),
    # "readonly" (plan mode: no writes, no shell side effects)
    "permission_mode": "auto",
    "confirm_dangerous": True,
    "allow_push_to_default_branch": False,
    "wake_lock": True,            # termux-wake-lock while a task runs
    "termux_notifications": True,
    "telegram_enabled": False,
    "telegram_chat_id": "",
    "show_thinking": False,
    "projects_dir": "~/projects",
    "git_author_name": "",
    "git_author_email": "",
    # USD per million tokens (for cost estimation only)
    "price_input": 5.0,
    "price_output": 25.0,
    "price_cache_write": 6.25,
    "price_cache_read": 0.5,
}

SECRET_KEYS = ("api_key", "github_token", "telegram_bot_token")


def _atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Config:
    def __init__(self) -> None:
        self.home = home_dir()
        self.path = self.home / "config.json"
        self.cred_path = self.home / "credentials.json"
        self.data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text("utf-8")))
            except (OSError, ValueError):
                pass
        self._creds: dict[str, str] = {}
        self._load_creds()

    # ---------------- settings ----------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def set(self, key: str, value: Any) -> None:
        if key in SECRET_KEYS:
            raise ValueError("secrets must be set via set_secret()")
        if key in DEFAULTS:
            value = coerce(value, DEFAULTS[key])
        self.data[key] = value
        self.save()

    def save(self) -> None:
        clean = {k: v for k, v in self.data.items() if k not in SECRET_KEYS}
        _atomic_write(self.path, json.dumps(clean, indent=2, ensure_ascii=False))

    # ---------------- secrets ----------------
    def _load_creds(self) -> None:
        if self.cred_path.exists():
            try:
                st = self.cred_path.stat()
                if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                    os.chmod(self.cred_path, 0o600)
                self._creds = json.loads(self.cred_path.read_text("utf-8"))
            except (OSError, ValueError):
                self._creds = {}

    def secret(self, name: str) -> str:
        env = {
            "api_key": ("TOOKEN_API_KEY", "OPUS_API_KEY", "ANTHROPIC_API_KEY"),
            "github_token": ("GITHUB_TOKEN", "GH_TOKEN"),
            "telegram_bot_token": ("TELEGRAM_BOT_TOKEN",),
        }.get(name, ())
        for e in env:
            if os.environ.get(e):
                return os.environ[e].strip()
        return (self._creds.get(name) or "").strip()

    def set_secret(self, name: str, value: str) -> None:
        if value:
            self._creds[name] = value.strip()
        else:
            self._creds.pop(name, None)
        _atomic_write(self.cred_path, json.dumps(self._creds, indent=2), mode=0o600)
        from .security import register_secret
        register_secret(value)

    def all_secrets(self) -> list[str]:
        return [s for s in (self.secret(k) for k in SECRET_KEYS) if s]

    def public_view(self) -> dict[str, Any]:
        out = {k: v for k, v in self.data.items() if k not in SECRET_KEYS}
        for k in SECRET_KEYS:
            out[k] = "set" if self.secret(k) else "not set"
        return out


def coerce(value: Any, like: Any) -> Any:
    if isinstance(value, str):
        if isinstance(like, bool):
            return value.strip().lower() in ("1", "true", "yes", "on", "y")
        if isinstance(like, int) and not isinstance(like, bool):
            return int(value)
        if isinstance(like, float):
            return float(value)
    return value
