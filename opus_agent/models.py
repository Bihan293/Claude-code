"""Model catalogue: short numbers for `/model`, API format per model.

Claude models talk the Anthropic Messages API (``/v1/messages``); GPT models are
served by Tooken Club through the OpenAI Chat Completions API
(``/v1/chat/completions``). The client converts automatically, so the agent loop,
tools, sessions and memory are identical for every model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    num: str
    id: str
    name: str
    api: str          # "anthropic" | "openai"
    note: str = ""


MODELS: list[ModelInfo] = [
    ModelInfo("1", "claude-opus-5-5", "Claude Opus 5.5", "anthropic", "самая умная, сложные задачи"),
    ModelInfo("2", "gpt-6.1-sol", "GPT-6.1 Sol", "openai", "OpenAI, код и reasoning"),
    ModelInfo("3", "claude-sonnet-5-5", "Claude Sonnet 5.5", "anthropic", "быстрее, для типовых задач"),
]

_ALIASES = {
    "opus": "claude-opus-5-5", "опус": "claude-opus-5-5",
    "gpt": "gpt-6.1-sol", "гпт": "gpt-6.1-sol", "sol": "gpt-6.1-sol", "gpt-6.1": "gpt-6.1-sol",
    "sonnet": "claude-sonnet-5-5", "сонет": "claude-sonnet-5-5", "соннет": "claude-sonnet-5-5",
}

_OPENAI_PREFIXES = ("gpt-", "gpt_", "o1", "o3", "o4", "chatgpt", "deepseek", "glm", "grok")


def resolve(choice: str) -> str | None:
    """'1' / 'opus' / 'claude-opus-5-5' -> model id. Unknown non-empty strings are returned
    as-is (custom model id); empty -> None."""
    c = (choice or "").strip()
    if not c:
        return None
    for m in MODELS:
        if c == m.num or c.lower() == m.id or c.lower() == m.name.lower():
            return m.id
    return _ALIASES.get(c.lower(), c)


def info(model_id: str) -> ModelInfo | None:
    return next((m for m in MODELS if m.id == model_id), None)


def api_for(model_id: str, override: str = "auto") -> str:
    if override in ("anthropic", "openai"):
        return override
    m = info(model_id)
    if m:
        return m.api
    return "openai" if (model_id or "").lower().startswith(_OPENAI_PREFIXES) else "anthropic"


def display_name(model_id: str) -> str:
    m = info(model_id)
    return f"{m.name} ({m.id})" if m else model_id


def menu(current: str) -> str:
    lines = ["Модели:"]
    for m in MODELS:
        mark = "  ← текущая" if m.id == current else ""
        lines.append(f"  {m.num}  {m.name}{mark}")
    if not info(current):
        lines.append(f"     текущая: {current} (своя)")
    return "\n".join(lines)
