"""Tool registry and execution context."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..security import log, redact


class ToolError(Exception):
    """Raised by tools for expected failures; message is shown to the model."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    func: Callable[..., str]
    writes: bool = False        # modifies files / state
    shell: bool = False         # executes commands
    readonly_ok: bool = True    # allowed in plan/readonly mode
    subagent_ok: bool = True    # available to sub-agents

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None,
         **flags: Any) -> Callable[[Callable[..., str]], Callable[..., str]]:
    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        schema = {"type": "object", "properties": properties, "required": required or []}
        REGISTRY[name] = Tool(name, description.strip(), schema, fn, **flags)
        return fn
    return deco


@dataclass
class ToolContext:
    cfg: Any
    cwd: Path
    session: Any = None
    ui: Any = None
    checkpoints: Any = None
    memory: Any = None
    llm: Any = None
    agent_factory: Any = None
    interactive: bool = True
    is_subagent: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    read_files: dict[str, float] = field(default_factory=dict)  # path -> mtime when read
    touched_files: set[str] = field(default_factory=set)
    pr_urls: list[str] = field(default_factory=list)

    def resolve(self, p: str) -> Path:
        if not p:
            raise ToolError("path is required")
        path = Path(os.path.expanduser(p))
        if not path.is_absolute():
            path = self.cwd / path
        return Path(os.path.normpath(str(path)))

    def env(self) -> dict[str, str]:
        from .gitauth import git_env
        return git_env(self.cfg)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - limit
    return (text[:head] + f"\n\n... [{omitted} chars truncated; narrow your query or use offset/limit] ...\n\n"
            + text[-tail:])


def run_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    """Execute a tool; returns (output, is_error). Never raises."""
    t = REGISTRY.get(name)
    if t is None:
        return f"Unknown tool '{name}'. Available: {', '.join(sorted(REGISTRY))}", True
    if "__invalid_json__" in args:
        return ("Your tool input was not valid JSON (probably truncated because output was too long). "
                "Retry with smaller input, e.g. write large files in several edit steps."), True
    mode = ctx.cfg.get("permission_mode", "auto")
    if mode == "readonly" and not t.readonly_ok:
        return (f"Tool '{name}' is blocked: agent is in read-only plan mode. "
                "Present a plan instead; the user will switch to auto mode to execute."), True
    try:
        if ctx.ui is not None and not ctx.ui.permit(ctx, t, args):
            return "User denied this action. Choose a different approach or ask the user.", True
        out = t.func(ctx, **args)
        if isinstance(out, list):  # rich content blocks (images)
            return out, False
        out = out if isinstance(out, str) else str(out)
        limit = int(ctx.cfg.get("tool_output_limit", 30000))
        return truncate(redact(out), limit), False
    except ToolError as e:
        return redact(f"Error: {e}"), True
    except TypeError as e:
        return redact(f"Error: bad arguments for {name}: {e}"), True
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s crashed", name)
        return redact(f"Error: {type(e).__name__}: {e}"), True
