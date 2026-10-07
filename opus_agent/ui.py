"""Terminal UI: streaming output, tool call display, spinner, permissions."""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .security import redact
from .tools.shell import is_dangerous


def _short(v: Any, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace("\n", "⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, console: Console) -> None:
        self.console = console
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.label = "thinking"
        self.t0 = 0.0

    def start(self, label: str = "thinking") -> None:
        if not self.console.is_terminal:
            return
        self.stop()
        self.label = label
        self.t0 = time.time()
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            f = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r\x1b[2m{f} {self.label}… {time.time() - self.t0:.0f}s  (Ctrl+C to interrupt)\x1b[0m\x1b[K")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)
        sys.stdout.write("\r\x1b[K")
        sys.stdout.flush()

    def stop(self) -> None:
        if self._th and self._th.is_alive():
            self._stop.set()
            self._th.join(timeout=1)
        self._th = None


class UI:
    def __init__(self, cfg, verbose: bool = False, quiet: bool = False) -> None:
        self.cfg = cfg
        self.console = Console(highlight=False, soft_wrap=False)
        self.spinner = Spinner(self.console)
        self.verbose = verbose
        self.quiet = quiet
        self._streaming = False
        self._thinking_shown = False
        self._buf = ""
        self._shell_lines = 0
        self._lock = threading.RLock()
        self.approve_all = False

    @property
    def width(self) -> int:
        return max(40, min(shutil.get_terminal_size((80, 24)).columns, 160))

    # ---------------------------------------------------------------- stream
    def thinking_start(self) -> None:
        self.spinner.start("thinking")

    def thinking_stop(self) -> None:
        self.spinner.stop()

    def stream_text(self, delta: str) -> None:
        if self.quiet:
            self._buf += delta
            return
        with self._lock:
            self.spinner.stop()
            if self._thinking_shown:
                sys.stdout.write("\x1b[0m\n")
                self._thinking_shown = False
            if not self._streaming:
                sys.stdout.write("\n")
                self._streaming = True
            sys.stdout.write(delta)
            sys.stdout.flush()

    def stream_thinking(self, delta: str) -> None:
        if not self.cfg.get("show_thinking") or self.quiet:
            return
        with self._lock:
            self.spinner.stop()
            if not self._thinking_shown:
                sys.stdout.write("\n\x1b[2;3m✻ ")
                self._thinking_shown = True
            sys.stdout.write(delta)
            sys.stdout.flush()

    def end_stream(self) -> None:
        with self._lock:
            if self._thinking_shown:
                sys.stdout.write("\x1b[0m\n")
                self._thinking_shown = False
            if self._streaming:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._streaming = False

    def tool_preparing(self, name: str) -> None:
        with self._lock:
            if self._streaming:
                self.end_stream()
            self.spinner.start(f"preparing {name}")

    # ---------------------------------------------------------------- tools
    def tool_call(self, name: str, args: dict[str, Any], sub: bool = False) -> None:
        self.spinner.stop()
        if self.quiet:
            return
        pre = "  ↳ " if sub else ""
        main = ""
        for k in ("command", "path", "pattern", "action", "url", "query", "description", "prompt", "job_id"):
            if k in args:
                main = _short(args[k], self.width - 20 - len(name))
                break
        if name == "todo_write":
            main = f"{len(args.get('todos', []))} items"
        with self._lock:
            self.end_stream()
            t = Text(pre + "● ", style="bold cyan" if not sub else "cyan")
            t.append(name, style="bold")
            if main:
                t.append(f"({main})", style="dim" if sub else "")
            self.console.print(t)
            if self.verbose and name in ("edit_file", "multi_edit") and not sub:
                self._show_edit(args)
        self._shell_lines = 0
        if name == "bash" and not sub:
            self.spinner.start("running")

    def _show_edit(self, args: dict[str, Any]) -> None:
        edits = args.get("edits") or [args]
        for e in edits[:5]:
            for l in str(e.get("old_string", "")).splitlines()[:8]:
                self.console.print(Text("    - " + l, style="red"))
            for l in str(e.get("new_string", "")).splitlines()[:8]:
                self.console.print(Text("    + " + l, style="green"))

    def shell_line(self, line: str) -> None:
        if self.quiet or not self.verbose:
            return
        with self._lock:
            self._shell_lines += 1
            if self._shell_lines <= 40:
                self.spinner.stop()
                self.console.print(Text("    │ " + redact(line.rstrip())[: self.width - 8], style="dim"))
            elif self._shell_lines == 41:
                self.console.print(Text("    │ …", style="dim"))

    def tool_result(self, name: str, out: Any, is_err: bool, dur: float, sub: bool = False) -> None:
        self.spinner.stop()
        if self.quiet:
            return
        text = out if isinstance(out, str) else "[image]"
        lines = text.strip().splitlines() or [""]
        pre = "      " if sub else "  "
        if is_err:
            summary = lines[0][: self.width - 10]
            style = "red"
        else:
            if name in ("read_file",):
                summary = f"{len(lines)} lines"
            elif name == "bash":
                summary = lines[-1][: self.width - 10]
                if not self.verbose and len(lines) > 1:
                    tail = [l for l in lines[:-1] if l.strip()][-3:]
                    for l in tail:
                        self.console.print(Text(pre + "│ " + l[: self.width - 8], style="dim"))
            elif name == "task":
                summary = f"report: {len(text)} chars"
            else:
                summary = lines[0][: self.width - 10] + (f" (+{len(lines) - 1} lines)" if len(lines) > 1 else "")
            style = "dim"
        with self._lock:
            self.console.print(Text(f"{pre}⎿ {summary}" + (f"  [{dur:.1f}s]" if dur > 2 else ""), style=style))

    def show_todos(self, todos: list[dict[str, Any]]) -> None:
        if self.quiet:
            return
        with self._lock:
            self.end_stream()
            for t in todos:
                mark = {"completed": "[green]✔[/]", "in_progress": "[yellow]▶[/]"}.get(t["status"], "[dim]☐[/]")
                style = "dim strike" if t["status"] == "completed" else ("bold" if t["status"] == "in_progress" else "")
                self.console.print(f"    {mark} [{style}]{_esc(t['content'])}[/]" if style else
                                   f"    {mark} {_esc(t['content'])}")

    # ---------------------------------------------------------------- misc output
    def info(self, msg: str) -> None:
        with self._lock:
            self.spinner.stop()
            self.end_stream()
            self.console.print(Text("  ℹ " + msg, style="dim yellow"))

    def error(self, msg: str) -> None:
        with self._lock:
            self.spinner.stop()
            self.end_stream()
            self.console.print(Text("✖ " + redact(msg), style="bold red"))

    def markdown(self, md: str) -> None:
        self.console.print(Markdown(md))

    def panel(self, body: str, title: str = "") -> None:
        self.console.print(Panel(body, title=title, border_style="cyan", expand=False))

    # ---------------------------------------------------------------- interaction
    def ask(self, question: str) -> str:
        with self._lock:
            self.spinner.stop()
            self.end_stream()
            self.console.print(Panel(question, title="❓ agent asks", border_style="yellow"))
        try:
            return input("answer> ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def confirm(self, msg: str) -> bool:
        with self._lock:
            self.spinner.stop()
            self.end_stream()
        try:
            a = input(f"  {msg} [y/N/a(all)] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if a in ("a", "all"):
            self.approve_all = True
            return True
        return a in ("y", "yes", "д", "да")

    def permit(self, ctx, tool, args: dict[str, Any]) -> bool:
        if not ctx.interactive and not ctx.is_subagent:
            return True
        mode = self.cfg.get("permission_mode", "auto")
        if tool.name == "bash" and self.cfg.get("confirm_dangerous", True) and is_dangerous(args.get("command", "")):
            return self.confirm(f"⚠ dangerous command: {_short(args.get('command', ''), 200)} — run?")
        if mode == "ask" and (tool.writes or tool.shell) and not self.approve_all:
            what = args.get("command") or args.get("path") or args.get("action") or ""
            return self.confirm(f"allow {tool.name}({_short(what, 120)})?")
        return True


def _esc(s: str) -> str:
    return s.replace("[", "\\[")
