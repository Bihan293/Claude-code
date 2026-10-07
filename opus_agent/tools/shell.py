"""Shell execution: foreground with timeout, background jobs, output polling."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from ..security import log, redact
from .base import ToolContext, ToolError, tool

DANGEROUS = [
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\$HOME)(\s|$)",
    r"\bmkfs\b", r"\bdd\s+if=.*of=/dev/", r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\bgit\s+push\b.*(--force|-f)\b.*\b(main|master)\b",
    r"\bgit\s+push\b.*\b(main|master)\b.*(--force|-f)\b",
    r"\bchmod\s+-R\s+777\s+/", r"\bshutdown\b|\breboot\b",
]
PUSH_DEFAULT = re.compile(r"\bgit\s+push\b[^|;&]*\b(origin\s+)?(HEAD:)?(main|master)\b")


class Job:
    def __init__(self, cmd: str, cwd: Path, env: dict[str, str], log_path: Path) -> None:
        self.id = uuid.uuid4().hex[:6]
        self.cmd = cmd
        self.log_path = log_path
        self.started = time.time()
        self._fh = log_path.open("wb")
        self.proc = subprocess.Popen(
            _shell_argv(cmd),
            cwd=str(cwd), env=env, stdout=self._fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        self.read_pos = 0

    def poll(self) -> int | None:
        rc = self.proc.poll()
        if rc is not None and not self._fh.closed:
            self._fh.close()
        return rc

    def new_output(self) -> str:
        try:
            with self.log_path.open("rb") as f:
                f.seek(self.read_pos)
                data = f.read()
                self.read_pos += len(data)
            return data.decode("utf-8", "replace")
        except OSError:
            return ""

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            time.sleep(0.5)
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.poll()


JOBS: dict[str, Job] = {}
_bash: bool | None = None


def _has_bash() -> bool:
    global _bash
    if _bash is None:
        from shutil import which
        _bash = which("bash") is not None
    return _bash


def _shell_argv(cmd: str) -> list[str]:
    # non-login shell for speed; login profile can be slow on Termux
    return ["bash", "-c", cmd] if _has_bash() else ["sh", "-c", cmd]


def is_dangerous(cmd: str) -> bool:
    return any(re.search(p, cmd) for p in DANGEROUS)


def kill_all_jobs() -> None:
    for j in list(JOBS.values()):
        if j.poll() is None:
            j.kill()


class _CwdTracker:
    MARK = "__OPUS_CWD__"


@tool("bash",
      """Execute a shell command (bash) in the project directory and return stdout+stderr and exit code.
The working directory persists between calls (a trailing `cd` is remembered). Environment:
Android Termux (pkg install <x> for system packages, pip/npm for libraries; no sudo/root).
Use run_in_background=true for servers/watchers/long builds, then poll with job_output.
Avoid interactive commands (use -y flags, non-interactive options). Chain related commands with &&.
Do NOT use bash for reading/searching/editing files when read_file/grep/glob/edit_file fit better.""",
      {"command": {"type": "string"},
       "timeout": {"type": "integer", "description": "Seconds (default 600, max 3600)"},
       "run_in_background": {"type": "boolean"},
       "description": {"type": "string", "description": "5-10 word summary of what it does"}},
      ["command"], shell=True, writes=True, readonly_ok=False)
def bash(ctx: ToolContext, command: str, timeout: int | None = None, run_in_background: bool = False,
         description: str = "") -> str:
    if not command.strip():
        raise ToolError("empty command")
    if not ctx.cfg.get("allow_push_to_default_branch", False) and PUSH_DEFAULT.search(command):
        raise ToolError("Pushing directly to main/master is disabled. Create a feature branch "
                        "(git switch -c <branch>), push it and open a PR. (Config: allow_push_to_default_branch)")
    env = ctx.env()
    if run_in_background:
        logs = ctx.cfg.home / "logs" / "jobs"
        logs.mkdir(parents=True, exist_ok=True)
        job = Job(command, ctx.cwd, env, logs / f"{uuid.uuid4().hex[:8]}.log")
        JOBS[job.id] = job
        time.sleep(1.5)
        rc = job.poll()
        first = job.new_output()
        state = "running" if rc is None else f"exited with {rc}"
        return f"Started background job {job.id} ({state}). Initial output:\n{first[-4000:]}"
    t = min(int(timeout or ctx.cfg.get("bash_timeout", 600)), 3600)
    marker = _CwdTracker.MARK
    wrapped = f"{command}\n__rc=$?; echo; echo \"{marker}$(pwd)\"; exit $__rc"
    start = time.time()
    proc = subprocess.Popen(_shell_argv(wrapped), cwd=str(ctx.cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    chunks: list[bytes] = []

    def reader() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            chunks.append(line)
            if ctx.ui is not None:
                ctx.ui.shell_line(line.decode("utf-8", "replace"))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    timed_out = interrupted = False
    while proc.poll() is None:
        if time.time() - start > t:
            timed_out = True
            break
        if ctx.stop_event.is_set():
            interrupted = True
            break
        time.sleep(0.1)
    if timed_out or interrupted:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(1)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    th.join(timeout=5)
    rc = proc.wait(timeout=10) if proc.poll() is None else proc.returncode
    out = b"".join(chunks).decode("utf-8", "replace")
    # extract and strip cwd marker
    m = None
    for m in re.finditer(re.escape(marker) + r"(.*)", out):
        pass
    if m:
        newcwd = Path(m.group(1).strip())
        if newcwd.is_dir():
            ctx.cwd = newcwd
            if ctx.session is not None:
                ctx.session.cwd = str(newcwd)
        out = out[:m.start()].rstrip("\n")
    out = _clean(out)
    dur = time.time() - start
    log.info("bash rc=%s %.1fs: %s", rc, dur, redact(command[:300]))
    status = f"exit code {rc}"
    if timed_out:
        status = f"TIMED OUT after {t}s (killed). Consider run_in_background=true"
    if interrupted:
        status = "INTERRUPTED by user"
    limit = int(ctx.cfg.get("tool_output_limit", 30000))
    if len(out) > limit:
        # keep the tail (errors are usually at the end)
        out = out[: limit // 4] + f"\n... [{len(out) - limit} chars omitted] ...\n" + out[-(limit * 3 // 4):]
    return f"{out}\n[{status}; {dur:.1f}s; cwd={ctx.cwd}]" if out else f"[{status}; no output; {dur:.1f}s; cwd={ctx.cwd}]"


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r(?!\n)")


def _clean(s: str) -> str:
    s = _ANSI.sub("", s)
    # collapse progress-bar spam: keep only last of repeated similar lines
    lines = s.split("\n")
    if len(lines) > 400:
        out = []
        prev_key = None
        for l in lines:
            key = re.sub(r"\d+", "#", l)[:60]
            if key == prev_key and out:
                out[-1] = l
            else:
                out.append(l)
            prev_key = key
        lines = out
    return "\n".join(lines)


@tool("job_output", "Get new output and status of a background job started with bash(run_in_background).",
      {"job_id": {"type": "string"}, "wait": {"type": "integer", "description": "Seconds to wait for more output (max 300)"}},
      ["job_id"])
def job_output(ctx: ToolContext, job_id: str, wait: int = 0) -> str:
    job = JOBS.get(job_id)
    if not job:
        raise ToolError(f"No job {job_id}. Jobs: {', '.join(JOBS) or 'none'}")
    end = time.time() + min(int(wait or 0), 300)
    while time.time() < end and job.poll() is None and not ctx.stop_event.is_set():
        time.sleep(0.5)
    rc = job.poll()
    out = _clean(job.new_output())
    state = "running" if rc is None else f"exited with code {rc}"
    return f"[job {job_id}: {state}; {time.time() - job.started:.0f}s]\n{out[-20000:] if out else '(no new output)'}"


@tool("job_kill", "Kill a background job.", {"job_id": {"type": "string"}}, ["job_id"], shell=True)
def job_kill(ctx: ToolContext, job_id: str) -> str:
    job = JOBS.get(job_id)
    if not job:
        raise ToolError(f"No job {job_id}")
    job.kill()
    return f"Killed job {job_id}"


@tool("job_list", "List background jobs.", {})
def job_list(ctx: ToolContext) -> str:
    if not JOBS:
        return "No background jobs."
    return "\n".join(f"{j.id}: {'running' if j.poll() is None else 'exit ' + str(j.poll())} — {j.cmd[:100]}"
                     for j in JOBS.values())
