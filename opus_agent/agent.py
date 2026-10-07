"""The agent loop: model call -> tool execution -> repeat, with persistence,
context management, interruption and sub-agents."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import context as ctxm
from .checkpoints import Checkpoints
from .llm import Callbacks, ContextTooLong, Interrupted, LLMClient, LLMError, Usage
from .memory import Memory
from .notify import notify, wake_lock
from .prompts import CORE, build_system
from .security import log, redact
from .session import Session
from .usage import record_usage
from .tools import REGISTRY, ToolContext, run_tool

CONTINUE_MSG = ("Your previous response was cut off by the output token limit. Continue exactly where you "
                "left off. If you were writing a large file, split it: write a smaller part and extend it "
                "with edit_file.")


class StreamCB(Callbacks):
    def __init__(self, agent: "Agent") -> None:
        self.agent = agent

    def on_text(self, delta: str) -> None:
        if self.agent.ui:
            self.agent.ui.stream_text(delta)

    def on_thinking(self, delta: str) -> None:
        if self.agent.ui:
            self.agent.ui.stream_thinking(delta)

    def on_tool_start(self, name: str) -> None:
        if self.agent.ui:
            self.agent.ui.tool_preparing(name)

    def on_retry(self, attempt: int, wait: float, reason: str) -> None:
        if self.agent.ui:
            self.agent.ui.info(f"API retry {attempt} in {wait:.0f}s ({reason})")

    def should_stop(self) -> bool:
        return self.agent.ctx.stop_event.is_set()


class QuietCB(Callbacks):
    def __init__(self, agent: "Agent") -> None:
        self.agent = agent

    def should_stop(self) -> bool:
        return self.agent.ctx.stop_event.is_set()


class Agent:
    def __init__(self, cfg, session: Session, ui=None, *, llm: LLMClient | None = None,
                 subagent: bool = False, mode: str = "general", parent_ctx: ToolContext | None = None,
                 interactive: bool = True) -> None:
        self.cfg = cfg
        self.session = session
        self.ui = ui
        self.subagent = subagent
        self.mode = mode
        self.llm = llm or LLMClient(cfg, cfg.secret("api_key"))
        cwd = Path(session.cwd or Path.cwd())
        if not cwd.is_dir():
            cwd = Path.cwd()
        session.cwd = str(cwd)
        self.memory = Memory(cfg.home, cwd)
        session.project_key = session.project_key or self.memory.key
        self.ctx = ToolContext(
            cfg=cfg, cwd=cwd, session=session, ui=ui,
            checkpoints=Checkpoints(cfg.home / "checkpoints", session.id),
            memory=self.memory, llm=self.llm, agent_factory=self._spawn_subagent,
            interactive=interactive and not subagent, is_subagent=subagent,
            stop_event=parent_ctx.stop_event if parent_ctx else threading.Event(),
        )
        if parent_ctx is not None:
            self.ctx.read_files = parent_ctx.read_files  # share freshness info
        self.session_usage = Usage()
        self.session_usage.add(session.usage or {})
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def quiet_cb(self) -> Callbacks:
        return QuietCB(self)

    def plan_mode(self) -> bool:
        return self.cfg.get("permission_mode") == "readonly" or self.mode == "explore"

    def tool_specs(self) -> list[dict[str, Any]]:
        # The same full list for main agent, sub-agents and every mode: tools are the first
        # part of the cached prefix, so an identical list lets sub-agents reuse the main
        # agent's cache. Disallowed tools are rejected at execution time (tool_allowed).
        return [t.spec() for t in REGISTRY.values()]  # stable order -> stable cache prefix

    def tool_allowed(self, name: str) -> str | None:
        t = REGISTRY.get(name)
        if t is None:
            return None
        if self.subagent and not t.subagent_ok:
            return f"Tool '{name}' is not available to sub-agents."
        if self.mode == "explore" and not t.readonly_ok:
            return f"Tool '{name}' is disabled in explore (read-only) mode. Use read-only tools."
        return None

    def refresh_system(self, force: bool = False) -> None:
        """Build the system prompt. Within a running conversation the dynamic block
        (git status, memory, date) is kept frozen: changing it would invalidate the prompt
        cache for the whole history and re-bill it at cache-write price on every turn.
        It is rebuilt for a new/compacted conversation, on mode change, or when forced."""
        key = f"{self.plan_mode()}|{self.subagent}|{self.mode}"
        meta = self.session.meta
        if (force or not self.session.system or len(self.session.messages) <= 1
                or meta.get("system_key") != key or self.session.system[0].get("text") != CORE):
            self.session.system = build_system(self.ctx.cwd, self.memory.render(), plan_mode=self.plan_mode(),
                                               subagent=self.subagent, explore=self.mode == "explore")
            meta["system_key"] = key

    def account(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self.session_usage.add(usage)
            self.session_usage.requests += 1
            self.session.usage = self.session_usage.to_dict()
            pt = sum(int(usage.get(k) or 0) for k in
                     ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            if pt:
                self.session.last_prompt_tokens = pt + int(usage.get("output_tokens") or 0)
        record_usage(self.cfg.home, usage, self.cfg)

    def save(self) -> None:
        try:
            self.session.save(self.cfg.home)
        except OSError as e:
            log.warning("session save failed: %s", e)

    def interrupt(self) -> None:
        self.ctx.stop_event.set()

    # ------------------------------------------------------------------ context management
    def manage_context(self, force: bool = False) -> None:
        window = int(self.cfg.get("context_window", 200000))
        msgs = self.session.messages
        base_n = int(self.session.meta.get("usage_msgs", 0))
        if self.session.last_prompt_tokens and 0 < base_n <= len(msgs):
            # exact count from the API for the prefix + estimate for what was appended since
            est = self.session.last_prompt_tokens + ctxm.estimate_tokens(msgs[base_n:])
        else:
            est = ctxm.estimate_tokens(msgs) + ctxm.estimate_tokens(self.session.system) + 4000
        meta = self.session.meta
        # hysteresis: after a prune, only prune again after significant growth, so the prompt
        # cache prefix is not invalidated on every single turn
        prune_at = max(window * float(self.cfg.get("prune_threshold", 0.55)),
                       meta.get("pruned_at", 0) + window * 0.08)
        if force or est > prune_at:
            saved = ctxm.prune(self.session.messages, int(self.cfg.get("keep_recent_tool_results", 8)))
            est2 = max(0, est - int(saved / 3.2))
            self.session.last_prompt_tokens = est2
            meta["usage_msgs"] = len(msgs)
            meta["pruned_at"] = est2
            if saved and self.ui:
                self.ui.info(f"context pruned (~{saved // 3200}k tokens freed)")
            if force or est2 > window * float(self.cfg.get("compact_threshold", 0.78)):
                if self.ui:
                    self.ui.info("compacting conversation…")
                ctxm.compact(self)
                meta["pruned_at"] = 0
                meta["usage_msgs"] = len(self.session.messages)
                self.save()

    # ------------------------------------------------------------------ main loop
    def run(self, prompt: str | list[dict[str, Any]] | None) -> str:
        """Run until the model ends its turn. prompt=None resumes an interrupted session."""
        s = self.session
        self.ctx.stop_event.clear()
        if not s.title and isinstance(prompt, str):
            s.title = prompt.strip().splitlines()[0][:80] if prompt.strip() else ""
        self.refresh_system()
        s.repair()
        if prompt is not None:
            s.turn += 1
            content = prompt if isinstance(prompt, list) else [{"type": "text", "text": prompt}]
            if s.messages and s.messages[-1]["role"] == "user":
                last = s.messages[-1]
                last["content"] = ctxm_blocks(last["content"]) + content
            else:
                s.messages.append({"role": "user", "content": content})
        elif not s.messages:
            return ""
        elif s.messages[-1]["role"] == "assistant":
            s.messages.append({"role": "user", "content": [{"type": "text", "text":
                               "The session was interrupted. Check the current state (git status, files) and "
                               "continue the task until it is complete."}]})
        s.status = "running"
        self.save()
        use_wakelock = self.cfg.get("wake_lock", True) and not self.subagent
        if use_wakelock:
            wake_lock(True)
        final_text = ""
        iterations = 0
        max_iter = int(self.cfg.get("max_iterations", 300)) if not self.subagent else 80
        overflow_retries = 0
        try:
            while True:
                iterations += 1
                if iterations > max_iter:
                    final_text = (final_text + f"\n\n[Stopped: reached max_iterations={max_iter}. "
                                  "Use /continue to keep going.]")
                    s.status = "interrupted"
                    break
                self.manage_context()
                try:
                    if self.ui and not self.subagent:
                        self.ui.thinking_start()
                    resp = self.llm.create(s.system, ctxm.clean_for_api(s.messages), self.tool_specs(),
                                           cb=StreamCB(self) if not self.subagent else QuietCB(self),
                                           thinking=True)
                except ContextTooLong:
                    overflow_retries += 1
                    if overflow_retries > 3:
                        raise
                    if self.ui:
                        self.ui.info("context limit hit – compacting")
                    self.manage_context(force=True)
                    continue
                finally:
                    if self.ui and not self.subagent:
                        self.ui.thinking_stop()
                overflow_retries = 0
                self.account(resp.usage)
                s.meta["usage_msgs"] = len(s.messages) + 1  # prefix incl. the assistant reply below
                content = resp.content or [{"type": "text", "text": "(empty response)"}]
                s.messages.append({"role": "assistant", "content": content})
                self.save()
                if resp.text.strip():
                    final_text = resp.text
                if self.ui and not self.subagent:
                    self.ui.end_stream()
                uses = resp.tool_uses
                if uses:
                    results = self.execute_tools(uses)
                    s.messages.append({"role": "user", "content": results})
                    self.save()
                    if self.ctx.stop_event.is_set():
                        s.status = "interrupted"
                        break
                    continue
                if resp.stop_reason == "max_tokens":
                    s.messages.append({"role": "user", "content": [{"type": "text", "text": CONTINUE_MSG}]})
                    continue
                if resp.stop_reason == "pause_turn":
                    s.messages.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
                    continue
                s.status = "done"
                break
        except Interrupted:
            s.status = "interrupted"
            final_text = final_text or "[interrupted]"
        except LLMError as e:
            s.status = "error"
            log.error("llm error: %s", e)
            self.save()
            if not self.subagent:
                notify(self.cfg, f"Task failed: {redact(str(e))[:300]}", kind="error", wait=True)
            raise
        finally:
            s.repair()
            self.save()
            if use_wakelock:
                wake_lock(False)
        if not self.subagent and s.status == "done":
            self._after_task(final_text)
        return final_text

    def _after_task(self, final_text: str) -> None:
        s = self.session
        try:
            first_user = next((m for m in s.messages if m["role"] == "user"), None)
            task = s.title or ""
            if first_user and not task:
                task = json.dumps(first_user["content"])[:150]
            summary = final_text.strip()[-1500:] if final_text else "(no summary)"
            self.memory.journal(task, summary)
        except OSError:
            pass
        prs = list(dict.fromkeys(self.ctx.pr_urls + s.meta.get("pr_urls", [])))
        msg = f"Task finished: {s.title[:120]}"
        if prs:
            msg += "\nPR: " + ", ".join(prs)
        notify(self.cfg, msg + "\n\n" + final_text.strip()[-1200:], kind="done", wait=True)

    # ------------------------------------------------------------------ tools
    def execute_tools(self, uses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}

        def one(u: dict[str, Any]) -> None:
            name, args, uid = u.get("name", ""), u.get("input") or {}, u["id"]
            if self.ctx.stop_event.is_set():
                results[uid] = {"type": "tool_result", "tool_use_id": uid, "is_error": True,
                                "content": "Cancelled by user interrupt."}
                return
            if self.ui:
                self.ui.tool_call(name, args, sub=self.subagent)
            t0 = time.time()
            denied = self.tool_allowed(name)
            if denied:
                out, is_err = denied, True
            else:
                out, is_err = run_tool(self.ctx, name, args if isinstance(args, dict) else {})
            if self.ui:
                self.ui.tool_result(name, out, is_err, time.time() - t0, sub=self.subagent)
            r: dict[str, Any] = {"type": "tool_result", "tool_use_id": uid, "content": out if out else "(no output)"}
            if is_err:
                r["is_error"] = True
            results[uid] = r

        def parallel_ok(u: dict[str, Any]) -> bool:
            t = REGISTRY.get(u.get("name", ""))
            return bool(t) and (not t.writes and not t.shell) and t.name != "ask_user"

        i = 0
        while i < len(uses):
            group = [uses[i]]
            if parallel_ok(uses[i]):
                while i + len(group) < len(uses) and parallel_ok(uses[i + len(group)]):
                    group.append(uses[i + len(group)])
            if len(group) > 1:
                with ThreadPoolExecutor(max_workers=min(6, len(group))) as ex:
                    list(ex.map(one, group))
            else:
                one(group[0])
            i += len(group)
        return [results[u["id"]] for u in uses]

    # ------------------------------------------------------------------ sub-agents
    def _spawn_subagent(self, ctx: ToolContext, prompt: str, description: str, mode: str) -> str:
        sub = Session(cwd=str(ctx.cwd), meta={"subagent": True, "parent": self.session.id},
                      title=f"[sub] {description}")
        small = self.cfg.get("small_model") or ""
        llm = self.llm
        if small and small != self.cfg.get("model"):
            llm = LLMClient(self.cfg, self.cfg.secret("api_key"))
            llm.cfg = _Override(self.cfg, model=small)
        agent = Agent(self.cfg, sub, self.ui, llm=llm, subagent=True,
                      mode="explore" if mode != "general" else "general", parent_ctx=ctx)
        if self.ui:
            self.ui.info(f"sub-agent started: {description}")
        try:
            text = agent.run(prompt)
        finally:
            # child usage is already in the global ledger; add to this session's totals only
            with self._lock:
                self.session_usage.add({k: v for k, v in agent.session_usage.to_dict().items()})
                self.session.usage = self.session_usage.to_dict()
            ctx.touched_files |= agent.ctx.touched_files
            ctx.pr_urls += agent.ctx.pr_urls
        return text or "(sub-agent returned no report)"


class _Override:
    def __init__(self, base, **over: Any) -> None:
        self._base, self._over = base, over

    def get(self, k: str, d: Any = None) -> Any:
        return self._over.get(k, self._base.get(k, d))

    def __getitem__(self, k: str) -> Any:
        return self._over[k] if k in self._over else self._base[k]

    def __getattr__(self, k: str) -> Any:
        return getattr(self._base, k)


def ctxm_blocks(c: Any) -> list[dict[str, Any]]:
    return [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
