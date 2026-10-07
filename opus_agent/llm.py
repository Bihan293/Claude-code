"""Anthropic Messages API client (tooken.club compatible).

Features: SSE streaming with live callbacks, non-stream fallback, exponential
backoff with Retry-After, prompt caching breakpoints, extended/adaptive
thinking with automatic downgrade if the endpoint rejects a parameter,
usage accounting.
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .security import log, redact

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


class LLMError(Exception):
    def __init__(self, msg: str, status: int | None = None, body: str = "") -> None:
        super().__init__(redact(msg))
        self.status = status
        self.body = redact(body)


class ContextTooLong(LLMError):
    pass


class Interrupted(Exception):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    requests: int = 0

    def add(self, u: dict[str, Any] | "Usage") -> None:
        d = u.__dict__ if isinstance(u, Usage) else (u or {})
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                  "cache_read_input_tokens", "requests"):
            v = d.get(k) or 0
            if isinstance(v, (int, float)):
                setattr(self, k, getattr(self, k) + int(v))

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)

    def cost(self, cfg) -> float:
        return (
            self.input_tokens * cfg.get("price_input", 5.0)
            + self.output_tokens * cfg.get("price_output", 25.0)
            + self.cache_creation_input_tokens * cfg.get("price_cache_write", 6.25)
            + self.cache_read_input_tokens * cfg.get("price_cache_read", 0.5)
        ) / 1_000_000

    @property
    def prompt_total(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


@dataclass
class Response:
    content: list[dict[str, Any]]
    stop_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class Callbacks:
    """Streaming hooks. Override what you need."""

    def on_text(self, delta: str) -> None: ...
    def on_thinking(self, delta: str) -> None: ...
    def on_tool_start(self, name: str) -> None: ...
    def on_block_end(self, block: dict[str, Any]) -> None: ...
    def on_retry(self, attempt: int, wait: float, reason: str) -> None: ...
    def should_stop(self) -> bool:
        return False


class LLMClient:
    def __init__(self, cfg, api_key: str) -> None:
        self.cfg = cfg
        self.api_key = api_key
        self.base = cfg["base_url"].rstrip("/")
        self.url = self.base + "/messages" if not self.base.endswith("/messages") else self.base
        self.total = Usage()
        # feature flags that get disabled automatically if endpoint rejects them
        self.thinking_mode = cfg.get("thinking", "adaptive")
        self.allow_effort = bool(cfg.get("effort"))
        self.allow_cache = bool(cfg.get("prompt_caching", True))
        self.allow_stream = bool(cfg.get("stream", True))
        self.allow_cache_ttl = cfg.get("cache_ttl", "5m") == "1h"
        timeout = httpx.Timeout(
            connect=cfg.get("connect_timeout", 30),
            read=cfg.get("request_timeout", 600),
            write=60,
            pool=60,
        )
        self.http = httpx.Client(timeout=timeout, http2=False, follow_redirects=True)

    # ------------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        h = {
            "content-type": "application/json",
            "anthropic-version": self.cfg.get("anthropic_version", "2023-06-01"),
            "accept": "application/json",
            "user-agent": "opus-agent/1.0 (termux)",
        }
        style = self.cfg.get("auth_style", "both")
        if style in ("x-api-key", "both"):
            h["x-api-key"] = self.api_key
        if style in ("bearer", "both"):
            h["authorization"] = f"Bearer {self.api_key}"
        betas = [b.strip() for b in (self.cfg.get("anthropic_beta") or "").split(",") if b.strip()]
        if self.allow_cache_ttl and self.allow_cache:
            betas.append("extended-cache-ttl-2025-04-11")
        if betas:
            h["anthropic-beta"] = ",".join(dict.fromkeys(betas))
        return h

    def build_payload(self, system: str | list, messages: list, tools: list | None,
                      model: str | None, max_tokens: int | None, thinking: bool,
                      stream: bool) -> dict[str, Any]:
        model = model or self.cfg["model"]
        max_tokens = max_tokens or self.cfg.get("max_tokens", 32000)
        msgs = copy.deepcopy(messages)
        sys_blocks = [{"type": "text", "text": system}] if isinstance(system, str) else copy.deepcopy(system)
        tools = copy.deepcopy(tools) if tools else None
        if self.allow_cache:
            cc: dict[str, Any] = {"type": "ephemeral"}
            if self.allow_cache_ttl:
                cc["ttl"] = "1h"
            # breakpoint 1: tools (static), 2: system, 3-4: tail of conversation
            if tools:
                tools[-1]["cache_control"] = dict(cc)
            if sys_blocks:
                sys_blocks[-1]["cache_control"] = dict(cc)
            _mark_tail_cache(msgs, dict(cc), count=2)
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": sys_blocks,
            "messages": msgs,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        if thinking and self.thinking_mode != "off":
            if self.thinking_mode == "adaptive":
                payload["thinking"] = {"type": "adaptive"}
            else:
                budget = min(int(self.cfg.get("thinking_budget", 12000)), max_tokens - 2000)
                if budget >= 1024:
                    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            # without thinking, thinking blocks in history are allowed, keep as is
            pass
        if self.allow_effort and self.cfg.get("effort"):
            payload["output_config"] = {"effort": self.cfg["effort"]}
        return payload

    # ------------------------------------------------------------------
    def create(self, system, messages, tools=None, *, model=None, max_tokens=None,
               thinking: bool = True, cb: Callbacks | None = None) -> Response:
        cb = cb or Callbacks()
        max_retries = int(self.cfg.get("max_retries", 8))
        attempt = 0
        while True:
            if cb.should_stop():
                raise Interrupted()
            stream = self.allow_stream
            payload = self.build_payload(system, messages, tools, model, max_tokens, thinking, stream)
            try:
                resp = self._stream(payload, cb) if stream else self._plain(payload)
                self.total.add(resp.usage)
                self.total.requests += 1
                log.info("llm ok model=%s stop=%s usage=%s", payload["model"], resp.stop_reason,
                         json.dumps(resp.usage))
                return resp
            except Interrupted:
                raise
            except LLMError as e:
                if e.status == 400 or e.status == 422:
                    if self._downgrade(e):
                        continue
                    low = (e.body or str(e)).lower()
                    if "prompt is too long" in low or "context" in low and "length" in low \
                            or "too many tokens" in low or "maximum context" in low:
                        raise ContextTooLong(str(e), e.status, e.body)
                    raise
                if e.status in (401, 403):
                    raise
                if e.status == 413:
                    raise ContextTooLong(str(e), e.status, e.body)
                if e.status is not None and e.status not in RETRY_STATUS:
                    raise
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = getattr(e, "retry_after", None) or min(90.0, (2 ** attempt) + random.random() * 2)
                log.warning("llm retry %s in %.1fs: %s", attempt, wait, e)
                cb.on_retry(attempt, wait, str(e)[:200])
                _sleep_interruptible(wait, cb)
            except (httpx.TransportError, httpx.TimeoutException, json.JSONDecodeError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise LLMError(f"network error: {e}")
                wait = min(90.0, (2 ** attempt) + random.random() * 2)
                log.warning("llm network retry %s in %.1fs: %r", attempt, wait, e)
                cb.on_retry(attempt, wait, f"network: {type(e).__name__}")
                _sleep_interruptible(wait, cb)

    def _downgrade(self, e: LLMError) -> bool:
        """Turn off an optional feature that the endpoint rejected. True if retry makes sense."""
        body = (e.body or str(e)).lower()
        if self.allow_effort and ("effort" in body or "output_config" in body):
            log.info("disabling effort param")
            self.allow_effort = False
            return True
        if self.allow_cache_ttl and ("ttl" in body or "beta" in body):
            self.allow_cache_ttl = False
            return True
        if "thinking" in body or "budget_tokens" in body or "adaptive" in body:
            if self.thinking_mode == "adaptive":
                log.info("adaptive thinking rejected -> enabled")
                self.thinking_mode = "enabled"
                return True
            if self.thinking_mode == "enabled":
                log.info("thinking rejected -> off")
                self.thinking_mode = "off"
                return True
        if self.allow_cache and "cache_control" in body:
            self.allow_cache = False
            return True
        if self.allow_stream and "stream" in body:
            self.allow_stream = False
            return True
        return False

    # ------------------------------------------------------------------
    def _raise_for(self, r: httpx.Response, body: str) -> None:
        msg = body
        try:
            j = json.loads(body)
            err = j.get("error") or {}
            if isinstance(err, dict):
                msg = f"{err.get('type', '')}: {err.get('message', '')}"
            elif isinstance(err, str):
                msg = err
        except ValueError:
            pass
        e = LLMError(f"HTTP {r.status_code}: {msg[:800]}", r.status_code, body[:4000])
        ra = r.headers.get("retry-after")
        if ra:
            try:
                e.retry_after = min(float(ra), 120.0)  # type: ignore[attr-defined]
            except ValueError:
                pass
        raise e

    def _plain(self, payload: dict[str, Any]) -> Response:
        payload = dict(payload)
        payload.pop("stream", None)
        r = self.http.post(self.url, headers=self.headers(), json=payload)
        body = r.text
        if r.status_code >= 400:
            self._raise_for(r, body)
        j = json.loads(body)
        if j.get("type") == "error":
            err = j.get("error") or {}
            raise LLMError(str(err), 500 if "overloaded" in str(err) else 400, body)
        return Response(j.get("content") or [], j.get("stop_reason"), j.get("usage") or {}, j.get("model", ""))

    def _stream(self, payload: dict[str, Any], cb: Callbacks) -> Response:
        blocks: dict[int, dict[str, Any]] = {}
        partial_json: dict[int, str] = {}
        usage: dict[str, Any] = {}
        stop_reason = None
        model = ""
        got_any = False
        with self.http.stream("POST", self.url, headers=self.headers(), json=payload) as r:
            if r.status_code >= 400:
                body = r.read().decode("utf-8", "replace")
                self._raise_for(r, body)
            ctype = r.headers.get("content-type", "")
            if "text/event-stream" not in ctype and "stream" not in ctype:
                # proxy ignored stream=true and returned plain JSON
                body = r.read().decode("utf-8", "replace")
                j = json.loads(body)
                if j.get("type") == "error":
                    raise LLMError(str(j.get("error")), 500, body)
                self.allow_stream = False
                content = j.get("content") or []
                for b in content:
                    if b.get("type") == "text":
                        cb.on_text(b.get("text", ""))
                return Response(content, j.get("stop_reason"), j.get("usage") or {}, j.get("model", ""))
            event = None
            data_lines: list[str] = []
            for line in r.iter_lines():
                if cb.should_stop():
                    raise Interrupted()
                if line == "":
                    if data_lines:
                        data = "\n".join(data_lines)
                        data_lines = []
                        if data.strip() == "[DONE]":
                            break
                        ev = json.loads(data)
                        et = ev.get("type") or event
                        got_any = True
                        if et == "message_start":
                            m = ev.get("message") or {}
                            model = m.get("model", "")
                            usage.update(m.get("usage") or {})
                        elif et == "content_block_start":
                            i = ev["index"]
                            b = dict(ev.get("content_block") or {})
                            if b.get("type") == "tool_use":
                                b["input"] = {}
                                partial_json[i] = ""
                                cb.on_tool_start(b.get("name", ""))
                            if b.get("type") == "text":
                                b["text"] = b.get("text", "")
                                if b["text"]:
                                    cb.on_text(b["text"])
                            if b.get("type") == "thinking":
                                b.setdefault("thinking", "")
                                b.setdefault("signature", "")
                            blocks[i] = b
                        elif et == "content_block_delta":
                            i = ev["index"]
                            d = ev.get("delta") or {}
                            b = blocks.setdefault(i, {"type": "text", "text": ""})
                            dt = d.get("type")
                            if dt == "text_delta":
                                b["text"] = b.get("text", "") + d.get("text", "")
                                cb.on_text(d.get("text", ""))
                            elif dt == "input_json_delta":
                                partial_json[i] = partial_json.get(i, "") + d.get("partial_json", "")
                            elif dt == "thinking_delta":
                                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
                                cb.on_thinking(d.get("thinking", ""))
                            elif dt == "signature_delta":
                                b["signature"] = b.get("signature", "") + d.get("signature", "")
                            elif dt == "citations_delta":
                                b.setdefault("citations", []).append(d.get("citation"))
                        elif et == "content_block_stop":
                            i = ev["index"]
                            b = blocks.get(i)
                            if b is not None and b.get("type") in ("tool_use", "server_tool_use"):
                                raw = partial_json.get(i, "")
                                try:
                                    b["input"] = json.loads(raw) if raw.strip() else {}
                                except ValueError:
                                    b["input"] = {"__invalid_json__": raw[:2000]}
                            if b is not None:
                                cb.on_block_end(b)
                        elif et == "message_delta":
                            d = ev.get("delta") or {}
                            if d.get("stop_reason"):
                                stop_reason = d["stop_reason"]
                            for k, v in (ev.get("usage") or {}).items():
                                if v is not None:
                                    usage[k] = v
                        elif et == "message_stop":
                            break
                        elif et == "error":
                            err = ev.get("error") or {}
                            st = 529 if "overloaded" in str(err) else 500
                            raise LLMError(f"stream error: {err}", st, json.dumps(err))
                    event = None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        if not got_any:
            raise LLMError("empty stream", 502)
        if stop_reason is None and not blocks:
            raise LLMError("stream ended prematurely", 502)
        content = [blocks[i] for i in sorted(blocks)]
        # drop empty text blocks (API rejects them on the next turn)
        content = [b for b in content if not (b.get("type") == "text" and not b.get("text"))]
        return Response(content, stop_reason or "end_turn", usage, model)


def _mark_tail_cache(msgs: list[dict[str, Any]], cc: dict[str, Any], count: int = 2) -> None:
    """Put cache_control on the last block of the last `count` user messages."""
    marked = 0
    for m in reversed(msgs):
        if marked >= count:
            break
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            if not c:
                continue
            m["content"] = [{"type": "text", "text": c, "cache_control": dict(cc)}]
            marked += 1
        elif isinstance(c, list) and c:
            for b in reversed(c):
                if b.get("type") in ("text", "tool_result", "image", "document"):
                    b["cache_control"] = dict(cc)
                    marked += 1
                    break


def _sleep_interruptible(seconds: float, cb: Callbacks) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if cb.should_stop():
            raise Interrupted()
        time.sleep(min(0.5, end - time.time()) if end > time.time() else 0)
