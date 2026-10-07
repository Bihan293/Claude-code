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

from . import models as model_catalog
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
        base = cfg["base_url"].rstrip("/")
        for suffix in ("/messages", "/chat/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        self.base = base
        self.url = self.base + "/messages"
        self.oa_url = self.base + "/chat/completions"
        self.total = Usage()
        self.reset_features()
        timeout = httpx.Timeout(
            connect=cfg.get("connect_timeout", 30),
            read=cfg.get("request_timeout", 600),
            write=60,
            pool=60,
        )
        self.http = httpx.Client(timeout=timeout, http2=False, follow_redirects=True)

    def reset_features(self) -> None:
        """(Re)initialise the optional-feature flags. Called on model switch, because a
        parameter rejected by one model may be supported by another."""
        cfg = self.cfg
        # feature flags that get disabled automatically if endpoint rejects them
        self.thinking_mode = cfg.get("thinking", "adaptive")
        self.allow_effort = bool(cfg.get("effort"))
        self.allow_cache = bool(cfg.get("prompt_caching", True))
        self.allow_stream = bool(cfg.get("stream", True))
        self.allow_cache_ttl = cfg.get("cache_ttl", "5m") == "1h"
        # OpenAI chat-completions flags
        self.oa_max_completion = True     # max_completion_tokens (new) vs max_tokens (legacy)
        self.oa_effort = bool(cfg.get("effort"))
        self.oa_stream_usage = True       # stream_options.include_usage

    def api_format(self, model: str | None = None) -> str:
        return model_catalog.api_for(model or self.cfg["model"], self.cfg.get("api_format", "auto"))

    # ------------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        h = {
            "content-type": "application/json",
            "anthropic-version": self.cfg.get("anthropic_version", "2023-06-01"),
            "accept": "application/json",
            "user-agent": "opus-agent/1.1 (termux)",
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
            # Cache is prefix-based (tools -> system -> messages), max 4 breakpoints:
            # 1: static system block (covers tools + static prompt, shared by every session),
            # 2: dynamic system block (env/memory), 3-4: tail of the conversation.
            if sys_blocks:
                sys_blocks[0]["cache_control"] = dict(cc)
                if len(sys_blocks) > 1:
                    sys_blocks[-1]["cache_control"] = dict(cc)
            elif tools:
                tools[-1]["cache_control"] = dict(cc)
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
            openai = self.api_format(model) == "openai"
            if openai:
                payload = self.build_openai_payload(system, messages, tools, model, max_tokens, stream)
            else:
                payload = self.build_payload(system, messages, tools, model, max_tokens, thinking, stream)
            try:
                if openai:
                    resp = self._oa_stream(payload, cb) if stream else self._oa_plain(payload)
                else:
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
                    if (self._oa_downgrade(e) if openai else self._downgrade(e)):
                        continue
                    low = (e.body or str(e)).lower()
                    if "prompt is too long" in low or "context" in low and "length" in low \
                            or "too many tokens" in low or "maximum context" in low \
                            or "context_length_exceeded" in low:
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

    # ================================================================== OpenAI format
    # GPT models on Tooken Club use /v1/chat/completions. Internally the agent keeps
    # Anthropic-shaped messages; we convert on the way out and back.
    def build_openai_payload(self, system: str | list, messages: list, tools: list | None,
                             model: str | None, max_tokens: int | None, stream: bool) -> dict[str, Any]:
        model = model or self.cfg["model"]
        max_tokens = max_tokens or self.cfg.get("max_tokens", 32000)
        sys_text = system if isinstance(system, str) else "\n\n".join(
            b.get("text", "") for b in system if b.get("type") == "text")
        out: list[dict[str, Any]] = [{"role": "system", "content": sys_text}] if sys_text else []
        out += to_openai_messages(messages)
        payload: dict[str, Any] = {"model": model, "messages": out}
        payload["max_completion_tokens" if self.oa_max_completion else "max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}} for t in tools]
        if stream:
            payload["stream"] = True
            if self.oa_stream_usage:
                payload["stream_options"] = {"include_usage": True}
        eff = (self.cfg.get("effort") or "").strip()
        if self.oa_effort and eff:
            payload["reasoning_effort"] = {"max": "high"}.get(eff, eff)
        return payload

    def _oa_downgrade(self, e: LLMError) -> bool:
        body = (e.body or str(e)).lower()
        if self.oa_max_completion and "max_completion_tokens" in body:
            self.oa_max_completion = False
            return True
        if self.oa_effort and ("reasoning_effort" in body or "reasoning" in body):
            self.oa_effort = False
            return True
        if self.oa_stream_usage and "stream_options" in body:
            self.oa_stream_usage = False
            return True
        if self.allow_stream and "stream" in body:
            self.allow_stream = False
            return True
        return False

    def _oa_plain(self, payload: dict[str, Any]) -> Response:
        payload = dict(payload)
        payload.pop("stream", None)
        payload.pop("stream_options", None)
        r = self.http.post(self.oa_url, headers=self.headers(), json=payload)
        body = r.text
        if r.status_code >= 400:
            self._raise_for(r, body)
        return _oa_response(json.loads(body))

    def _oa_stream(self, payload: dict[str, Any], cb: Callbacks) -> Response:
        text = ""
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish = None
        model = ""
        got_any = False
        with self.http.stream("POST", self.oa_url, headers=self.headers(), json=payload) as r:
            if r.status_code >= 400:
                self._raise_for(r, r.read().decode("utf-8", "replace"))
            ctype = r.headers.get("content-type", "")
            if "text/event-stream" not in ctype and "stream" not in ctype:
                self.allow_stream = False
                resp = _oa_response(json.loads(r.read().decode("utf-8", "replace")))
                if resp.text:
                    cb.on_text(resp.text)
                return resp
            for line in r.iter_lines():
                if cb.should_stop():
                    raise Interrupted()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                got_any = True
                if ev.get("error"):
                    err = ev["error"]
                    raise LLMError(f"stream error: {err}", 529 if "overload" in str(err).lower() else 500,
                                   json.dumps(err))
                model = ev.get("model") or model
                if ev.get("usage"):
                    usage = _oa_usage(ev["usage"])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content") or d.get("reasoning"):
                        cb.on_thinking(str(d.get("reasoning_content") or d.get("reasoning")))
                    if d.get("content"):
                        text += d["content"]
                        cb.on_text(d["content"])
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", 0)
                        c = calls.setdefault(i, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            c["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            if not c["name"]:
                                cb.on_tool_start(fn["name"])
                            c["name"] += fn["name"]
                        if fn.get("arguments"):
                            c["args"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        if not got_any:
            raise LLMError("empty stream", 502)
        if finish is None and not text and not calls:
            raise LLMError("stream ended prematurely", 502)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        for i in sorted(calls):
            content.append(_oa_tool_block(calls[i]["id"], calls[i]["name"], calls[i]["args"]))
        for b in content:
            cb.on_block_end(b)
        return Response(content, _oa_stop(finish, bool(calls)), usage, model)


def _oa_stop(finish: str | None, has_calls: bool) -> str:
    if has_calls or finish in ("tool_calls", "function_call"):
        return "tool_use"
    return {"length": "max_tokens", "content_filter": "refusal"}.get(finish or "", "end_turn")


def _oa_usage(u: dict[str, Any]) -> dict[str, int]:
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {"input_tokens": max(0, prompt - cached), "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0, "output_tokens": int(u.get("completion_tokens") or 0)}


def _oa_tool_block(cid: str, name: str, args: str) -> dict[str, Any]:
    try:
        inp = json.loads(args) if args.strip() else {}
        if not isinstance(inp, dict):
            inp = {"value": inp}
    except ValueError:
        inp = {"__invalid_json__": args[:2000]}
    return {"type": "tool_use", "id": cid or f"call_{random.getrandbits(48):x}", "name": name, "input": inp}


def _oa_response(j: dict[str, Any]) -> Response:
    if j.get("error"):
        err = j["error"]
        raise LLMError(str(err), 529 if "overload" in str(err).lower() else 400, json.dumps(err))
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content: list[dict[str, Any]] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    calls = msg.get("tool_calls") or []
    for tc in calls:
        fn = tc.get("function") or {}
        content.append(_oa_tool_block(tc.get("id", ""), fn.get("name", ""), fn.get("arguments") or ""))
    return Response(content, _oa_stop(ch.get("finish_reason"), bool(calls)), _oa_usage(j.get("usage") or {}),
                    j.get("model", ""))


def _oa_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    return "\n".join(x.get("text", "") for x in c or [] if x.get("type") == "text")


def _oa_image(b: dict[str, Any]) -> dict[str, Any] | None:
    src = b.get("source") or {}
    if src.get("type") == "base64":
        return {"type": "image_url", "image_url": {"url": f"data:{src.get('media_type')};base64,{src.get('data')}"}}
    if src.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url")}}
    return None


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped history -> OpenAI chat messages. Thinking blocks are dropped
    (they are model-specific), tool_result blocks become role=tool messages."""
    out: list[dict[str, Any]] = []
    for m in messages:
        c = m.get("content")
        blocks = [{"type": "text", "text": c}] if isinstance(c, str) else list(c or [])
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b.get("name", ""),
                                   "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            if text or calls:
                out.append(msg)
            continue
        parts: list[dict[str, Any]] = []
        for b in blocks:
            t = b.get("type")
            if t == "tool_result":
                rc = b.get("content")
                body = _oa_text(rc) or "(no output)"
                if b.get("is_error"):
                    body = "ERROR: " + body
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": body})
                if isinstance(rc, list):  # images returned by read_file
                    parts += [p for p in (_oa_image(x) for x in rc if x.get("type") == "image") if p]
            elif t == "text" and b.get("text"):
                parts.append({"type": "text", "text": b["text"]})
            elif t == "image":
                p = _oa_image(b)
                if p:
                    parts.append(p)
        if parts:
            if all(p["type"] == "text" for p in parts):
                out.append({"role": "user", "content": "\n\n".join(p["text"] for p in parts)})
            else:
                out.append({"role": "user", "content": parts})
    return out


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
