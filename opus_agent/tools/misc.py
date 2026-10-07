"""Planning (todo), memory, web, user interaction and sub-agent tools."""
from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .base import ToolContext, ToolError, tool


# ---------------------------------------------------------------- todo / plan
@tool("todo_write",
      """Create/replace the task plan (persisted with the session, survives restarts).
Use it for any task with 3+ steps: write the full list at the start, then update statuses as you go
(exactly one item in_progress). Items: {content, status: pending|in_progress|completed}.""",
      {"todos": {"type": "array", "items": {"type": "object", "properties": {
          "content": {"type": "string"},
          "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
          "required": ["content", "status"]}}},
      ["todos"])
def todo_write(ctx: ToolContext, todos: list[dict[str, Any]]) -> str:
    clean = [{"content": str(t.get("content", ""))[:300], "status": t.get("status", "pending")} for t in todos]
    if ctx.session is not None:
        ctx.session.todos = clean
    if ctx.ui is not None:
        ctx.ui.show_todos(clean)
    done = sum(t["status"] == "completed" for t in clean)
    return f"Plan saved ({done}/{len(clean)} completed)."


# ---------------------------------------------------------------- memory
@tool("memory",
      """Persistent memory across sessions. scope='project' (this repo: architecture, build/test commands,
conventions, gotchas) or 'global' (user preferences, environment facts). Actions: view, add {text},
replace {old, new}, rewrite {text} (full replacement – use to consolidate). Save only durable, useful,
non-secret facts; never store tokens/keys.""",
      {"action": {"type": "string", "enum": ["view", "add", "replace", "rewrite"]},
       "scope": {"type": "string", "enum": ["project", "global"]},
       "text": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
      ["action"])
def memory_tool(ctx: ToolContext, action: str, scope: str = "project", text: str = "", old: str = "",
                new: str = "") -> str:
    m = ctx.memory
    if m is None:
        raise ToolError("memory unavailable")
    from ..security import redact
    if action == "view":
        return m.read(scope) or "(empty)"
    if action == "add":
        if not text.strip():
            raise ToolError("text required")
        return m.add(scope, redact(text))
    if action == "replace":
        return m.replace(scope, old, redact(new))
    if action == "rewrite":
        m.write(scope, redact(text))
        return "Memory rewritten."
    raise ToolError("bad action")


# ---------------------------------------------------------------- web
def _html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style|noscript|svg|head).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", s)
    s = re.sub(r"(?i)<li[^>]*>", "\n- ", s)
    s = re.sub(r"(?i)<h(\d)[^>]*>", lambda m: "\n" + "#" * int(m.group(1)) + " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


@tool("web_fetch",
      "Fetch a URL (docs, API references, raw files) and return readable text. Use start to page through long pages.",
      {"url": {"type": "string"}, "start": {"type": "integer", "description": "char offset"},
       "max_chars": {"type": "integer"}},
      ["url"])
def web_fetch(ctx: ToolContext, url: str, start: int = 0, max_chars: int = 20000) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        r = httpx.get(url, timeout=45, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (Linux; Android 14) opus-agent"})
    except httpx.HTTPError as e:
        raise ToolError(f"fetch failed: {e}")
    ct = r.headers.get("content-type", "")
    text = r.text
    if "html" in ct:
        text = _html_to_text(text)
    start = max(0, int(start or 0))
    chunk = text[start:start + int(max_chars or 20000)]
    more = len(text) - start - len(chunk)
    return f"[{r.status_code} {url}]\n{chunk}" + (f"\n... ({more} more chars; use start={start + len(chunk)})"
                                                    if more > 0 else "")


@tool("web_search", "Search the web (DuckDuckGo). Returns titles, URLs and snippets.",
      {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"])
def web_search(ctx: ToolContext, query: str, max_results: int = 8) -> str:
    try:
        r = httpx.post("https://html.duckduckgo.com/html/", data={"q": query}, timeout=30,
                       headers={"User-Agent": "Mozilla/5.0 (Linux; Android 14)"}, follow_redirects=True)
    except httpx.HTTPError as e:
        raise ToolError(f"search failed: {e}")
    results = []
    for m in re.finditer(r'(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
                         r'(?:class="result__snippet"[^>]*>(.*?)</a>)?', r.text):
        href = html.unescape(m.group(1))
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        title = _html_to_text(m.group(2))
        snip = _html_to_text(m.group(3) or "")
        results.append(f"- {title}\n  {href}\n  {snip}")
        if len(results) >= int(max_results or 8):
            break
    return "\n".join(results) or "No results (search may be blocked; try web_fetch on a known URL)."


# ---------------------------------------------------------------- user interaction
@tool("ask_user",
      """Ask the user a question and wait for the answer. Use ONLY when truly blocked (ambiguous
requirement with costly wrong guess, missing credentials). In autonomous runs prefer a reasonable
assumption and mention it in the final report.""",
      {"question": {"type": "string"}}, ["question"], subagent_ok=False)
def ask_user(ctx: ToolContext, question: str) -> str:
    if ctx.ui is None or not ctx.interactive:
        return "User is not available (autonomous mode). Make a reasonable assumption, document it, continue."
    from ..notify import notify
    notify(ctx.cfg, f"Question: {question[:300]}", kind="question")
    ans = ctx.ui.ask(question)
    return f"User answered: {ans}" if ans else "User gave no answer; proceed with best judgement."


# ---------------------------------------------------------------- sub-agent
@tool("task",
      """Launch a sub-agent with a fresh context to do a self-contained job and return a concise report.
Ideal for broad exploration ("find where X is implemented and how Y flows"), research, or reviewing
a diff – keeps the main context small and saves tokens. Sub-agents have the same tools (except task
and ask_user). mode='explore' = read-only; mode='general' = can edit/run. Give a complete, specific
prompt: the sub-agent does not see this conversation. Several task calls in one response run in parallel.""",
      {"description": {"type": "string", "description": "3-6 word label"},
       "prompt": {"type": "string"},
       "mode": {"type": "string", "enum": ["explore", "general"]}},
      ["prompt"], subagent_ok=False)
def task(ctx: ToolContext, prompt: str, description: str = "", mode: str = "explore") -> str:
    if ctx.agent_factory is None:
        raise ToolError("sub-agents unavailable")
    return ctx.agent_factory(ctx, prompt, description or "subtask", mode)
