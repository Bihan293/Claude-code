"""Context management – keeps long autonomous runs inside the context window
while spending as few tokens as possible.

Levels (cheapest first):
1. Tool output limits at the source (tools truncate, shell keeps the tail).
2. Pruning: old, large tool results are replaced by short stubs; thinking blocks of
   finished turns are dropped. Done in one batch when the threshold is crossed so the
   prompt cache is invalidated rarely.
3. Compaction: one model call summarises the whole history into a dense "state of the
   work" document; the conversation restarts from that summary + plan + touched files.
"""
from __future__ import annotations

import json
from typing import Any

from .security import log

STUB_MIN_CHARS = 1200


def estimate_tokens(obj: Any) -> int:
    if isinstance(obj, str):
        s = obj
    else:
        s = json.dumps(obj, ensure_ascii=False)
    # ~3.2 chars/token for code-ish text, images are counted separately
    return int(len(s) / 3.2) + 1


def _block_text_len(b: dict[str, Any]) -> int:
    c = b.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        n = 0
        for x in c:
            if x.get("type") == "text":
                n += len(x.get("text", ""))
            elif x.get("type") == "image":
                n += 6000
        return n
    return 0


def last_human_index(messages: list[dict[str, Any]]) -> int:
    """Index of the last user message that is a real prompt (not only tool results)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] != "user":
            continue
        c = m["content"]
        if isinstance(c, str):
            return i
        if any(b.get("type") == "text" for b in c) and not all(b.get("type") == "tool_result" for b in c):
            return i
    return 0


def prune(messages: list[dict[str, Any]], keep_recent: int = 8) -> int:
    """Stub out old tool results and old thinking. Returns estimated chars saved."""
    saved = 0
    # collect tool_result locations, newest first
    locs = []
    for mi, m in enumerate(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for bi, b in enumerate(m["content"]):
                if b.get("type") == "tool_result":
                    locs.append((mi, bi))
    tool_names = {}
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    tool_names[b["id"]] = (b.get("name"), b.get("input"))
    for mi, bi in locs[:-keep_recent] if keep_recent else locs:
        b = messages[mi]["content"][bi]
        n = _block_text_len(b)
        if n < STUB_MIN_CHARS or b.get("_pruned"):
            continue
        name, inp = tool_names.get(b.get("tool_use_id"), ("tool", {}))
        c = b.get("content")
        text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c if x.get("type") == "text")
        head = text[:300].rstrip()
        tail = text[-300:].lstrip() if len(text) > 900 else ""
        arg = json.dumps(inp, ensure_ascii=False)[:160] if inp else ""
        b["content"] = (f"[pruned old {name} output ({n} chars) to save context; args={arg}. "
                        f"Re-run the tool if you need it again.]\n{head}" + (f"\n...\n{tail}" if tail else ""))
        b["_pruned"] = True
        saved += n - len(b["content"])
    # drop thinking blocks from completed turns
    lh = last_human_index(messages)
    for m in messages[:lh]:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            before = len(m["content"])
            kept = [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]
            if len(kept) != before:
                saved += sum(len(b.get("thinking", "")) for b in m["content"] if b.get("type") == "thinking")
                m["content"] = kept or [{"type": "text", "text": "(thinking)"}]
    # also shrink giant tool_use inputs (e.g. write_file content) in old turns
    for m in messages[:lh]:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_use" and isinstance(b.get("input"), dict):
                    for k, v in list(b["input"].items()):
                        if isinstance(v, str) and len(v) > 3000:
                            b["input"][k] = v[:400] + f"\n...[{len(v) - 400} chars elided from history]"
                            saved += len(v) - 400
    log.info("pruned context, saved ~%d chars", saved)
    return saved


def strip_thinking(messages: list[dict[str, Any]]) -> int:
    """Remove all thinking blocks (needed when switching model: signatures are model-bound)."""
    n = 0
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            kept = [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]
            if len(kept) != len(m["content"]):
                n += len(m["content"]) - len(kept)
                m["content"] = kept or [{"type": "text", "text": "(thinking)"}]
    return n


def clean_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove internal markers before sending."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [{k: v for k, v in b.items() if not k.startswith("_")} for b in c]
        out.append({"role": m["role"], "content": c})
    return out


COMPACT_PROMPT = """Your context window is almost full. Write a dense, complete handoff summary of the work \
so far so that you can continue seamlessly from it alone (the earlier conversation will be deleted).
Include, in this order:
1. The user's original request(s), verbatim if short, and any later instructions/preferences.
2. Key facts about the project: architecture, important files/modules and their roles, build/test/lint \
commands that work, conventions.
3. Everything done so far: files created/modified (with what changed), commands run and results, commits, \
branch name, pushes, PR/issue URLs, CI status.
4. Current state: what is in progress, failing tests/errors with exact messages, hypotheses.
5. Precise next steps to finish the task.
Be specific (paths, function names, line numbers, exact commands). Do not call tools. Output only the summary."""


def compact(agent) -> str:
    """Summarise history with one model call and restart the conversation from it."""
    s = agent.session
    msgs = clean_for_api(s.messages)
    # never end with assistant tool_use without results (repair ensures pairing)
    req = msgs + [{"role": "user", "content": COMPACT_PROMPT}]
    if msgs and msgs[-1]["role"] == "user":
        # merge into last user message to keep alternation
        last = msgs[-1]
        content = last["content"] if isinstance(last["content"], list) else [{"type": "text", "text": last["content"]}]
        req = msgs[:-1] + [{"role": "user", "content": content + [{"type": "text", "text": COMPACT_PROMPT}]}]
    # Same system, tools and thinking setting as the main loop: changing any of them would
    # invalidate the prompt cache, and this call re-sends the entire (large) history.
    resp = agent.llm.create(s.system, req, tools=agent.tool_specs(), max_tokens=None, thinking=True,
                            cb=agent.quiet_cb())
    agent.account(resp.usage)
    summary = resp.text.strip() or "(summary unavailable)"
    todos = "\n".join(f"- [{t['status']}] {t['content']}" for t in s.todos) or "(none)"
    touched = "\n".join(sorted(agent.ctx.touched_files)[-60:]) or "(none)"
    s.messages = [{
        "role": "user",
        "content": (f"<compacted_context n=\"{s.compactions + 1}\">\n{summary}\n</compacted_context>\n\n"
                    f"<current_plan>\n{todos}\n</current_plan>\n<files_touched>\n{touched}\n</files_touched>\n\n"
                    "The conversation was compacted to save context. Continue the task from where you left off. "
                    "Re-read any file before editing it."),
    }]
    s.compactions += 1
    agent.ctx.read_files.clear()
    agent.refresh_system(force=True)  # cache is cold anyway: pick up fresh env/memory
    s.last_prompt_tokens = estimate_tokens(s.messages) + estimate_tokens(s.system)
    log.info("compacted session %s (#%d)", s.id, s.compactions)
    return summary
