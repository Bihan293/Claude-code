"""Optional notifications: Telegram (bot API) and Termux:API (termux-notification).
Failures are swallowed – the agent never depends on them."""
from __future__ import annotations

import shutil
import subprocess
import threading

import httpx

from .security import log, redact

ICONS = {"done": "✅", "error": "❌", "pr": "🔀", "info": "ℹ️", "question": "❓"}


def _telegram(cfg, text: str) -> None:
    token = cfg.secret("telegram_bot_token")
    chat = cfg.get("telegram_chat_id")
    if not (token and chat):
        return
    try:
        httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   json={"chat_id": chat, "text": text[:4000], "disable_web_page_preview": True},
                   timeout=15)
    except Exception as e:  # noqa: BLE001
        log.warning("telegram failed: %s", redact(str(e)))


def _termux(text: str, kind: str) -> None:
    if not shutil.which("termux-notification"):
        return
    try:
        subprocess.run(["termux-notification", "--title", f"opus {ICONS.get(kind, '')} {kind}",
                        "--content", text[:500], "--id", "opus-agent"], timeout=10,
                       capture_output=True)
    except (OSError, subprocess.TimeoutExpired):
        pass


def notify(cfg, text: str, kind: str = "info", wait: bool = False) -> None:
    text = redact(f"{ICONS.get(kind, '')} opus-agent: {text}")
    jobs = []
    if cfg.get("telegram_enabled"):
        jobs.append(threading.Thread(target=_telegram, args=(cfg, text), daemon=True))
    if cfg.get("termux_notifications"):
        jobs.append(threading.Thread(target=_termux, args=(text, kind), daemon=True))
    for j in jobs:
        j.start()
    if wait:
        for j in jobs:
            j.join(timeout=20)


def telegram_test(cfg) -> str:
    token = cfg.secret("telegram_bot_token")
    if not token:
        return "no bot token"
    r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   json={"chat_id": cfg.get("telegram_chat_id"), "text": "opus-agent: test notification ✅"},
                   timeout=15)
    return "ok" if r.status_code == 200 else redact(f"HTTP {r.status_code}: {r.text[:300]}")


def telegram_discover_chat(token: str) -> str | None:
    """Read getUpdates to find the chat id after the user sent /start to the bot."""
    try:
        r = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15)
        for u in reversed(r.json().get("result", [])):
            msg = u.get("message") or u.get("channel_post") or {}
            if msg.get("chat", {}).get("id"):
                return str(msg["chat"]["id"])
    except Exception:  # noqa: BLE001
        return None
    return None


def wake_lock(on: bool) -> None:
    exe = "termux-wake-lock" if on else "termux-wake-unlock"
    if shutil.which(exe):
        try:
            subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
