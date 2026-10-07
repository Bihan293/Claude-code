"""Interactive setup: API key, GitHub token, git identity, Telegram."""
from __future__ import annotations

import getpass
import subprocess

import httpx

from .security import redact, register_secret


def _ask(prompt: str, default: str = "") -> str:
    try:
        v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        v = ""
    return v or default


def _secret(prompt: str) -> str:
    try:
        return getpass.getpass(f"{prompt} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def check_api(cfg, key: str) -> tuple[bool, str]:
    from .llm import LLMClient, LLMError
    c = LLMClient(cfg, key)
    c.thinking_mode = "off"
    c.allow_stream = False
    try:
        r = c.create("Reply with the single word: ok", [{"role": "user", "content": "ping"}],
                     max_tokens=16, thinking=False)
        return True, f"model replied: {r.text.strip()[:40]!r}"
    except LLMError as e:
        return False, str(e)
    except httpx.HTTPError as e:
        return False, redact(str(e))


def setup_api(cfg, force: bool = False) -> bool:
    print("\n== Tooken Club / Anthropic API ==")
    if cfg.secret("api_key") and not force:
        print("API key already configured.")
        return True
    base = _ask("Base URL", cfg.get("base_url"))
    model = _ask("Model", cfg.get("model"))
    cfg.set("base_url", base)
    cfg.set("model", model)
    for _ in range(3):
        key = _secret("API key")
        if not key:
            print("Empty key.")
            continue
        register_secret(key)
        print("Checking key…")
        ok, msg = check_api(cfg, key)
        if ok:
            cfg.set_secret("api_key", key)
            print(f"✔ API OK ({msg}). Key saved to {cfg.cred_path} (chmod 600).")
            return True
        print(f"✖ check failed: {msg}")
        if _ask("Save anyway? (y/N)", "n").lower().startswith("y"):
            cfg.set_secret("api_key", key)
            return True
    return False


def setup_github(cfg) -> None:
    print("\n== GitHub ==")
    print("Create a token: https://github.com/settings/tokens (classic: scopes repo, workflow; or fine-grained:\n"
          "Contents, Pull requests, Issues, Actions, Checks = read/write). Leave empty to skip.")
    tok = _secret("GitHub token")
    if not tok:
        return
    register_secret(tok)
    cfg.set_secret("github_token", tok)
    from .tools.base import ToolError
    from .tools.github import gh_request
    try:
        u = gh_request(cfg, "GET", "user")
        print(f"✔ GitHub authenticated as {u['login']}")
        if not cfg.get("git_author_name"):
            cfg.set("git_author_name", u.get("name") or u["login"])
        if not cfg.get("git_author_email"):
            email = u.get("email") or f"{u['id']}+{u['login']}@users.noreply.github.com"
            cfg.set("git_author_email", email)
        _ensure_git_identity(cfg)
    except ToolError as e:
        print(f"✖ {e}")


def _ensure_git_identity(cfg) -> None:
    for key, val in (("user.name", cfg.get("git_author_name")), ("user.email", cfg.get("git_author_email"))):
        try:
            cur = subprocess.run(["git", "config", "--global", key], capture_output=True, text=True).stdout.strip()
            if not cur and val:
                subprocess.run(["git", "config", "--global", key, val], check=False)
        except OSError:
            pass
    try:
        subprocess.run(["git", "config", "--global", "init.defaultBranch", "main"], check=False,
                       capture_output=True)
    except OSError:
        pass


def setup_telegram(cfg) -> None:
    from .notify import telegram_discover_chat, telegram_test
    print("\n== Telegram notifications (optional) ==")
    print("1) Create a bot with @BotFather, copy the token. 2) Send /start to your bot. Empty = skip.")
    tok = _secret("Bot token")
    if not tok:
        if _ask("Disable Telegram notifications? (y/N)", "n").lower().startswith("y"):
            cfg.set("telegram_enabled", False)
        return
    register_secret(tok)
    cfg.set_secret("telegram_bot_token", tok)
    chat = telegram_discover_chat(tok) or ""
    chat = _ask("Chat ID", chat)
    cfg.set("telegram_chat_id", chat)
    cfg.set("telegram_enabled", True)
    print("Test:", telegram_test(cfg))


def first_run(cfg) -> bool:
    print("Welcome to opus-agent — autonomous coding agent for Termux.")
    if not setup_api(cfg):
        print("API key is required. Run `opus setup` again.")
        return False
    if not cfg.secret("github_token") and _ask("Connect GitHub now? (Y/n)", "y").lower().startswith("y"):
        setup_github(cfg)
    if not cfg.get("telegram_enabled") and _ask("Set up Telegram notifications? (y/N)", "n").lower().startswith("y"):
        setup_telegram(cfg)
    print("\nSetup complete. Change settings later: opus config, opus github login, opus telegram setup\n")
    return True
