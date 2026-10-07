"""Git/gh authentication through environment only (token never written into
remote URLs, .git/config, command lines or logs)."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from ..config import home_dir

_ASKPASS = """#!/bin/sh
case "$1" in
  *sername*) echo "x-access-token" ;;
  *) echo "$OPUS_GIT_TOKEN" ;;
esac
"""


def askpass_path() -> Path:
    p = home_dir() / "git-askpass.sh"
    if not p.exists() or p.read_text() != _ASKPASS:
        p.write_text(_ASKPASS)
        p.chmod(stat.S_IRWXU)
    return p


def git_env(cfg) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TERM", "dumb")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PAGER"] = "cat"
    env["GIT_PAGER"] = "cat"
    env["GH_PAGER"] = "cat"
    env["GH_PROMPT_DISABLED"] = "1"
    env["NO_COLOR"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["DEBIAN_FRONTEND"] = "noninteractive"
    env.setdefault("CI", "1")
    env.setdefault("EDITOR", "true")
    env["GIT_EDITOR"] = "true"
    tok = cfg.secret("github_token")
    if tok:
        env["OPUS_GIT_TOKEN"] = tok
        env["GIT_ASKPASS"] = str(askpass_path())
        env["GH_TOKEN"] = tok
        env["GITHUB_TOKEN"] = tok
    name, email = cfg.get("git_author_name"), cfg.get("git_author_email")
    if name:
        env.setdefault("GIT_AUTHOR_NAME", name)
        env.setdefault("GIT_COMMITTER_NAME", name)
    if email:
        env.setdefault("GIT_AUTHOR_EMAIL", email)
        env.setdefault("GIT_COMMITTER_EMAIL", email)
    # never leak the LLM key to child processes
    for k in ("TOOKEN_API_KEY", "OPUS_API_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"):
        env.pop(k, None)
    return env
