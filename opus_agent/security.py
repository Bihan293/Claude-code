"""Secret redaction + logging that never leaks credentials."""
from __future__ import annotations

import logging
import logging.handlers
import re
import threading
from pathlib import Path

_lock = threading.Lock()
_secrets: set[str] = set()

# Generic token patterns (redacted even if not registered)
_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}\b"),  # telegram bot token
    re.compile(r"(?i)(authorization:\s*(?:bearer|token)\s+)[^\s\"']+"),
    re.compile(r"(?i)(x-api-key:\s*)[^\s\"']+"),
    re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+@"),  # creds in URLs
]


def register_secret(value: str | None) -> None:
    if value and len(value) >= 6:
        with _lock:
            _secrets.add(value)


def redact(text: str) -> str:
    if not text:
        return text
    if not isinstance(text, str):
        text = str(text)
    with _lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for s in secrets:
        if s in text:
            text = text.replace(s, "***REDACTED***")
    for p in _PATTERNS:
        if p.groups:
            text = p.sub(lambda m: m.group(1) + "***REDACTED***", text)
        else:
            text = p.sub("***REDACTED***", text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(log_dir: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("opus")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    h = logging.handlers.RotatingFileHandler(
        log_dir / "agent.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    h.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    try:
        (log_dir / "agent.log").chmod(0o600)
    except OSError:
        pass
    return logger


log = logging.getLogger("opus")
