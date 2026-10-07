"""Persistent memory.

* global memory   ~/.opus-agent/memory/global.md         (user preferences, environment facts)
* project memory  ~/.opus-agent/memory/projects/<key>.md  (architecture notes, commands, gotchas)
* task journal    ~/.opus-agent/memory/projects/<key>.journal.md (short summaries of past tasks)
* repo instruction files: OPUS.md / CLAUDE.md / AGENTS.md / .github/copilot-instructions.md
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import time
from pathlib import Path

INSTRUCTION_FILES = ("OPUS.md", "CLAUDE.md", "AGENTS.md", ".github/copilot-instructions.md", ".cursorrules")
MAX_MEMORY_CHARS = 12000
MAX_JOURNAL_ENTRIES = 12


def project_root(cwd: Path) -> Path:
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(cwd), capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cwd


def project_key(root: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", root.name)[:40] or "root"
    return f"{name}-{hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:8]}"


class Memory:
    def __init__(self, home: Path, cwd: Path) -> None:
        self.dir = home / "memory"
        (self.dir / "projects").mkdir(parents=True, exist_ok=True)
        self.set_cwd(cwd)

    def set_cwd(self, cwd: Path) -> None:
        self.root = project_root(cwd)
        self.key = project_key(self.root)
        self.global_path = self.dir / "global.md"
        self.project_path = self.dir / "projects" / f"{self.key}.md"
        self.journal_path = self.dir / "projects" / f"{self.key}.journal.md"

    def path(self, scope: str) -> Path:
        return self.global_path if scope == "global" else self.project_path

    def read(self, scope: str) -> str:
        p = self.path(scope)
        return p.read_text("utf-8") if p.exists() else ""

    def write(self, scope: str, text: str) -> None:
        p = self.path(scope)
        p.write_text(text.strip() + "\n", "utf-8")

    def add(self, scope: str, text: str) -> str:
        cur = self.read(scope).rstrip()
        entry = "- " + text.strip().replace("\n", "\n  ")
        new = (cur + "\n" + entry) if cur else entry
        if len(new) > MAX_MEMORY_CHARS:
            return (f"Memory '{scope}' would exceed {MAX_MEMORY_CHARS} chars. Consolidate it first with "
                    "action=rewrite (merge/remove outdated entries).")
        self.write(scope, new)
        return f"Saved to {scope} memory."

    def replace(self, scope: str, old: str, new: str) -> str:
        cur = self.read(scope)
        if old not in cur:
            return "Text not found in memory."
        self.write(scope, cur.replace(old, new, 1))
        return "Memory updated."

    def journal(self, task: str, summary: str) -> None:
        entries = self.journal_entries()
        stamp = time.strftime("%Y-%m-%d %H:%M")
        entries.append(f"## {stamp} — {task.strip()[:150]}\n{summary.strip()[:1200]}")
        entries = entries[-MAX_JOURNAL_ENTRIES:]
        self.journal_path.write_text("\n\n".join(entries) + "\n", "utf-8")

    def journal_entries(self) -> list[str]:
        if not self.journal_path.exists():
            return []
        txt = self.journal_path.read_text("utf-8")
        parts = re.split(r"\n(?=## \d{4}-)", txt.strip())
        return [p for p in parts if p.strip()]

    def instruction_files(self) -> list[tuple[str, str]]:
        out = []
        seen = set()
        for base in dict.fromkeys([self.root, Path.cwd()]):
            for name in INSTRUCTION_FILES:
                p = base / name
                if p.exists() and p.is_file() and p.resolve() not in seen:
                    seen.add(p.resolve())
                    try:
                        out.append((str(p), p.read_text("utf-8", errors="replace")[:20000]))
                    except OSError:
                        pass
        return out

    def render(self) -> str:
        parts = []
        g = self.read("global").strip()
        if g:
            parts.append(f"<global_memory>\n{g}\n</global_memory>")
        pm = self.read("project").strip()
        if pm:
            parts.append(f"<project_memory project=\"{self.root}\">\n{pm}\n</project_memory>")
        j = [e[:700] for e in self.journal_entries()[-3:]]
        if j:
            parts.append("<recent_tasks_in_this_project>\n" + "\n\n".join(j) + "\n</recent_tasks_in_this_project>")
        for path, text in self.instruction_files():
            parts.append(f"<project_instructions file=\"{path}\">\n{text}\n</project_instructions>")
        return "\n\n".join(parts)
