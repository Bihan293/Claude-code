"""Filesystem tools: read, write, edit, multi_edit, delete, move, list, glob, grep."""
from __future__ import annotations

import base64
import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import ToolContext, ToolError, tool

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
               "dist", "build", ".gradle", ".idea", "target", ".next", ".cache", ".tox", "coverage",
               ".ruff_cache", ".dart_tool", "Pods"}
IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
             ".webp": "image/webp"}
MAX_READ_LINES = 2000


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
        return b"\0" in chunk
    except OSError:
        return False


def _backup(ctx: ToolContext, path: Path) -> None:
    if ctx.checkpoints is not None:
        turn = getattr(ctx.session, "turn", 0) if ctx.session else 0
        ctx.checkpoints.backup(path, turn)
    ctx.touched_files.add(str(path))


def _check_fresh(ctx: ToolContext, path: Path) -> None:
    """Refuse to edit a file that was modified externally since we last read it."""
    key = str(path)
    if key in ctx.read_files and path.exists():
        if abs(path.stat().st_mtime - ctx.read_files[key]) > 1e-6:
            # allow if we touched it ourselves via shell; just warn by re-reading requirement
            ctx.read_files.pop(key, None)
            raise ToolError(f"{path} changed on disk since you last read it. Read it again before editing.")


def _mark_read(ctx: ToolContext, path: Path) -> None:
    try:
        ctx.read_files[str(path)] = path.stat().st_mtime
    except OSError:
        pass


@tool("read_file",
      """Read a text file and return it with line numbers (cat -n format). Use offset/limit
(1-based line numbers) for big files instead of reading everything. Images (png/jpg/gif/webp)
are returned for visual inspection. Read several files in parallel by emitting multiple
tool calls in one response.""",
      {"path": {"type": "string", "description": "File path (absolute or relative to cwd)"},
       "offset": {"type": "integer", "description": "First line to read (1-based)"},
       "limit": {"type": "integer", "description": "Max number of lines (default 2000)"}},
      ["path"])
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> Any:
    p = ctx.resolve(path)
    if not p.exists():
        parent = p.parent
        hint = ""
        if parent.exists():
            sims = [x.name for x in parent.iterdir() if x.name.lower().startswith(p.stem.lower()[:3])][:10]
            if sims:
                hint = f" Similar: {', '.join(sims)}"
        raise ToolError(f"File not found: {p}.{hint}")
    if p.is_dir():
        raise ToolError(f"{p} is a directory; use list_dir.")
    if p.suffix.lower() in IMAGE_EXT and p.stat().st_size < 4_000_000:
        data = base64.b64encode(p.read_bytes()).decode()
        return [{"type": "image", "source": {"type": "base64", "media_type": IMAGE_EXT[p.suffix.lower()],
                                              "data": data}}]
    if _is_binary(p):
        return f"{p} is a binary file ({p.stat().st_size} bytes)."
    offset = max(1, int(offset or 1))
    limit = max(1, int(limit or MAX_READ_LINES))
    out = []
    total = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            total = i
            if i < offset:
                continue
            if i >= offset + limit:
                continue
            line = line.rstrip("\n")
            if len(line) > 2000:
                line = line[:2000] + " ...[line truncated]"
            out.append(f"{i:6}\t{line}")
    _mark_read(ctx, p)
    if not out:
        return f"(file has {total} lines; nothing at offset {offset})" if total else "(empty file)"
    tail = ""
    if offset + limit - 1 < total:
        tail = f"\n... ({total - (offset + limit - 1)} more lines; total {total}. Use offset={offset + limit})"
    return "\n".join(out) + tail


@tool("write_file",
      """Create or overwrite a file with the given full content (parent dirs are created).
Prefer edit_file for modifying existing files – it is far cheaper in tokens.
You must read_file an existing file before overwriting it.""",
      {"path": {"type": "string"}, "content": {"type": "string"}},
      ["path", "content"], writes=True, readonly_ok=False)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    p = ctx.resolve(path)
    if p.exists() and p.is_dir():
        raise ToolError(f"{p} is a directory")
    existed = p.exists()
    if existed and str(p) not in ctx.read_files and p.stat().st_size > 0:
        raise ToolError(f"{p} exists. Read it first (read_file) or use edit_file.")
    if existed:
        _check_fresh(ctx, p)
    _backup(ctx, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    _mark_read(ctx, p)
    n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"{'Overwrote' if existed else 'Created'} {p} ({n} lines)"


def _apply_edit(text: str, old: str, new: str, replace_all: bool, path: Path) -> tuple[str, int]:
    if old == new:
        raise ToolError("old_string and new_string are identical")
    if old == "":
        if text:
            raise ToolError("old_string is empty but file is not empty")
        return new, 1
    count = text.count(old)
    if count == 0:
        # tolerant fallback: normalise trailing whitespace / CRLF
        norm_text = text.replace("\r\n", "\n")
        norm_old = old.replace("\r\n", "\n")
        if norm_text.count(norm_old) == 1:
            return norm_text.replace(norm_old, new, 1), 1
        stripped = "\n".join(l.rstrip() for l in norm_old.split("\n"))
        lines_text = "\n".join(l.rstrip() for l in norm_text.split("\n"))
        if stripped and lines_text.count(stripped) == 1:
            return lines_text.replace(stripped, new, 1), 1
        first = old.strip().split("\n")[0][:80]
        hint = ""
        if first:
            for i, line in enumerate(text.split("\n"), 1):
                if first.strip() and first.strip() in line:
                    hint = f" A similar line exists at line {i}: {line.strip()[:120]!r}"
                    break
        raise ToolError(f"old_string not found in {path}. It must match exactly, including "
                        f"indentation. Re-read the file.{hint}")
    if count > 1 and not replace_all:
        raise ToolError(f"old_string occurs {count} times in {path}. Add more surrounding context to make it "
                        "unique, or set replace_all=true.")
    return (text.replace(old, new) if replace_all else text.replace(old, new, 1)), count


def _snippet(text: str, new: str) -> str:
    idx = text.find(new) if new else -1
    if idx < 0:
        return ""
    start_line = text.count("\n", 0, idx) + 1
    lines = text.split("\n")
    a = max(0, start_line - 3)
    b = min(len(lines), start_line + new.count("\n") + 2)
    if b - a > 40:
        return ""
    return "\n".join(f"{i + 1:6}\t{lines[i]}" for i in range(a, b))


@tool("edit_file",
      """Exact string replacement in a file. old_string must match the file exactly (including
indentation, without the line-number prefix from read_file) and be unique unless replace_all=true.
To create a new file use write_file. Read the file before editing.""",
      {"path": {"type": "string"},
       "old_string": {"type": "string", "description": "Exact text to replace"},
       "new_string": {"type": "string", "description": "Replacement text"},
       "replace_all": {"type": "boolean", "description": "Replace every occurrence"}},
      ["path", "old_string", "new_string"], writes=True, readonly_ok=False)
def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = ctx.resolve(path)
    if not p.exists():
        if old_string == "":
            return write_file(ctx, path, new_string)
        raise ToolError(f"File not found: {p}")
    _check_fresh(ctx, p)
    text = p.read_text(encoding="utf-8", errors="replace")
    new_text, n = _apply_edit(text, old_string, new_string, replace_all, p)
    _backup(ctx, p)
    p.write_text(new_text, encoding="utf-8")
    _mark_read(ctx, p)
    snip = _snippet(new_text, new_string)
    return f"Edited {p} ({n} replacement{'s' if n > 1 else ''})." + (f"\n{snip}" if snip else "")


@tool("multi_edit",
      """Apply several exact replacements to ONE file atomically (all or nothing), in order.
Each edit has old_string, new_string, optional replace_all. Cheaper than many edit_file calls.""",
      {"path": {"type": "string"},
       "edits": {"type": "array", "items": {"type": "object", "properties": {
           "old_string": {"type": "string"}, "new_string": {"type": "string"},
           "replace_all": {"type": "boolean"}}, "required": ["old_string", "new_string"]}}},
      ["path", "edits"], writes=True, readonly_ok=False)
def multi_edit(ctx: ToolContext, path: str, edits: list[dict[str, Any]]) -> str:
    p = ctx.resolve(path)
    if not edits:
        raise ToolError("edits is empty")
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    if p.exists():
        _check_fresh(ctx, p)
    total = 0
    for i, e in enumerate(edits, 1):
        try:
            text, n = _apply_edit(text, e.get("old_string", ""), e.get("new_string", ""),
                                  bool(e.get("replace_all")), p)
        except ToolError as err:
            raise ToolError(f"edit #{i} failed (no changes written): {err}")
        total += n
    _backup(ctx, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _mark_read(ctx, p)
    return f"Applied {len(edits)} edits ({total} replacements) to {p}"


@tool("delete_path", "Delete a file or directory (recursively). Backed up for /undo.",
      {"path": {"type": "string"}}, ["path"], writes=True, readonly_ok=False)
def delete_path(ctx: ToolContext, path: str) -> str:
    p = ctx.resolve(path)
    if not p.exists() and not p.is_symlink():
        raise ToolError(f"Not found: {p}")
    if p in (Path.home(), Path("/"), ctx.cwd) or len(p.parts) <= 2:
        raise ToolError(f"Refusing to delete {p}")
    _backup(ctx, p)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()
    ctx.read_files.pop(str(p), None)
    return f"Deleted {p}"


@tool("move_path", "Move/rename a file or directory.",
      {"source": {"type": "string"}, "destination": {"type": "string"}},
      ["source", "destination"], writes=True, readonly_ok=False)
def move_path(ctx: ToolContext, source: str, destination: str) -> str:
    s, d = ctx.resolve(source), ctx.resolve(destination)
    if not s.exists():
        raise ToolError(f"Not found: {s}")
    if d.exists():
        raise ToolError(f"Destination exists: {d}")
    _backup(ctx, s)
    _backup(ctx, d)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"Moved {s} -> {d}"


def _walk(root: Path, max_depth: int = 99):
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.endswith(".egg-info"))
        if len(dp.parts) - root_depth >= max_depth:
            dirnames[:] = []
        yield dp, dirnames, sorted(filenames)


@tool("list_dir",
      "List a directory as a tree (ignores .git, node_modules, build dirs, etc).",
      {"path": {"type": "string", "description": "Directory (default cwd)"},
       "depth": {"type": "integer", "description": "Max depth (default 2)"}})
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2) -> str:
    root = ctx.resolve(path or ".")
    if not root.is_dir():
        raise ToolError(f"Not a directory: {root}")
    lines = [f"{root}/"]
    count = 0
    for dp, dirs, files in _walk(root, max(1, int(depth))):
        level = len(dp.parts) - len(root.parts)
        ind = "  " * (level + 1)
        if dp != root:
            lines.append("  " * level + f"{dp.name}/")
        for f in files:
            count += 1
            if count > 1500:
                lines.append("... (truncated, use glob)")
                return "\n".join(lines)
            try:
                sz = (dp / f).stat().st_size
            except OSError:
                sz = 0
            lines.append(f"{ind}{f}  ({_human(sz)})")
        if level + 1 >= depth:
            for d in dirs:
                lines.append(f"{ind}{d}/ ...")
    return "\n".join(lines)


def _human(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}T"


@tool("glob", "Find files by glob pattern, e.g. '**/*.py' or 'src/**/test_*.ts'. Sorted by mtime (newest first).",
      {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Base dir (default cwd)"}},
      ["pattern"])
def glob_tool(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    root = ctx.resolve(path or ".")
    res = []
    for dp, _dirs, files in _walk(root):
        for f in files:
            full = dp / f
            rel = str(full.relative_to(root))
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f, pattern) or \
                    (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])):
                res.append(full)
        if len(res) > 5000:
            break
    res.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    if not res:
        return "No files matched."
    shown = res[:500]
    return "\n".join(str(r.relative_to(root)) for r in shown) + \
        (f"\n... {len(res) - 500} more" if len(res) > 500 else "")


@tool("grep",
      """Search file contents with a regex (ripgrep if available). Modes: 'content' (matching lines
with line numbers, default), 'files' (only paths), 'count'. Use 'glob' to filter files, e.g. '*.py'.""",
      {"pattern": {"type": "string", "description": "Regex"},
       "path": {"type": "string", "description": "File or dir (default cwd)"},
       "glob": {"type": "string", "description": "File filter, e.g. '*.{ts,tsx}'"},
       "mode": {"type": "string", "enum": ["content", "files", "count"]},
       "ignore_case": {"type": "boolean"},
       "context": {"type": "integer", "description": "Lines of context around matches"},
       "max_results": {"type": "integer"}},
      ["pattern"])
def grep(ctx: ToolContext, pattern: str, path: str = ".", glob: str = "", mode: str = "content",
         ignore_case: bool = False, context: int = 0, max_results: int = 300) -> str:
    root = ctx.resolve(path or ".")
    rg = shutil.which("rg")
    max_results = max(1, int(max_results or 300))
    if rg:
        cmd = [rg, "--no-heading", "--color=never", "-n", "--max-columns=400", "--max-columns-preview"]
        if ignore_case:
            cmd.append("-i")
        if mode == "files":
            cmd = [rg, "-l", "--color=never"] + (["-i"] if ignore_case else [])
        elif mode == "count":
            cmd = [rg, "-c", "--color=never"] + (["-i"] if ignore_case else [])
        elif context:
            cmd += ["-C", str(int(context))]
        if glob:
            cmd += ["--glob", glob]
        cmd += ["-e", pattern, str(root)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, errors="replace")
        except subprocess.TimeoutExpired:
            raise ToolError("search timed out; narrow the path or pattern")
        if r.returncode not in (0, 1):
            raise ToolError(r.stderr.strip()[:2000] or "rg failed")
        lines = r.stdout.splitlines()
    else:
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"bad regex: {e}")
        lines = []
        files = [root] if root.is_file() else [dp / f for dp, _d, fs in _walk(root) for f in fs]
        for fp in files:
            if glob and not _glob_match(fp.name, glob):
                continue
            if _is_binary(fp):
                continue
            try:
                txt = fp.read_text("utf-8", errors="replace").splitlines()
            except OSError:
                continue
            hits = [i for i, l in enumerate(txt) if rx.search(l)]
            if not hits:
                continue
            if mode == "files":
                lines.append(str(fp))
            elif mode == "count":
                lines.append(f"{fp}:{len(hits)}")
            else:
                for i in hits:
                    lines.append(f"{fp}:{i + 1}:{txt[i][:400]}")
            if len(lines) > max_results * 2:
                break
    if not lines:
        return "No matches."
    roots = str(root) + os.sep
    lines = [l.replace(roots, "", 1) if l.startswith(roots) else l for l in lines]
    more = len(lines) - max_results
    return "\n".join(lines[:max_results]) + (f"\n... {more} more lines (refine the search)" if more > 0 else "")


def _glob_match(name: str, pattern: str) -> bool:
    m = re.match(r"^(.*)\{(.+)\}(.*)$", pattern)
    if m:
        return any(fnmatch.fnmatch(name, m.group(1) + alt + m.group(3)) for alt in m.group(2).split(","))
    return fnmatch.fnmatch(name, pattern)
