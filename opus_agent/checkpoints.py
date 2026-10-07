"""File checkpoints: every write/edit/delete made by the agent is backed up so
`/undo` can roll back changes even outside git."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path


class Checkpoints:
    def __init__(self, root: Path, session_id: str) -> None:
        self.dir = root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "index.jsonl"

    def backup(self, path: Path, turn: int = 0) -> None:
        entry = {"path": str(path), "time": time.time(), "turn": turn, "existed": path.exists()}
        if path.exists() and path.is_file():
            blob = self.dir / uuid.uuid4().hex
            shutil.copy2(path, blob)
            entry["blob"] = blob.name
        elif path.exists() and path.is_dir():
            blob = self.dir / (uuid.uuid4().hex + ".dir")
            shutil.copytree(path, blob, symlinks=True)
            entry["blob"] = blob.name
            entry["is_dir"] = True
        with self.index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self) -> list[dict]:
        if not self.index.exists():
            return []
        out = []
        for line in self.index.read_text("utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    def undo(self, turns: int = 1) -> list[str]:
        """Revert all changes made in the last `turns` distinct turns."""
        ents = self.entries()
        if not ents:
            return []
        turn_ids = sorted({e.get("turn", 0) for e in ents})
        target = set(turn_ids[-turns:])
        restored: list[str] = []
        keep = []
        for e in reversed(ents):
            if e.get("turn", 0) not in target:
                keep.append(e)
                continue
            p = Path(e["path"])
            if e.get("existed") and e.get("blob"):
                src = self.dir / e["blob"]
                if p.exists():
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                p.parent.mkdir(parents=True, exist_ok=True)
                if e.get("is_dir"):
                    shutil.copytree(src, p, symlinks=True)
                else:
                    shutil.copy2(src, p)
            elif not e.get("existed"):
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
            restored.append(str(p))
        keep.reverse()
        self.index.write_text("".join(json.dumps(e) + "\n" for e in keep), "utf-8")
        return list(dict.fromkeys(restored))
