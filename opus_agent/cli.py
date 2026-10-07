"""Command line interface.

    opus                       interactive REPL in the current directory
    opus "task"                REPL, starting with this task
    opus -p "task"             headless: run task autonomously, print report, exit
    opus -c / --continue       continue the latest session of this project
    opus -r ID / --resume ID   resume a specific session
    opus --detach "task"       run headless in background (survives closing the REPL), log to file
    opus setup | config | github | clone | sessions | status | usage | memory | telegram | doctor | logs
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import __version__
from .config import Config
from .security import log, redact, register_secret, setup_logging

SUBCOMMANDS = {"setup", "config", "github", "clone", "sessions", "status", "usage", "memory", "telegram",
               "doctor", "logs", "projects", "help", "version"}

SLASH_HELP = """[bold]Slash commands[/]
  /help                 this help
  /status               session, branch, context size, tokens
  /usage, /cost         token usage & estimated cost
  /todos                current plan
  /compact              summarise conversation now (frees context)
  /clear, /new          start a fresh session (memory is kept)
  /continue             continue an interrupted task
  /sessions             list sessions;  /resume <id>
  /mode auto|ask|plan   autonomy: auto=full, ask=confirm writes/shell, plan=read-only planning
  /model [name]         show/switch model;  /thinking adaptive|enabled|off
  /undo [n]             revert file changes of the last n turns
  /diff                 git diff --stat of the working tree
  /cd <path>            change project directory;  /projects  list ~/projects
  /clone <owner/repo>   clone a GitHub repo and switch to it
  /memory [global]      show memory;  /remember <text>  save a project note
  /init                 analyse repo and write OPUS.md (project instructions)
  /issue <n>            implement GitHub issue #n end-to-end and open a PR
  /pr                   commit current work on a branch, push and open/update a PR
  /review [n]           review PR #n (or current diff)
  /fixci                check CI for the current branch and fix failures
  /verbose              toggle detailed tool output;  /exit  quit
[dim]Multi-line input: end a line with \\ or paste text. Esc+Enter inserts a newline. Ctrl+C interrupts the agent.[/]"""

TEMPLATES = {
    "init": ("Analyse this repository thoroughly (use sub-agents for exploration if it is large) and create or "
             "update OPUS.md in the repo root: a concise guide for an AI coding agent with project overview, "
             "architecture map (key dirs/files and roles), exact build/test/lint/run commands that work here, "
             "code conventions, and gotchas. Verify the commands by running them. Also save the key facts to "
             "project memory. Do not commit unless I ask."),
    "issue": ("Implement GitHub issue #{arg} end-to-end: read the issue and its comments, study the relevant "
              "code, create a feature branch, implement the change with tests, run all checks and fix problems, "
              "commit, push, open a PR that says 'Closes #{arg}', then check CI and fix failures. Finish with "
              "the final report including the PR URL."),
    "pr": ("Take the current uncommitted/unpushed work in this repo: review the diff for problems, run the "
           "project's checks and fix failures, make sure we are on a feature branch (create one if on the "
           "default branch), commit with a good conventional message, push and create or update the PR with a "
           "proper description. Then check CI. Report the PR URL."),
    "review": ("Do a thorough code review of {target}. Look for bugs, edge cases, security issues, missing "
               "tests, style inconsistencies. Run the tests. Give a prioritised list of findings with file:line "
               "references and concrete suggested fixes. Do not change code unless I ask."),
    "fixci": ("Check the CI status for the current branch/PR (github action=checks). If checks are pending, poll "
              "until they finish (max ~15 min). For every failure fetch the logs, reproduce locally if possible, "
              "fix the root cause, run checks locally, commit and push. Repeat until CI is green or the failure "
              "is clearly unrelated to our change (explain). Report the final status."),
}


# ====================================================================== helpers
def make_ui(cfg, args):
    from .ui import UI
    return UI(cfg, verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))


def ensure_api_key(cfg, interactive: bool = True) -> bool:
    if cfg.secret("api_key"):
        return True
    if not interactive or not sys.stdin.isatty():
        print("No API key configured. Run `opus setup` (or set TOOKEN_API_KEY).", file=sys.stderr)
        return False
    from .setup_wizard import first_run
    return first_run(cfg)


def git_out(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def fmt_tokens(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else (f"{n / 1e3:.1f}k" if n >= 1e3 else str(n))


def usage_line(u: dict, cfg) -> str:
    from .usage import cache_hit_rate, cost
    return (f"in {fmt_tokens(u.get('input_tokens', 0))} · out {fmt_tokens(u.get('output_tokens', 0))} · "
            f"cache w {fmt_tokens(u.get('cache_creation_input_tokens', 0))} / r "
            f"{fmt_tokens(u.get('cache_read_input_tokens', 0))} (hit {cache_hit_rate(u):.0%}) · "
            f"req {u.get('requests', 0)} · ≈${cost(u, cfg):.2f}")


# ====================================================================== running
class Runner:
    """Runs the agent in a worker thread so Ctrl+C can interrupt cleanly."""

    def __init__(self, cfg, ui, session=None, interactive: bool = True) -> None:
        from .agent import Agent
        from .session import Session
        self.cfg = cfg
        self.ui = ui
        self.interactive = interactive
        self.session = session or Session(cwd=str(Path.cwd()))
        self.agent = Agent(cfg, self.session, ui, interactive=interactive)

    def new_session(self, cwd: Path | None = None) -> None:
        from .agent import Agent
        from .session import Session
        llm = self.agent.llm
        self.session = Session(cwd=str(cwd or self.agent.ctx.cwd))
        self.agent = Agent(self.cfg, self.session, self.ui, llm=llm, interactive=self.interactive)

    def load(self, sid: str) -> None:
        from .agent import Agent
        from .session import Session
        llm = self.agent.llm
        self.session = Session.load(self.cfg.home, sid)
        self.agent = Agent(self.cfg, self.session, self.ui, llm=llm, interactive=self.interactive)
        os.chdir(self.agent.ctx.cwd)

    def run(self, prompt) -> str | None:
        from .llm import LLMError
        result: dict = {}

        def work() -> None:
            try:
                result["text"] = self.agent.run(prompt)
            except LLMError as e:
                result["error"] = e
            except Exception as e:  # noqa: BLE001
                log.exception("agent crashed")
                result["error"] = e

        th = threading.Thread(target=work, daemon=True)
        t0 = time.time()
        th.start()
        interrupts = 0
        while th.is_alive():
            try:
                th.join(0.2)
            except KeyboardInterrupt:
                interrupts += 1
                self.agent.interrupt()
                self.ui.info("interrupting… (Ctrl+C again to force)")
                if interrupts >= 2:
                    from .tools.shell import kill_all_jobs
                    kill_all_jobs()
                    break
        th.join(5)
        self.ui.end_stream()
        if "error" in result:
            e = result["error"]
            self.ui.error(f"{type(e).__name__}: {e}")
            if getattr(e, "status", None) in (401, 403):
                self.ui.info("API key rejected. Run /setup or `opus setup --force`.")
            self.ui.info("Session saved. Use /continue to retry from this point.")
            return None
        st = self.session.status
        dur = time.time() - t0
        self.ui.console.print(f"[dim]── {st} in {dur:.0f}s · {usage_line(self.session.usage, self.cfg)}[/]")
        prs = list(dict.fromkeys(self.agent.ctx.pr_urls))
        if prs:
            self.ui.console.print("[bold green]PR:[/] " + "  ".join(prs))
        return result.get("text")


# ====================================================================== REPL
def repl(cfg, args, runner: Runner, initial: str | None) -> int:
    ui = runner.ui
    ui.console.print(f"[bold cyan]opus-agent {__version__}[/] · model [bold]{cfg['model']}[/] · "
                     f"mode [bold]{cfg.get('permission_mode')}[/] · [dim]{runner.agent.ctx.cwd}[/]")
    br = git_out(["rev-parse", "--abbrev-ref", "HEAD"], runner.agent.ctx.cwd)
    if br:
        ui.console.print(f"[dim]git branch: {br}[/]")
    if runner.session.messages:
        ui.console.print(f"[dim]resumed session {runner.session.id}: {runner.session.title} "
                         f"({runner.session.status})[/]")
    ui.console.print("[dim]/help for commands · Ctrl+D to exit[/]\n")
    read = _make_reader(cfg)
    if initial:
        runner.run(initial)
    while True:
        try:
            line = read()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if line is None:
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            r = slash(cfg, runner, line)
            if r == "exit":
                break
            if isinstance(r, str) and r:
                runner.run(r)
            elif r is None and line.split()[0] in ("/continue", "/resume") and runner.session.messages:
                pass
            continue
        runner.run(line)
    ui.console.print("[dim]bye. resume later with: opus -c[/]")
    return 0


def _make_reader(cfg):
    hist = cfg.home / "history"
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.completion import WordCompleter
        if not sys.stdin.isatty():
            raise ImportError
        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        cmds = [w for w in re.findall(r"^\s+(/\w+)", SLASH_HELP, re.M)] + ["/remember", "/resume", "/setup",
                                                                             "/thinking", "/projects", "/new"]
        ps = PromptSession(history=FileHistory(str(hist)), key_bindings=kb, multiline=False,
                           completer=WordCompleter(sorted(set(cmds)), sentence=True), complete_while_typing=False,
                           enable_history_search=True)

        def read() -> str:
            text = ps.prompt([("class:prompt", "> ")])
            while text.endswith("\\"):
                text = text[:-1] + "\n" + ps.prompt("… ")
            return text
        return read
    except Exception:  # noqa: BLE001
        def read_plain() -> str:
            text = input("> ")
            while text.endswith("\\"):
                text = text[:-1] + "\n" + input("… ")
            return text
        return read_plain


def slash(cfg, runner: Runner, line: str):
    ui = runner.ui
    agent = runner.agent
    s = runner.session
    parts = line.split(maxsplit=1)
    cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
    if cmd in ("/exit", "/quit", "/q"):
        return "exit"
    if cmd == "/help":
        ui.console.print(SLASH_HELP)
    elif cmd in ("/usage", "/cost"):
        ui.console.print("session: " + usage_line(s.usage, cfg))
        cmd_usage(cfg, None)
    elif cmd == "/status":
        ctxk = s.last_prompt_tokens
        win = cfg.get("context_window")
        ui.console.print(f"session {s.id} [{s.status}] turn {s.turn} · compactions {s.compactions}\n"
                         f"cwd {agent.ctx.cwd} · branch {git_out(['rev-parse', '--abbrev-ref', 'HEAD'], agent.ctx.cwd) or '-'}\n"
                         f"context ≈{fmt_tokens(ctxk)} / {fmt_tokens(win)} ({ctxk / win:.0%}) · model {cfg['model']} · "
                         f"mode {cfg.get('permission_mode')}\n{usage_line(s.usage, cfg)}", markup=False)
        if s.todos:
            ui.show_todos(s.todos)
    elif cmd == "/todos":
        ui.show_todos(s.todos) if s.todos else ui.info("no plan")
    elif cmd == "/compact":
        from . import context as ctxm
        agent.refresh_system()
        s.repair()
        if not s.messages:
            ui.info("nothing to compact")
        else:
            ui.info("compacting…")
            summary = ctxm.compact(agent)
            agent.save()
            ui.info(f"done; summary {len(summary)} chars")
    elif cmd in ("/clear", "/new"):
        runner.new_session()
        ui.info(f"new session {runner.session.id}")
    elif cmd == "/continue":
        if not s.messages:
            ui.info("nothing to continue")
            return None
        runner.run(arg or None)
    elif cmd == "/sessions":
        cmd_sessions(cfg, agent.memory.key)
    elif cmd == "/resume":
        if not arg:
            cmd_sessions(cfg, agent.memory.key)
            return None
        try:
            runner.load(arg)
            ui.info(f"loaded {runner.session.id}: {runner.session.title} [{runner.session.status}] — /continue to resume")
        except FileNotFoundError as e:
            ui.error(str(e))
    elif cmd == "/mode":
        m = {"plan": "readonly", "readonly": "readonly", "auto": "auto", "ask": "ask"}.get(arg)
        if not m:
            ui.info(f"mode: {cfg.get('permission_mode')} (use auto|ask|plan)")
        else:
            cfg.set("permission_mode", m)
            ui.info(f"mode → {m}")
    elif cmd == "/model":
        if arg:
            cfg.set("model", arg)
        ui.info(f"model: {cfg['model']}")
    elif cmd == "/thinking":
        if arg in ("adaptive", "enabled", "off"):
            cfg.set("thinking", arg)
            agent.llm.thinking_mode = arg
        ui.info(f"thinking: {agent.llm.thinking_mode}")
    elif cmd == "/verbose":
        ui.verbose = not ui.verbose
        ui.info(f"verbose {'on' if ui.verbose else 'off'}")
    elif cmd == "/undo":
        n = int(arg) if arg.isdigit() else 1
        files = agent.ctx.checkpoints.undo(n)
        agent.ctx.read_files.clear()
        ui.info(("restored: " + ", ".join(files)) if files else "nothing to undo")
        if files:
            s.messages.append({"role": "user", "content": [{"type": "text", "text":
                               f"[note] The user reverted your file changes of the last {n} turn(s): "
                               f"{', '.join(files)[:2000]}"}]})
            s.messages.append({"role": "assistant", "content": [{"type": "text", "text": "Understood."}]})
            agent.save()
    elif cmd == "/diff":
        ui.console.print(git_out(["diff", "--stat"], agent.ctx.cwd) or "(no changes)", markup=False)
    elif cmd == "/cd":
        p = Path(os.path.expanduser(arg or "~")).resolve()
        if not p.is_dir():
            ui.error(f"no such dir {p}")
        else:
            os.chdir(p)
            runner.new_session(p)
            ui.info(f"project → {p} (new session)")
    elif cmd == "/projects":
        cmd_projects(cfg)
    elif cmd == "/clone":
        p = do_clone(cfg, arg)
        if p:
            os.chdir(p)
            runner.new_session(p)
            ui.info(f"project → {p}")
    elif cmd == "/memory":
        scope = "global" if arg.startswith("g") else "project"
        ui.console.print(agent.memory.read(scope) or f"({scope} memory empty)", markup=False)
        ui.console.print(f"[dim]{agent.memory.path(scope)}[/]")
    elif cmd == "/remember":
        if arg:
            ui.info(agent.memory.add("project", redact(arg)))
    elif cmd == "/setup":
        from .setup_wizard import setup_api
        setup_api(cfg, force=True)
        agent.llm.api_key = cfg.secret("api_key")
    elif cmd == "/init":
        return TEMPLATES["init"]
    elif cmd == "/issue":
        if not arg.lstrip("#").isdigit():
            ui.error("usage: /issue <number>")
            return None
        return TEMPLATES["issue"].format(arg=arg.lstrip("#"))
    elif cmd == "/pr":
        return TEMPLATES["pr"] + (f"\nExtra instructions: {arg}" if arg else "")
    elif cmd == "/review":
        target = f"PR #{arg.lstrip('#')} (use github pr_get and pr_diff)" if arg else \
            "the current changes (git diff against the default branch, plus uncommitted changes)"
        return TEMPLATES["review"].format(target=target)
    elif cmd == "/fixci":
        return TEMPLATES["fixci"]
    else:
        ui.error(f"unknown command {cmd}; /help")
    return None


# ====================================================================== subcommands
def cmd_sessions(cfg, project_key: str | None = None) -> None:
    from .session import Session
    rows = Session.list(cfg.home, project_key, 25) or Session.list(cfg.home, None, 25)
    if not rows:
        print("no sessions")
        return
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(r["updated"] or 0))
        tok = (r.get("usage") or {}).get("output_tokens", 0)
        print(f"{r['id']}  {when}  [{r['status']:<11}] t{r['turn']:<3} out {fmt_tokens(tok):>6}  "
              f"{(r['title'] or '')[:60]}  ({r['cwd']})")


def cmd_usage(cfg, days: int | None = 14) -> None:
    from .usage import load
    data = load(cfg.home)
    if not data:
        print("no usage recorded yet")
        return
    keys = sorted(data)[-(days or 7):]
    tot: dict = {}
    for k in keys:
        d = data[k]
        print(f"{k}: {usage_line(d, cfg)}")
        for kk, v in d.items():
            tot[kk] = tot.get(kk, 0) + v
    print(f"total: {usage_line(tot, cfg)}")


def cmd_projects(cfg) -> None:
    root = Path(os.path.expanduser(cfg.get("projects_dir", "~/projects")))
    if not root.is_dir():
        print(f"{root} does not exist (clone something: opus clone owner/repo)")
        return
    for p in sorted(root.iterdir()):
        if p.is_dir():
            br = git_out(["rev-parse", "--abbrev-ref", "HEAD"], p)
            print(f"  {p.name:<30} {br}")


def do_clone(cfg, spec: str) -> Path | None:
    spec = spec.strip()
    if not spec:
        print("usage: clone owner/repo | URL [dir]")
        return None
    parts = spec.split()
    src = parts[0]
    if re.fullmatch(r"[\w.-]+/[\w.-]+", src):
        url = f"https://github.com/{src}.git"
    else:
        url = src
    name = re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1])
    root = Path(os.path.expanduser(cfg.get("projects_dir", "~/projects")))
    root.mkdir(parents=True, exist_ok=True)
    dest = Path(os.path.expanduser(parts[1])) if len(parts) > 1 else root / name
    from .tools.gitauth import git_env
    if dest.exists() and (dest / ".git").exists():
        print(f"{dest} exists → git fetch --all --prune && pull")
        subprocess.run(["git", "fetch", "--all", "--prune"], cwd=dest, env=git_env(cfg))
        subprocess.run(["git", "pull", "--ff-only"], cwd=dest, env=git_env(cfg))
        return dest
    r = subprocess.run(["git", "clone", url, str(dest)], env=git_env(cfg))
    if r.returncode != 0:
        print("clone failed (private repo? run `opus github login`)")
        return None
    return dest


def cmd_github(cfg, rest: list[str]) -> int:
    sub = rest[0] if rest else "status"
    from .tools.base import ToolError
    from .tools.github import whoami
    if sub == "login":
        from .setup_wizard import setup_github
        setup_github(cfg)
    elif sub == "logout":
        cfg.set_secret("github_token", "")
        print("GitHub token removed")
    elif sub == "status":
        if not cfg.secret("github_token"):
            print("not connected (opus github login)")
            return 1
        try:
            print(f"connected as {whoami(cfg)}")
        except ToolError as e:
            print(e)
            return 1
    else:
        print("usage: opus github login|logout|status")
    return 0


def cmd_config(cfg, rest: list[str]) -> int:
    import json
    if not rest:
        print(json.dumps(cfg.public_view(), indent=2, ensure_ascii=False))
        print(f"\nfile: {cfg.path}")
        return 0
    if len(rest) == 1:
        print(cfg.public_view().get(rest[0], "(unset)"))
        return 0
    key, val = rest[0], " ".join(rest[1:])
    if key in ("api_key", "github_token", "telegram_bot_token"):
        print("Use `opus setup`, `opus github login` or `opus telegram setup` for secrets.")
        return 1
    cfg.set(key, val)
    print(f"{key} = {cfg.get(key)!r}")
    return 0


def cmd_memory(cfg, rest: list[str]) -> int:
    from .memory import Memory
    m = Memory(cfg.home, Path.cwd())
    sub = rest[0] if rest else "show"
    scope = "global" if len(rest) > 1 and rest[1].startswith("g") else "project"
    if sub == "show":
        for sc in ("global", "project"):
            print(f"== {sc} ({m.path(sc)}) ==\n{m.read(sc) or '(empty)'}\n")
        j = m.journal_entries()
        if j:
            print("== recent tasks ==\n" + "\n\n".join(j[-5:]))
    elif sub == "edit":
        editor = os.environ.get("EDITOR") or ("nano" if subprocess.run(["which", "nano"], capture_output=True).returncode == 0 else "vi")
        p = m.path(scope)
        p.touch()
        subprocess.run([editor, str(p)])
    elif sub == "clear":
        m.path(scope).unlink(missing_ok=True)
        if scope == "project":
            m.journal_path.unlink(missing_ok=True)
        print(f"{scope} memory cleared")
    else:
        print("usage: opus memory show | edit [global] | clear [global]")
    return 0


def cmd_telegram(cfg, rest: list[str]) -> int:
    from .notify import telegram_test
    sub = rest[0] if rest else "setup"
    if sub == "setup":
        from .setup_wizard import setup_telegram
        setup_telegram(cfg)
    elif sub == "test":
        print(telegram_test(cfg))
    elif sub == "off":
        cfg.set("telegram_enabled", False)
        print("telegram notifications off")
    elif sub == "on":
        cfg.set("telegram_enabled", True)
        print("telegram notifications on")
    return 0


def cmd_doctor(cfg) -> int:
    import shutil
    ok = True
    print(f"opus-agent {__version__}  python {sys.version.split()[0]}  home {cfg.home}")
    for t, need in (("git", True), ("rg", False), ("gh", False), ("termux-notification", False),
                    ("termux-wake-lock", False), ("bash", True)):
        found = shutil.which(t)
        print(f"  {'✔' if found else ('✖' if need else '·')} {t}{'' if found else ('  (pkg install ' + {'rg': 'ripgrep', 'termux-notification': 'termux-api', 'termux-wake-lock': 'termux-api'}.get(t, t) + ')')}")
        ok &= bool(found) or not need
    print(f"  {'✔' if cfg.secret('api_key') else '✖'} API key  ({cfg.get('base_url')}, {cfg.get('model')})")
    print(f"  {'✔' if cfg.secret('github_token') else '·'} GitHub token")
    print(f"  {'✔' if cfg.get('telegram_enabled') else '·'} Telegram")
    if cfg.secret("api_key"):
        from .setup_wizard import check_api
        good, msg = check_api(cfg, cfg.secret("api_key"))
        print(f"  {'✔' if good else '✖'} API check: {msg}")
        ok &= good
    return 0 if ok else 1


def cmd_status(cfg) -> int:
    from .session import Session
    pid_file = cfg.home / "detached.pid"
    if pid_file.exists():
        pid, sid, logf = (pid_file.read_text().split("\n") + ["", "", ""])[:3]
        alive = False
        try:
            os.kill(int(pid), 0)
            alive = True
        except (OSError, ValueError):
            pass
        print(f"detached run pid {pid}: {'RUNNING' if alive else 'finished'} · log {logf}")
    s = Session.latest(cfg.home)
    if s:
        print(f"latest session {s.id} [{s.status}] {s.title}\n  cwd {s.cwd}\n  {usage_line(s.usage, cfg)}")
        for t in s.todos:
            mark = {"completed": "✔", "in_progress": "▶"}.get(t["status"], "☐")
            print(f"   {mark} {t['content']}")
        if s.meta.get("pr_urls"):
            print("  PR: " + ", ".join(s.meta["pr_urls"]))
    return 0


def detach(cfg, args) -> int:
    logs = cfg.home / "logs"
    logf = logs / f"run-{time.strftime('%Y%m%d-%H%M%S')}.log"
    argv = [sys.executable, "-m", "opus_agent", "-p", "--yes"]
    if args.continue_:
        argv.append("-c")
    if args.resume:
        argv += ["-r", args.resume]
    if args.prompt:
        argv.append(" ".join(args.prompt))
    with open(logf, "wb") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=os.getcwd())
    (cfg.home / "detached.pid").write_text(f"{p.pid}\n\n{logf}")
    print(f"started in background (pid {p.pid}). Log: {logf}\n  opus status · tail -f {logf}")
    return 0


# ====================================================================== main
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = Config()
    setup_logging(cfg.home / "logs", debug=bool(os.environ.get("OPUS_DEBUG")))
    for sec in cfg.all_secrets():
        register_secret(sec)

    if argv and argv[0] in SUBCOMMANDS:
        sub, rest = argv[0], argv[1:]
        if sub in ("help",):
            print(__doc__)
            return 0
        if sub == "version":
            print(__version__)
            return 0
        if sub == "setup":
            from .setup_wizard import first_run, setup_api
            if "--force" in rest or cfg.secret("api_key"):
                setup_api(cfg, force=True)
                return 0
            return 0 if first_run(cfg) else 1
        if sub == "config":
            return cmd_config(cfg, rest)
        if sub == "github":
            return cmd_github(cfg, rest)
        if sub == "clone":
            p = do_clone(cfg, " ".join(rest))
            if p:
                print(f"cloned → {p}\n  cd {p} && opus")
            return 0 if p else 1
        if sub == "sessions":
            cmd_sessions(cfg)
            return 0
        if sub == "status":
            return cmd_status(cfg)
        if sub == "usage":
            cmd_usage(cfg, int(rest[0]) if rest and rest[0].isdigit() else 14)
            return 0
        if sub == "memory":
            return cmd_memory(cfg, rest)
        if sub == "telegram":
            return cmd_telegram(cfg, rest)
        if sub == "doctor":
            return cmd_doctor(cfg)
        if sub == "projects":
            cmd_projects(cfg)
            return 0
        if sub == "logs":
            os.execvp("tail", ["tail", "-n", "100", str(cfg.home / "logs" / "agent.log")])

    ap = argparse.ArgumentParser(prog="opus", description="Autonomous coding agent for Termux (Claude Code-like).",
                                 epilog="subcommands: " + ", ".join(sorted(SUBCOMMANDS)))
    ap.add_argument("prompt", nargs="*", help="task to run")
    ap.add_argument("-p", "--print", dest="headless", action="store_true", help="headless: run task, print report, exit")
    ap.add_argument("-c", "--continue", dest="continue_", action="store_true", help="continue latest session")
    ap.add_argument("-r", "--resume", help="resume session id")
    ap.add_argument("-C", "--cd", help="project directory")
    ap.add_argument("-m", "--model", help="override model")
    ap.add_argument("--mode", choices=["auto", "ask", "plan"], help="permission mode for this run")
    ap.add_argument("-y", "--yes", action="store_true", help="never ask the user (fully autonomous)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true", help="headless: print only the final report")
    ap.add_argument("--detach", action="store_true", help="run headless in background")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    if args.cd:
        os.chdir(os.path.expanduser(args.cd))
    if args.model:
        cfg.data["model"] = args.model
    if args.mode:
        cfg.data["permission_mode"] = {"plan": "readonly"}.get(args.mode, args.mode)
    if args.detach:
        if not cfg.secret("api_key"):
            print("configure the API key first: opus setup")
            return 1
        return detach(cfg, args)

    interactive = sys.stdin.isatty() and not args.yes
    if not ensure_api_key(cfg, interactive=sys.stdin.isatty()):
        return 1
    ui = make_ui(cfg, args)

    from .session import Session
    from .memory import project_key, project_root
    session = None
    if args.resume:
        try:
            session = Session.load(cfg.home, args.resume)
        except FileNotFoundError as e:
            print(e)
            return 1
    elif args.continue_:
        session = Session.latest(cfg.home, project_key(project_root(Path.cwd()))) or Session.latest(cfg.home)
        if session is None:
            print("no session to continue")
    if session is not None and session.cwd and Path(session.cwd).is_dir():
        os.chdir(session.cwd)

    runner = Runner(cfg, ui, session, interactive=interactive and not args.headless)
    prompt = " ".join(args.prompt).strip() or None
    if prompt == "-":
        prompt = sys.stdin.read()

    if args.headless:
        if prompt is None and not (session and session.messages):
            if not sys.stdin.isatty():
                prompt = sys.stdin.read().strip()
            if not prompt:
                print("no task given", file=sys.stderr)
                return 2
        text = runner.run(prompt)
        if args.quiet and text:
            print(text)
        return 0 if runner.session.status == "done" else 1

    if session is not None and session.status in ("running", "interrupted", "error") and not prompt:
        ui.info(f"session {session.id} was {session.status}. Type /continue to resume it.")
    return repl(cfg, args, runner, prompt)
