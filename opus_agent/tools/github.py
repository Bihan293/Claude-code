"""GitHub integration over the REST API (works without `gh`; token via env)."""
from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any

import httpx

from ..security import redact
from .base import ToolContext, ToolError, tool

API = "https://api.github.com"


def _headers(ctx_cfg) -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
         "User-Agent": "opus-agent"}
    tok = ctx_cfg.secret("github_token")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def gh_request(cfg, method: str, path: str, *, params: dict | None = None, body: Any = None,
               raw: bool = False, timeout: float = 60) -> Any:
    url = path if path.startswith("http") else API + "/" + path.lstrip("/")
    try:
        r = httpx.request(method, url, headers=_headers(cfg), params=params,
                          json=body if body is not None else None, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as e:
        raise ToolError(f"GitHub request failed: {e}")
    if r.status_code >= 400:
        msg = r.text[:1500]
        try:
            j = r.json()
            msg = j.get("message", "") + (" " + json.dumps(j.get("errors"))[:800] if j.get("errors") else "")
        except ValueError:
            pass
        if r.status_code == 401:
            msg += " (token missing/invalid: run `opus github login`)"
        raise ToolError(f"GitHub {method} {path} -> {r.status_code}: {redact(msg)}")
    if raw:
        return r
    if r.status_code == 204 or not r.content:
        return {}
    try:
        return r.json()
    except ValueError:
        return r.text


def detect_repo(ctx: ToolContext) -> str:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(ctx.cwd), capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        url = ""
    m = re.search(r"github\.com[:/]+([^/]+)/([^/\s]+?)(?:\.git)?/?$", url)
    if not m:
        raise ToolError("Cannot detect GitHub repo from `git remote get-url origin`; pass repo='owner/name'.")
    return f"{m.group(1)}/{m.group(2)}"


def current_branch(ctx: ToolContext) -> str:
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ctx.cwd), capture_output=True,
                          text=True, timeout=10).stdout.strip()


def _fmt_issue(i: dict) -> str:
    labels = ",".join(l["name"] for l in i.get("labels", []))
    kind = "PR" if "pull_request" in i else "Issue"
    return f"#{i['number']} [{i['state']}] {kind}: {i['title']} ({i['user']['login']}){' [' + labels + ']' if labels else ''}"


@tool("github",
      """GitHub operations for the current repo (auto-detected from origin) or `repo`='owner/name'.
Actions:
- repo_info
- issue_list {state?, labels?, limit?} | issue_get {number} (with comments) | issue_create {title, body, labels?}
- issue_comment {number, body} | issue_close {number}
- pr_list {state?} | pr_get {number} (details, files, reviews, comments) | pr_diff {number}
- pr_create {title, body, head?, base?, draft?}  (head defaults to current branch, base to default branch)
- pr_comment {number, body} | pr_merge {number, method?}
- checks {ref? or number?, wait?}  (CI status: check runs + commit statuses + workflow runs.
  wait=seconds (max 900) blocks until all checks finish – use it instead of sleep+poll loops)
- run_logs {run_id}  (failed job logs of a workflow run, tail of each failed step)
- rerun {run_id}
Push your branch with `git push -u origin <branch>` (bash) before pr_create.""",
      {"action": {"type": "string"},
       "repo": {"type": "string"},
       "number": {"type": "integer"},
       "title": {"type": "string"}, "body": {"type": "string"},
       "head": {"type": "string"}, "base": {"type": "string"}, "draft": {"type": "boolean"},
       "state": {"type": "string"}, "labels": {"type": "string"}, "limit": {"type": "integer"},
       "ref": {"type": "string"}, "run_id": {"type": "integer"}, "method": {"type": "string"},
       "wait": {"type": "integer"}},
      ["action"], readonly_ok=True)
def github(ctx: ToolContext, action: str, repo: str = "", **kw: Any) -> str:
    cfg = ctx.cfg
    repo = repo or detect_repo(ctx)
    R = f"repos/{repo}"
    n = kw.get("number")
    write_actions = {"issue_create", "issue_comment", "issue_close", "pr_create", "pr_comment", "pr_merge", "rerun"}
    if action in write_actions and cfg.get("permission_mode") == "readonly":
        raise ToolError("read-only mode: write actions disabled")
    if action in write_actions and not cfg.secret("github_token"):
        raise ToolError("No GitHub token. Ask the user to run `opus github login`.")

    if action == "repo_info":
        j = gh_request(cfg, "GET", R)
        return (f"{j['full_name']} default_branch={j['default_branch']} private={j['private']} "
                f"stars={j['stargazers_count']} open_issues={j['open_issues_count']}\n"
                f"language={j.get('language')} perms={j.get('permissions')}\n{j.get('description') or ''}")
    if action == "issue_list":
        params = {"state": kw.get("state") or "open", "per_page": min(int(kw.get("limit") or 30), 100)}
        if kw.get("labels"):
            params["labels"] = kw["labels"]
        items = gh_request(cfg, "GET", f"{R}/issues", params=params)
        return "\n".join(_fmt_issue(i) for i in items) or "No issues."
    if action == "issue_get":
        i = gh_request(cfg, "GET", f"{R}/issues/{n}")
        cs = gh_request(cfg, "GET", f"{R}/issues/{n}/comments", params={"per_page": 50})
        out = [_fmt_issue(i), i.get("html_url", ""), "", i.get("body") or "(no body)"]
        for c in cs:
            out.append(f"\n--- comment by {c['user']['login']} at {c['created_at']}\n{c['body']}")
        return "\n".join(out)
    if action == "issue_create":
        body = {"title": kw["title"], "body": kw.get("body", "")}
        if kw.get("labels"):
            body["labels"] = [x.strip() for x in kw["labels"].split(",")]
        j = gh_request(cfg, "POST", f"{R}/issues", body=body)
        return f"Created issue #{j['number']}: {j['html_url']}"
    if action == "issue_comment" or action == "pr_comment":
        j = gh_request(cfg, "POST", f"{R}/issues/{n}/comments", body={"body": kw["body"]})
        return f"Commented: {j['html_url']}"
    if action == "issue_close":
        gh_request(cfg, "PATCH", f"{R}/issues/{n}", body={"state": "closed"})
        return f"Closed #{n}"
    if action == "pr_list":
        items = gh_request(cfg, "GET", f"{R}/pulls", params={"state": kw.get("state") or "open", "per_page": 50})
        return "\n".join(f"#{p['number']} [{p['state']}{' draft' if p.get('draft') else ''}] {p['title']} "
                         f"({p['head']['ref']} -> {p['base']['ref']}) {p['html_url']}" for p in items) or "No PRs."
    if action == "pr_get":
        p = gh_request(cfg, "GET", f"{R}/pulls/{n}")
        files = gh_request(cfg, "GET", f"{R}/pulls/{n}/files", params={"per_page": 100})
        reviews = gh_request(cfg, "GET", f"{R}/pulls/{n}/reviews")
        rc = gh_request(cfg, "GET", f"{R}/pulls/{n}/comments", params={"per_page": 100})
        ic = gh_request(cfg, "GET", f"{R}/issues/{n}/comments", params={"per_page": 100})
        out = [f"PR #{n}: {p['title']} [{p['state']}] mergeable={p.get('mergeable')} "
               f"{p['head']['ref']} -> {p['base']['ref']}", p["html_url"], "", p.get("body") or "", "", "Files:"]
        out += [f"  {f['status']} {f['filename']} +{f['additions']} -{f['deletions']}" for f in files]
        for r in reviews:
            out.append(f"\nReview by {r['user']['login']}: {r['state']}\n{r.get('body') or ''}")
        for c in rc:
            out.append(f"\nReview comment {c['user']['login']} on {c['path']}:{c.get('line') or c.get('original_line')}\n"
                       f"{c['body']}")
        for c in ic:
            out.append(f"\nComment {c['user']['login']}: {c['body']}")
        return "\n".join(out)
    if action == "pr_diff":
        r = httpx.get(f"{API}/{R}/pulls/{n}", headers={**_headers(cfg), "Accept": "application/vnd.github.diff"},
                      timeout=60, follow_redirects=True)
        return r.text
    if action == "pr_create":
        head = kw.get("head") or current_branch(ctx)
        base = kw.get("base") or gh_request(cfg, "GET", R)["default_branch"]
        if head == base:
            raise ToolError(f"head == base ({head}). Create a feature branch first.")
        owner = repo.split("/")[0]
        existing = gh_request(cfg, "GET", f"{R}/pulls", params={"head": f"{owner}:{head}", "state": "open"})
        if existing:
            p = existing[0]
            if kw.get("body") or kw.get("title"):
                upd = {k: kw[k] for k in ("title", "body") if kw.get(k)}
                gh_request(cfg, "PATCH", f"{R}/pulls/{p['number']}", body=upd)
            ctx.pr_urls.append(p["html_url"])
            return f"PR already exists (updated): #{p['number']} {p['html_url']}"
        body = {"title": kw["title"], "body": kw.get("body", ""), "head": head, "base": base,
                "draft": bool(kw.get("draft"))}
        p = gh_request(cfg, "POST", f"{R}/pulls", body=body)
        ctx.pr_urls.append(p["html_url"])
        if ctx.session is not None:
            ctx.session.meta.setdefault("pr_urls", []).append(p["html_url"])
        from ..notify import notify
        notify(cfg, f"PR created: {p['title']}\n{p['html_url']}", kind="pr")
        return f"Created PR #{p['number']}: {p['html_url']}"
    if action == "pr_merge":
        j = gh_request(cfg, "PUT", f"{R}/pulls/{n}/merge", body={"merge_method": kw.get("method") or "squash"})
        return f"Merged: {j.get('message')}"
    if action == "checks":
        wait = min(max(int(kw.get("wait") or 0), 0), 900)
        deadline = time.time() + wait
        while True:
            report, pending = _checks(ctx, cfg, R, n, kw.get("ref"))
            if not pending or time.time() >= deadline or ctx.stop_event.is_set():
                if pending and wait:
                    report += f"\n  (still pending after waiting {wait}s)"
                return report
            time.sleep(min(20.0, max(1.0, deadline - time.time())))
    if action == "run_logs":
        rid = kw.get("run_id")
        jobs = gh_request(cfg, "GET", f"{R}/actions/runs/{rid}/jobs", params={"per_page": 50})
        out = []
        for j in jobs.get("jobs", []):
            out.append(f"Job '{j['name']}' {j['status']}/{j.get('conclusion')}")
            failed_steps = [s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"]
            if failed_steps:
                out.append("  failed steps: " + ", ".join(failed_steps))
            if j.get("conclusion") == "failure":
                try:
                    r = gh_request(cfg, "GET", f"{R}/actions/jobs/{j['id']}/logs", raw=True, timeout=120)
                    text = r.text
                    text = re.sub(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", "", text, flags=re.M)
                    # keep around error lines + tail
                    lines = text.splitlines()
                    err_idx = [i for i, l in enumerate(lines) if re.search(r"(?i)error|fail|exception|traceback", l)]
                    keep: set[int] = set(range(max(0, len(lines) - 120), len(lines)))
                    for i in err_idx[:40]:
                        keep.update(range(max(0, i - 3), min(len(lines), i + 6)))
                    snippet = "\n".join(lines[i] for i in sorted(keep))
                    out.append(snippet[-15000:])
                except ToolError as e:
                    out.append(f"  (logs unavailable: {e})")
        return "\n".join(out) or "No jobs."
    if action == "rerun":
        gh_request(cfg, "POST", f"{R}/actions/runs/{kw.get('run_id')}/rerun-failed-jobs")
        return "Re-run requested."
    raise ToolError(f"unknown action {action}")


@tool("github_api",
      "Raw GitHub REST call for anything not covered by `github`. path like 'repos/o/r/labels'.",
      {"method": {"type": "string", "enum": ["GET", "POST", "PATCH", "PUT", "DELETE"]},
       "path": {"type": "string"}, "params": {"type": "object"}, "body": {"type": "object"}},
      ["method", "path"], readonly_ok=False)
def github_api(ctx: ToolContext, method: str, path: str, params: dict | None = None, body: dict | None = None) -> str:
    j = gh_request(ctx.cfg, method.upper(), path, params=params, body=body)
    return json.dumps(j, indent=1, ensure_ascii=False)[:60000]


def whoami(cfg) -> str:
    j = gh_request(cfg, "GET", "user")
    return j.get("login", "?")


def _checks(ctx: ToolContext, cfg, R: str, n: Any, ref: str | None) -> tuple[str, bool]:
    """One CI snapshot. Returns (report, pending). Green items are collapsed to keep
    the result short (it is re-sent with every later model call)."""
    if n and not ref:
        ref = gh_request(cfg, "GET", f"{R}/pulls/{n}")["head"]["sha"]
    if not ref:
        ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ctx.cwd), capture_output=True, text=True,
                             timeout=10).stdout.strip()
    out = [f"Checks for {ref[:12]}:"]
    ok: list[str] = []
    pending = False
    cr = gh_request(cfg, "GET", f"{R}/commits/{ref}/check-runs", params={"per_page": 100})
    for c in cr.get("check_runs", []):
        if c["status"] != "completed":
            pending = True
        if c.get("conclusion") in ("success", "skipped", "neutral"):
            ok.append(c["name"])
            continue
        out.append(f"  {c['name']}: {c['status']}/{c.get('conclusion')} {c.get('html_url', '')}")
        if c.get("conclusion") in ("failure", "timed_out") and c.get("output", {}).get("summary"):
            out.append("    " + (c["output"]["summary"] or "")[:800].replace("\n", "\n    "))
    st = gh_request(cfg, "GET", f"{R}/commits/{ref}/status")
    for x in st.get("statuses", []):
        if x["state"] == "pending":
            pending = True
        if x["state"] == "success":
            ok.append(x["context"])
            continue
        out.append(f"  status {x['context']}: {x['state']} {x.get('description') or ''}")
    runs = gh_request(cfg, "GET", f"{R}/actions/runs", params={"head_sha": ref, "per_page": 20})
    for w in runs.get("workflow_runs", []):
        if w["status"] != "completed":
            pending = True
        if w.get("conclusion") == "success":
            continue
        out.append(f"  workflow run {w['id']} '{w['name']}': {w['status']}/{w.get('conclusion')} {w['html_url']}")
    if ok:
        out.append(f"  passed ({len(ok)}): " + ", ".join(ok[:30]))
    if len(out) == 1:
        out.append("  (no checks yet – CI may not be configured or not started; retry with wait=60)")
        pending = False
    return "\n".join(out), pending
