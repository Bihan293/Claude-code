import json
from conftest import message_events

from opus_agent.agent import Agent
from opus_agent.session import Session


def tool_step(*uses, text=""):
    blocks = ([{"type": "thinking", "thinking": "hmm"}] if True else []) + \
        ([{"type": "text", "text": text}] if text else []) + \
        [{"type": "tool_use", "id": f"tu_{i}_{u[0]}", "name": u[0], "input": u[1]} for i, u in enumerate(uses)]
    return lambda body: (200, message_events(blocks, "tool_use"))


def text_step(t):
    return lambda body: (200, message_events([{"type": "text", "text": t}]))


def test_full_loop_edits_files_and_runs_shell(cfg, mock_api, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def f():\n    return 1\n")
    mock_api.script = [
        tool_step(("read_file", {"path": "a.py"}), ("grep", {"pattern": "return"})),
        tool_step(("edit_file", {"path": "a.py", "old_string": "return 1", "new_string": "return 2"})),
        tool_step(("bash", {"command": "python3 -c 'import a; print(a.f())' && cd .."})),
        tool_step(("todo_write", {"todos": [{"content": "x", "status": "completed"}]})),
        text_step("## Summary\nAll done."),
    ]
    s = Session(cwd=str(proj))
    a = Agent(cfg, s, None, interactive=False)
    out = a.run("change f to return 2")
    assert "All done" in out
    assert (proj / "a.py").read_text() == "def f():\n    return 2\n"
    assert s.status == "done"
    assert s.todos == [{"content": "x", "status": "completed"}]
    # tool results were sent back properly paired
    last = mock_api.requests[3]["body"]["messages"]
    res = [b for b in last[-1]["content"] if b["type"] == "tool_result"]
    assert "2" in json.dumps(res)
    # bash cwd tracked
    assert a.ctx.cwd == tmp_path
    # usage accumulated, prompt caching markers present, auth header present
    assert s.usage["output_tokens"] > 0
    req = mock_api.requests[1]
    assert req["headers"]["x-api-key"] == "sk-test-key-1234567890abcdef"
    body = req["body"]
    assert body["tools"][-1]["cache_control"]["type"] == "ephemeral"
    assert body["system"][-1].get("cache_control")
    assert body["thinking"] == {"type": "adaptive"}
    # session persisted and loadable
    s2 = Session.load(cfg.home, s.id)
    assert s2.status == "done" and len(s2.messages) == len(s.messages)
    # undo restores
    assert a.ctx.checkpoints.undo(1)
    assert (proj / "a.py").read_text() == "def f():\n    return 1\n"


def test_thinking_downgrade_and_retry(cfg, mock_api, tmp_path):
    mock_api.script = [
        lambda b: (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "thinking.type: adaptive not supported"}}),
        lambda b: (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        text_step("ok"),
    ]
    cfg.data["max_retries"] = 3
    import opus_agent.llm as llm
    llm.random.random = lambda: 0
    orig = llm._sleep_interruptible
    llm._sleep_interruptible = lambda s, cb: None
    try:
        a = Agent(cfg, Session(cwd=str(tmp_path)), None, interactive=False)
        assert a.run("hi") == "ok"
    finally:
        llm._sleep_interruptible = orig
    assert mock_api.requests[2]["body"]["thinking"]["type"] == "enabled"


def test_resume_after_crash_repairs_tool_pairs(cfg, mock_api, tmp_path):
    s = Session(cwd=str(tmp_path))
    s.messages = [
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]},
    ]
    s.status = "running"
    s.save(cfg.home)
    mock_api.script = [text_step("resumed")]
    s2 = Session.load(cfg.home, s.id)
    a = Agent(cfg, s2, None, interactive=False)
    assert a.run(None) == "resumed"
    sent = mock_api.requests[0]["body"]["messages"]
    assert sent[2]["content"][0]["type"] == "tool_result" and sent[2]["content"][0]["tool_use_id"] == "t1"


def test_max_tokens_continuation(cfg, mock_api, tmp_path):
    mock_api.script = [lambda b: (200, message_events([{"type": "text", "text": "part1"}], "max_tokens")),
                       text_step("part2")]
    a = Agent(cfg, Session(cwd=str(tmp_path)), None, interactive=False)
    assert a.run("long") == "part2"
    assert "cut off" in json.dumps(mock_api.requests[1]["body"]["messages"][-1])


def test_compaction(cfg, mock_api, tmp_path):
    cfg.data["context_window"] = 20000
    cfg.data["tool_output_limit"] = 200000
    big = ("x" * 99 + "\n") * 600
    (tmp_path / "big.txt").write_text(big)
    mock_api.script = [
        tool_step(("read_file", {"path": "big.txt"})),
        text_step("SUMMARY OF WORK"),   # compaction call
        text_step("final"),
    ]
    s = Session(cwd=str(tmp_path))
    a = Agent(cfg, s, None, interactive=False)
    assert a.run("read") == "final"
    assert s.compactions == 1
    assert "SUMMARY OF WORK" in json.dumps(mock_api.requests[2]["body"]["messages"])


def test_subagent(cfg, mock_api, tmp_path):
    mock_api.script = [
        tool_step(("task", {"prompt": "explore", "description": "explore"})),
        text_step("sub report"),        # sub-agent
        text_step("main done"),
    ]
    a = Agent(cfg, Session(cwd=str(tmp_path)), None, interactive=False)
    assert a.run("go") == "main done"
    sub_tools = {t["name"] for t in mock_api.requests[1]["body"]["tools"]}
    assert "task" not in sub_tools and "write_file" not in sub_tools and "bash" not in sub_tools
    assert "sub report" in json.dumps(mock_api.requests[2]["body"]["messages"])


def test_push_to_main_blocked(cfg, tmp_path):
    from opus_agent.tools import ToolContext, run_tool
    ctx = ToolContext(cfg=cfg, cwd=tmp_path)
    out, err = run_tool(ctx, "bash", {"command": "git push origin main"})
    assert err and "disabled" in out


def test_git_env_has_no_llm_key(cfg):
    from opus_agent.tools.gitauth import git_env
    env = git_env(cfg)
    assert "TOOKEN_API_KEY" not in env
    assert all("sk-test-key" not in v for v in env.values())
