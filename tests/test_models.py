import json

from conftest import message_events, openai_sse

from opus_agent.agent import Agent
from opus_agent.llm import to_openai_messages
from opus_agent.models import MODELS, api_for, menu, resolve
from opus_agent.session import Session


def test_catalogue_numbers():
    assert [(m.num, m.id) for m in MODELS] == [
        ("1", "claude-opus-5-5"), ("2", "gpt-6.1-sol"), ("3", "claude-sonnet-5-5")]
    assert resolve("1") == "claude-opus-5-5"
    assert resolve("2") == "gpt-6.1-sol"
    assert resolve("3") == "claude-sonnet-5-5"
    assert resolve("сонет") == "claude-sonnet-5-5" and resolve("GPT") == "gpt-6.1-sol"
    assert resolve("my-custom-model") == "my-custom-model" and resolve("") is None
    assert api_for("gpt-6.1-sol") == "openai" and api_for("claude-sonnet-5-5") == "anthropic"
    assert api_for("gpt-6-luna") == "openai" and api_for("claude-x") == "anthropic"
    m = menu("gpt-6.1-sol")
    assert m.splitlines() == ["Модели:", "  1  Claude Opus 5.5", "  2  GPT-6.1 Sol  ← текущая",
                              "  3  Claude Sonnet 5.5"]


def test_gpt_full_loop_via_chat_completions(cfg, mock_api, tmp_path):
    cfg.data["model"] = "gpt-6.1-sol"
    (tmp_path / "a.txt").write_text("hello\n")
    mock_api.script = [
        lambda b: (200, openai_sse("Reading", [("call_1", "read_file", {"path": "a.txt"})])),
        lambda b: (200, openai_sse("All done.")),
    ]
    s = Session(cwd=str(tmp_path))
    a = Agent(cfg, s, None, interactive=False)
    assert a.run("read a.txt") == "All done."
    r0, r1 = mock_api.requests
    assert r0["path"].endswith("/chat/completions") and r0["body"]["model"] == "gpt-6.1-sol"
    assert r0["body"]["messages"][0]["role"] == "system"
    assert r0["body"]["tools"][0]["type"] == "function"
    assert "thinking" not in r0["body"] and "cache_control" not in json.dumps(r0["body"])
    msgs = r1["body"]["messages"]
    assert msgs[-2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[-1]["role"] == "tool" and "hello" in msgs[-1]["content"]
    assert s.usage["cache_read_input_tokens"] == 200 and s.usage["input_tokens"] == 40


def test_openai_conversion_drops_thinking_and_keeps_pairs():
    hist = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"},
                                          {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "f", "is_error": True},
                                     {"type": "text", "text": "next"}]},
    ]
    out = to_openai_messages(hist)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "user"]
    assert "thinking" not in json.dumps(out) and out[2]["content"].startswith("ERROR")


def test_slash_model_menu_then_number(cfg, mock_api, tmp_path, monkeypatch):
    from opus_agent import cli
    from opus_agent.ui import UI

    monkeypatch.chdir(tmp_path)
    runner = cli.Runner(cfg, UI(cfg), Session(cwd=str(tmp_path)), interactive=False)
    runner.session.messages = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                                          {"type": "text", "text": "a"}]}]
    runner.agent.llm.thinking_mode = "off"      # downgraded for old model ...
    lines = iter(["/model", "3", "/model 2", "/model", "1", "/exit"])
    monkeypatch.setattr(cli, "_make_reader", lambda c: (lambda: next(lines)))
    seen = []
    orig = cli.switch_model
    monkeypatch.setattr(cli, "switch_model", lambda c, r, ch: (orig(c, r, ch), seen.append(c["model"])))
    cli.repl(cfg, None, runner, None)
    assert seen == ["claude-sonnet-5-5", "gpt-6.1-sol", "claude-opus-5-5"]
    assert cfg["model"] == "claude-opus-5-5"
    assert runner.agent.llm.thinking_mode == "adaptive"       # ... reset on switch
    assert "thinking" not in json.dumps(runner.session.messages)
    assert mock_api.requests == []                             # "3"/"1" were not sent to the model


def test_system_prompt_frozen_within_conversation(cfg, mock_api, tmp_path):
    mock_api.script = [lambda b: (200, message_events([{"type": "text", "text": "one"}])),
                       lambda b: (200, message_events([{"type": "text", "text": "two"}]))]
    s = Session(cwd=str(tmp_path))
    a = Agent(cfg, s, None, interactive=False)
    a.run("first")
    (tmp_path / "new_untracked.txt").write_text("x")  # would change env/journal block
    a.memory.add("project", "a new fact")
    a.run("second")
    assert mock_api.requests[0]["body"]["system"] == mock_api.requests[1]["body"]["system"]
