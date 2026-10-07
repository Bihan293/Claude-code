import os

from opus_agent.config import Config
from opus_agent.context import prune
from opus_agent.security import redact, register_secret
from opus_agent.tools import ToolContext, run_tool


def ctx(tmp_path):
    return ToolContext(cfg=Config(), cwd=tmp_path)


def test_edit_requires_unique_and_read(tmp_path):
    c = ctx(tmp_path)
    p = tmp_path / "f.txt"
    p.write_text("a\na\nb\n")
    out, err = run_tool(c, "write_file", {"path": "f.txt", "content": "z"})
    assert err and "Read it first" in out
    out, err = run_tool(c, "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "c"})
    assert err and "2 times" in out
    out, err = run_tool(c, "edit_file", {"path": "f.txt", "old_string": "a", "new_string": "c", "replace_all": True})
    assert not err and p.read_text() == "c\nc\nb\n"
    out, err = run_tool(c, "multi_edit", {"path": "f.txt", "edits": [
        {"old_string": "b", "new_string": "B"}, {"old_string": "missing", "new_string": "x"}]})
    assert err and p.read_text() == "c\nc\nb\n"  # atomic


def test_stale_file_detection(tmp_path):
    c = ctx(tmp_path)
    p = tmp_path / "s.txt"
    p.write_text("one\n")
    run_tool(c, "read_file", {"path": "s.txt"})
    os.utime(p, (1, 1))
    out, err = run_tool(c, "edit_file", {"path": "s.txt", "old_string": "one", "new_string": "two"})
    assert err and "changed on disk" in out


def test_read_offset_and_grep_glob(tmp_path):
    c = ctx(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("\n".join(f"line{i}" for i in range(1, 101)))
    out, _ = run_tool(c, "read_file", {"path": "src/m.py", "offset": 10, "limit": 2})
    assert "line10" in out and "line12" not in out and "more lines" in out
    out, _ = run_tool(c, "grep", {"pattern": "line5\\d", "mode": "count"})
    assert "10" in out
    out, _ = run_tool(c, "glob", {"pattern": "**/*.py"})
    assert "src/m.py" in out


def test_bash_background_and_timeout(tmp_path):
    c = ctx(tmp_path)
    out, err = run_tool(c, "bash", {"command": "sleep 5", "timeout": 1})
    assert "TIMED OUT" in out
    out, _ = run_tool(c, "bash", {"command": "echo hi; sleep 0.2; echo bye", "run_in_background": True})
    jid = out.split("job ")[1].split()[0]
    out, _ = run_tool(c, "job_output", {"job_id": jid, "wait": 3})
    assert "exited with code 0" in out


def test_redaction():
    register_secret("supersecretvalue123")
    assert "supersecret" not in redact("key=supersecretvalue123")
    assert "ghp_" not in redact("token ghp_abcdefghijklmnopqrstuvwxyz0123")
    assert "user:pw@" not in redact("https://user:pw@github.com/x")


def test_tool_output_is_redacted(tmp_path):
    c = ctx(tmp_path)
    out, _ = run_tool(c, "bash", {"command": "echo $TOOKEN_API_KEY; echo sk-test-key-1234567890abcdef"})
    assert "sk-test-key" not in out


def test_prune_stubs_old_results():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    for i in range(12):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {"path": "x"}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "y" * 5000}]})
    saved = prune(msgs, keep_recent=4)
    assert saved > 30000
    assert msgs[2]["content"][0]["content"].startswith("[pruned")
    assert msgs[-1]["content"][0]["content"] == "y" * 5000


def test_secrets_not_in_config_file(opus_home):
    c = Config()
    c.set_secret("github_token", "ghp_secretsecretsecretsecret12")
    c.set("model", "m")
    assert "ghp_" not in c.path.read_text()
    assert oct(c.cred_path.stat().st_mode)[-3:] == "600"
