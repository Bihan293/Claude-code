import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture(autouse=True)
def opus_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPUS_HOME", str(tmp_path / "home"))
    for k in ("ANTHROPIC_API_KEY", "TOOKEN_API_KEY", "OPUS_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TOOKEN_API_KEY", "sk-test-key-1234567890abcdef")
    yield tmp_path / "home"


def sse(events):
    out = []
    for e in events:
        out.append(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n")
    return "".join(out).encode()


def message_events(blocks, stop_reason="end_turn", usage=None):
    ev = [{"type": "message_start", "message": {"id": "m", "model": "claude-opus-5-5",
                                                  "usage": {"input_tokens": 100, "cache_read_input_tokens": 50,
                                                            "cache_creation_input_tokens": 10, "output_tokens": 1}}}]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}})
        elif b["type"] == "thinking":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": b["thinking"]}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": "sig"}})
        elif b["type"] == "tool_use":
            ev.append({"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
            js = json.dumps(b["input"])
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": js[:5]}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": js[5:]}})
        ev.append({"type": "content_block_stop", "index": i})
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage or {"output_tokens": 20}})
    ev.append({"type": "message_stop"})
    return ev


def openai_sse(text="", tool_calls=None, finish="stop", usage=None):
    """OpenAI chat.completions streaming body. tool_calls: [(id, name, args_dict)]."""
    chunks = []
    if text:
        for part in (text[:3], text[3:]):
            if part:
                chunks.append({"model": "gpt-6.1-sol", "choices": [{"index": 0, "delta": {"content": part}}]})
    for i, (cid, name, args) in enumerate(tool_calls or []):
        js = json.dumps(args)
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "id": cid, "type": "function", "function": {"name": name, "arguments": js[:4]}}]}}]})
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "function": {"arguments": js[4:]}}]}}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else finish}]})
    chunks.append({"choices": [], "usage": usage or {"prompt_tokens": 120, "completion_tokens": 15,
                                                     "prompt_tokens_details": {"cached_tokens": 100}}})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


class MockAnthropic:
    """Scripted Anthropic Messages endpoint. `script` is a list of callables(request_json) -> (status, events|body)."""

    def __init__(self):
        self.requests = []
        self.script = []
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n))
                mock.requests.append({"headers": dict(self.headers), "body": body, "path": self.path})
                step = mock.script.pop(0) if mock.script else (lambda b: (200, message_events([{"type": "text", "text": "done"}])))
                status, payload = step(body)
                if status != 200:
                    data = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if isinstance(payload, (bytes, str)):   # raw body (e.g. OpenAI SSE)
                    data = payload.encode() if isinstance(payload, str) else payload
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream" if body.get("stream") else "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if body.get("stream"):
                    data = sse(payload)
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    data = json.dumps({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                       "usage": {"input_tokens": 5, "output_tokens": 1}}).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def mock_api():
    m = MockAnthropic()
    yield m
    m.close()


@pytest.fixture
def cfg(mock_api, opus_home):
    from opus_agent.config import Config
    c = Config()
    c.data.update(base_url=mock_api.url, max_retries=2, wake_lock=False, termux_notifications=False,
                  telegram_enabled=False)
    return c
