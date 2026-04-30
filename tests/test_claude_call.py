"""Tests for the cache key + JSON parsing layer of claude_call.

Avoids actually invoking Claude — we monkey-patch _invoke_claude to capture
call arguments and to return canned text.
"""


from evalforge import claude_call as cc


def _stub_invoke_factory(responses):
    """Return a stub that yields the next response on each call."""
    it = iter(responses)

    def _stub(model, system, prompt, schema, timeout):
        return next(it)

    return _stub


def test_cache_key_changes_with_inputs():
    a = cc._hash("haiku", "sys", "prompt", None)
    b = cc._hash("haiku", "sys", "prompt", None)
    assert a == b
    assert cc._hash("sonnet", "sys", "prompt", None) != a
    assert cc._hash("haiku", "sys2", "prompt", None) != a
    assert cc._hash("haiku", "sys", "prompt2", None) != a
    assert cc._hash("haiku", "sys", "prompt", "{}") != a


def test_claude_call_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cc, "_invoke_claude", _stub_invoke_factory(["PONG", "SHOULDNT_RUN"]))
    monkeypatch.setattr(cc, "record_call", lambda **_: None)

    r1 = cc.claude_call("ping?", model="haiku", system="be terse")
    r2 = cc.claude_call("ping?", model="haiku", system="be terse")

    assert r1.text == "PONG" and not r1.cached
    assert r2.text == "PONG" and r2.cached  # cache hit, stub not consumed twice


def test_claude_call_json_strips_fences(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path)
    fenced = "```json\n{\"name\": \"ok\"}\n```"
    monkeypatch.setattr(cc, "_invoke_claude", _stub_invoke_factory([fenced]))
    monkeypatch.setattr(cc, "record_call", lambda **_: None)

    out = cc.claude_call_json("hi", model="haiku")
    assert out == {"name": "ok"}


def test_claude_call_retries_then_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cc, "record_call", lambda **_: None)

    calls = {"n": 0}

    def flaky(model, system, prompt, schema, timeout):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "OK"

    monkeypatch.setattr(cc, "_invoke_claude", flaky)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # don't actually wait
    r = cc.claude_call("test", model="haiku", max_retries=3)
    assert r.text == "OK"
    assert calls["n"] == 2
