"""Research runs against thinking models that loop until the token cap.

A Qwen3.x model at the server's default xhigh effort, called at research's
temperature 0.3, can spend the whole report budget inside its reasoning and
stop on ``length`` with an empty ``content``. The non-streaming parsers used
to fall back to ``reasoning_content`` and hand that monologue back as the
report. These cover the truncation guard and the research-only thinking /
effort settings that let an operator avoid the loop.
"""
import asyncio

import httpx

from src import llm_core
from src.research_utils import resolve_research_reasoning


def _choice(content, reasoning=None, finish_reason=None):
    msg = {"content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    choice = {"message": msg}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"choices": [choice]}


def _patch_async(monkeypatch, payload, seen=None):
    class _FakeAsyncClient:
        async def post(self, *a, **kw):
            if seen is not None:
                seen.append(kw.get("json"))
            req = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
            return httpx.Response(200, request=req, json=payload)

    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeAsyncClient())


def test_choice_text_drops_reasoning_truncated_by_length():
    choice = _choice("", "Need maybe maybe include... Stop.", "length")["choices"][0]
    assert llm_core._openai_choice_text(choice, "qwen38-flash") == ""


def test_choice_text_keeps_reasoning_fallback_when_model_finished():
    choice = _choice("", "the whole answer lives here", "stop")["choices"][0]
    assert llm_core._openai_choice_text(choice) == "the whole answer lives here"


def test_choice_text_keeps_content_even_when_truncated():
    choice = _choice("partial report", "thoughts", "length")["choices"][0]
    assert llm_core._openai_choice_text(choice) == "partial report"


def test_sync_llm_call_returns_empty_for_truncated_reasoning(monkeypatch):
    req = httpx.Request("POST", "http://test/v1/chat/completions")
    monkeypatch.setattr(
        llm_core.httpx, "post",
        lambda *a, **kw: httpx.Response(
            200, request=req, json=_choice("", "looping thoughts", "length")
        ),
    )
    result = llm_core.llm_call(
        "http://test/v1", "qwen3-8b", [{"role": "user", "content": "report"}]
    )
    assert result == ""


def test_async_llm_call_returns_empty_for_truncated_reasoning(monkeypatch):
    _patch_async(monkeypatch, _choice("", "looping thoughts", "length"))
    result = asyncio.run(llm_core.llm_call_async(
        "http://test-trunc/v1", "qwen3-8b", [{"role": "user", "content": "report"}]
    ))
    assert result == ""


def test_resolve_research_reasoning_reads_valid_values():
    settings = {"research_thinking_mode": " Off ", "research_reasoning_effort": "LOW"}
    assert resolve_research_reasoning(lambda k, d=None: settings.get(k, d)) == ("off", "low")


def test_resolve_research_reasoning_ignores_unset_and_junk():
    settings = {"research_thinking_mode": "maybe", "research_reasoning_effort": 7}
    assert resolve_research_reasoning(lambda k, d=None: settings.get(k, d)) == (None, None)
    assert resolve_research_reasoning(lambda k, d=None: d) == (None, None)


def test_deep_researcher_forwards_reasoning_controls(monkeypatch):
    from src.deep_research import DeepResearcher

    seen = []
    _patch_async(monkeypatch, _choice("a report", None, "stop"), seen)
    researcher = DeepResearcher(
        llm_endpoint="http://127.0.0.1:8080/v1/chat/completions",
        llm_model="qwen38-flash-openclaw",
        thinking_mode="off",
        reasoning_effort="low",
    )
    out = asyncio.run(researcher._llm([{"role": "user", "content": "go"}]))
    assert out == "a report"
    payload = seen[-1]
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["reasoning_effort"] == "low"


def test_deep_researcher_leaves_defaults_alone(monkeypatch):
    from src.deep_research import DeepResearcher

    seen = []
    _patch_async(monkeypatch, _choice("a report", None, "stop"), seen)
    researcher = DeepResearcher(
        llm_endpoint="http://127.0.0.1:8080/v1/chat/completions",
        llm_model="qwen38-flash-openclaw",
    )
    asyncio.run(researcher._llm([{"role": "user", "content": "go again"}]))
    payload = seen[-1]
    assert "enable_thinking" not in (payload.get("chat_template_kwargs") or {})
    assert "reasoning_effort" not in payload
