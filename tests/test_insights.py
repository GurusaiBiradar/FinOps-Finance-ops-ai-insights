"""Model fallback chain and pre-generated fallback. requests.post is replaced
with a fake, so these tests never hit the network or need a real API key."""

import json

import pytest
import requests

from src import answers, insights

OK_PAYLOAD = {"candidates": [{"content": {"parts": [{"text": " Narrated summary. "}]}}]}
FACTS = {"question": "Any budget overruns?", "total_spend": 100.0}


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("raise_for_status reached for a handled status")


@pytest.fixture
def fake_post(monkeypatch):
    """Queue up responses (or exceptions); returns the list of URLs called."""
    calls = []

    def install(*outcomes):
        queue = list(outcomes)

        def post(url, **kwargs):
            calls.append(url)
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(insights.requests, "post", post)
        return calls

    return install


def model_in(url):
    return url.split("/models/")[1].split(":")[0]


# --- Live model chain ------------------------------------------------------

def test_falls_back_to_next_model_when_first_is_busy(fake_post):
    calls = fake_post(FakeResponse(503), FakeResponse(200, OK_PAYLOAD))
    text, model = insights.generate_insight_with_model(FACTS, api_key="test")
    assert text == "Narrated summary."
    assert model == insights.GEMINI_MODELS[1]
    assert [model_in(u) for u in calls] == insights.GEMINI_MODELS[:2]


def test_timeouts_and_retired_models_also_fall_through(fake_post):
    fake_post(requests.Timeout(), FakeResponse(404), FakeResponse(200, OK_PAYLOAD))
    _, model = insights.generate_insight_with_model(FACTS, api_key="test")
    assert model == insights.GEMINI_MODELS[2]


def test_truncated_answer_is_never_returned(fake_post):
    truncated = {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "Yes, there are"}]}}]}
    fake_post(FakeResponse(200, truncated), FakeResponse(200, OK_PAYLOAD))
    text, model = insights.generate_insight_with_model(FACTS, api_key="test")
    assert text == "Narrated summary."
    assert model == insights.GEMINI_MODELS[1]


def test_all_models_busy_raises_gemini_unavailable(fake_post):
    calls = fake_post(*[FakeResponse(503)] * len(insights.GEMINI_MODELS))
    with pytest.raises(insights.GeminiUnavailable, match="high demand"):
        insights.generate_insight_with_model(FACTS, api_key="test")
    assert len(calls) == len(insights.GEMINI_MODELS)


def test_invalid_key_stops_immediately_without_trying_other_models(fake_post):
    calls = fake_post(FakeResponse(400))
    with pytest.raises(RuntimeError, match="API_KEY"):
        insights.generate_insight_with_model(FACTS, api_key="test")
    assert len(calls) == 1


# --- Pre-generated fallback ---------------------------------------------------

@pytest.fixture
def saved_answer(tmp_path, monkeypatch):
    path = tmp_path / "answers.json"
    entry = {"question": FACTS["question"], "text": "Saved answer.", "model": "m", "generated_on": "2026-09-28"}
    path.write_text(json.dumps({answers.facts_fingerprint(FACTS): entry}), encoding="utf-8")
    monkeypatch.setattr(answers, "PREGENERATED_PATH", path)


def test_live_answer_is_preferred_over_saved_one(fake_post, saved_answer, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    fake_post(FakeResponse(200, OK_PAYLOAD))
    result = answers.answer(FACTS)
    assert result["source"] == "live"
    assert result["text"] == "Narrated summary."


def test_saved_answer_used_when_every_model_fails(fake_post, saved_answer, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    fake_post(*[FakeResponse(503)] * len(insights.GEMINI_MODELS))
    result = answers.answer(FACTS)
    assert result["source"] == "pregenerated"
    assert result["text"] == "Saved answer."


def test_saved_answer_ignored_when_facts_changed(fake_post, saved_answer, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    fake_post(*[FakeResponse(503)] * len(insights.GEMINI_MODELS))
    with pytest.raises(insights.GeminiUnavailable):
        answers.answer({**FACTS, "total_spend": 999.0})
