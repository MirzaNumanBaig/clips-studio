"""The local monthly spending limit (llm/providers/spend.py).

A second layer on top of the provider's own limit: it counts each response's
`usage`, prices it from the config's rates, and refuses a request that could
pass the cap. No test touches the network: llm/providers/http.transport is
faked, and the key is a made-up string in a temp folder.
"""

import json

import pytest

from llm.providers import http, keys, spend
from llm.providers.base import LLMError
from llm.providers.catalog import PROVIDERS
from llm.providers.cloud_backend import CloudBackend

KEY = "sk-ant-" + "a1b2c3d4" * 8
MODEL = "claude-test-model"
RATES = {"anthropic/" + MODEL: {"input": 3.0, "output": 15.0}}


class Response:
    def __init__(self, body, status=200):
        self.status_code, self._body, self.headers, self.text = status, body, {}, str(body)

    def json(self):
        return self._body


def answer(tokens_in=1000, tokens_out=500, usage=True):
    body = {"content": [{"type": "text", "text": "{\"ok\": true}"}], "stop_reason": "end_turn"}
    if usage:
        body["usage"] = {"input_tokens": tokens_in, "output_tokens": tokens_out}
    return Response(body)


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def fake(method, url, **kw):
        calls.append((method, url))
        return fake.next()

    fake.calls, fake.next = calls, lambda: answer()
    monkeypatch.setattr(http, "transport", fake)
    monkeypatch.setattr(spend, "_broken", None)
    return fake


@pytest.fixture
def backend(tmp_path):
    keys.save_key(tmp_path, "anthropic", KEY)
    return CloudBackend(PROVIDERS["anthropic"], MODEL, tmp_path)


def configure(tmp_path, **changes):
    config = {"monthly_cap_usd": 5.0, "default_rate": {"input": 15.0, "output": 75.0}, "rates": RATES}
    config.update(changes)
    (tmp_path / spend.CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")


def ledger(tmp_path):
    return json.loads((tmp_path / spend.LEDGER_NAME).read_text(encoding="utf-8"))


def test_a_request_is_counted_from_its_usage_at_the_models_rate(tmp_path, backend, transport):
    configure(tmp_path)
    backend.generate("hello")
    # 1000 in at $3/M + 500 out at $15/M
    assert ledger(tmp_path)["usd"] == pytest.approx(0.0105)
    assert ledger(tmp_path)["calls"] == 1


def test_the_default_cap_applies_with_no_config_file_and_the_file_is_created(tmp_path, backend, transport):
    backend.generate("hello")
    config = json.loads((tmp_path / spend.CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["monthly_cap_usd"] == spend.DEFAULT_CAP_USD
    assert ledger(tmp_path)["usd"] > 0  # priced at the deliberately high default rate


def test_a_request_that_could_pass_the_cap_is_not_sent(tmp_path, backend, transport):
    configure(tmp_path, monthly_cap_usd=0.001)  # below one request's worst case
    with pytest.raises(LLMError) as stopped:
        backend.generate("hello")
    assert stopped.value.kind == "no_credits"
    assert "monthly AI spending limit" in stopped.value.message and "local safety limit" in stopped.value.message
    assert str(tmp_path / spend.CONFIG_NAME) in stopped.value.message
    assert transport.calls == []


def test_spending_up_to_the_cap_then_stops(tmp_path, backend, transport):
    # Each request really costs $0.0105 but could cost up to ~$0.1235 (8192 output tokens at $15/M).
    configure(tmp_path, monthly_cap_usd=0.14)
    backend.generate("one")
    backend.generate("two")
    with pytest.raises(LLMError) as stopped:
        backend.generate("three")
    assert stopped.value.kind == "no_credits"
    assert len(transport.calls) == 2
    assert ledger(tmp_path)["usd"] == pytest.approx(0.021)


def test_a_response_with_no_usage_is_charged_the_worst_case(tmp_path, backend, transport):
    configure(tmp_path)
    transport.next = lambda: answer(usage=False)
    backend.generate("hello")
    assert ledger(tmp_path)["usd"] > 0.1  # the reserve, not zero


def test_failed_requests_cost_nothing(tmp_path, backend, transport):
    configure(tmp_path)
    transport.next = lambda: Response({"error": {"type": "invalid_request_error", "message": "no"}}, status=400)
    with pytest.raises(LLMError):
        backend.generate("hello")
    assert not (tmp_path / spend.LEDGER_NAME).exists()


def test_null_cap_turns_the_limit_off_and_counts_nothing(tmp_path, backend, transport):
    configure(tmp_path, monthly_cap_usd=None)
    backend.generate("hello")
    assert not (tmp_path / spend.LEDGER_NAME).exists()


@pytest.mark.parametrize("damage", ["{not json", "[]", json.dumps({"monthly_cap_usd": "lots"}),
                                    json.dumps({"monthly_cap_usd": -1}),
                                    json.dumps({"monthly_cap_usd": 5, "default_rate": {"input": 1}})])
def test_an_unreadable_or_invalid_config_blocks_rather_than_unlimits(tmp_path, backend, transport, damage):
    (tmp_path / spend.CONFIG_NAME).write_text(damage, encoding="utf-8")
    with pytest.raises(LLMError) as stopped:
        backend.generate("hello")
    assert stopped.value.kind == "no_credits"
    assert transport.calls == []


def test_an_unreadable_ledger_blocks(tmp_path, backend, transport):
    configure(tmp_path)
    (tmp_path / spend.LEDGER_NAME).write_text("garbage", encoding="utf-8")
    with pytest.raises(LLMError):
        backend.generate("hello")
    assert transport.calls == []


def test_a_new_month_starts_the_count_over(tmp_path, backend, transport):
    configure(tmp_path)
    (tmp_path / spend.LEDGER_NAME).write_text(json.dumps({"month": "2000-01", "usd": 4.99, "calls": 9}))
    backend.generate("hello")
    assert ledger(tmp_path)["usd"] == pytest.approx(0.0105) and ledger(tmp_path)["calls"] == 1


def test_a_paid_request_that_cannot_be_written_down_blocks_the_next(tmp_path, backend, transport, monkeypatch):
    configure(tmp_path)
    real = spend._write

    def failing(path, data):
        if path.name == spend.LEDGER_NAME:
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(spend, "_write", failing)
    backend.generate("hello")  # went out and was paid for; the job is not failed after the fact
    with pytest.raises(LLMError) as stopped:
        backend.generate("again")
    assert stopped.value.kind == "no_credits" and len(transport.calls) == 1


def test_setup_calls_and_unguarded_requests_are_never_limited(tmp_path, transport):
    # Listing models and checking a key run outside CloudBackend's guard.
    configure(tmp_path, monthly_cap_usd=0.0)
    transport.next = lambda: Response({"data": []})
    http.send(PROVIDERS["anthropic"], KEY, "GET", "/models")
    http.send(PROVIDERS["anthropic"], KEY, "POST", "/messages", json_body={"model": MODEL, "max_tokens": 1})
    assert len(transport.calls) == 2


@pytest.mark.parametrize("body, expected", [
    ({"usage": {"input_tokens": 10, "output_tokens": 5}}, (10, 5)),
    ({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}, (7, 3)),
    ({"usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 40}}, (160, 5)),
    ({"usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2, "thoughtsTokenCount": 4}}, (8, 6)),
    ({"usage": "nope"}, None),
    ({}, None),
])
def test_usage_is_read_in_every_providers_wording(body, expected):
    assert spend._usage(body) == expected


def test_status_reports_the_cap_and_what_is_left(tmp_path):
    configure(tmp_path, monthly_cap_usd=2.0)
    report = spend.status(tmp_path)
    assert report["monthly_cap_usd"] == 2.0 and report["remaining_usd"] == 2.0 and report["requests"] == 0
