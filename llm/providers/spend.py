"""A local monthly spending limit for cloud AI requests (local modification).

This is a SECOND layer, not the limit. It only sees requests this app makes,
and the dollar figure is an ESTIMATE: tokens counted from each response's
`usage` block, priced from the rates in the config file. The real ceiling is
the provider's own spend limit (Anthropic Console, OpenRouter key limit), and
that is the one to rely on.

How it works
- CloudBackend opens a `guard` around each model request. http.send, the one
  door every request goes through, then calls `check` before sending and
  `record` after a success. Outside a guard (listing models, checking a key,
  local Ollama) nothing here runs, so setup is never blocked.
- `check` refuses a request when this month's spend plus the worst case for
  THIS request (its prompt size plus its max output tokens) would pass the cap,
  so a single long answer cannot step over it.
- The tally lives in the data folder, resets each UTC month, and is written
  atomically under a lock. Anything unreadable or unwritable fails CLOSED: the
  request is refused, never allowed unmetered. Only an explicit
  "monthly_cap_usd": null in the config switches the limit off.
- Parallel requests can overshoot by about one request each; the check is not
  a reservation.

Files (in the app's data folder; edit the first, never needed for the others)
    ai_spend_limit.json   the cap and the rates
    ai_spend_ledger.json  this month's estimated spend
    ai_spend.lock         held for milliseconds while the ledger is updated

    python -m llm.providers.spend [data_dir]   shows the cap and this month's use
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.paths import discard
from llm.providers.base import LLMError

CONFIG_NAME = "ai_spend_limit.json"
LEDGER_NAME = "ai_spend_ledger.json"
LOCK_NAME = "ai_spend.lock"

DEFAULT_CAP_USD = 5.0
# A deliberately HIGH placeholder, not any provider's price, so the estimate
# errs toward stopping early. Put the real per-model rates (from the provider's
# pricing page) in "rates" and the estimate becomes accurate.
DEFAULT_RATE = {"input": 15.0, "output": 75.0}  # USD per million tokens
FALLBACK_MAX_OUTPUT = 8192
CHARS_PER_TOKEN = 3.0  # English runs nearer 4, so this over-counts the prompt on purpose
CACHE_WRITE_FACTOR = 1.25

_HELP = (
    "monthly_cap_usd: stop sending AI requests once this month's ESTIMATED spend would pass it "
    "(null turns the limit off). rates: USD per million tokens, keyed \"provider/model\", e.g. "
    "{\"anthropic/<model-id>\": {\"input\": 3.0, \"output\": 15.0}}. Models not listed are priced at "
    "default_rate, which is a deliberately high placeholder. This is a local estimate, not the "
    "provider's bill: set the provider's own spend limit too."
)

_active: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar("clips_spend_scope", default=None)
_thread_lock = threading.Lock()
_broken: str | None = None  # set when a paid request could not be written down; blocks until restart


@dataclass(frozen=True)
class _Scope:
    data_dir: Path
    provider: str
    model: str


@dataclass(frozen=True)
class Pending:
    """What `check` hands to `record` for one request."""

    scope: _Scope
    rate: tuple[float, float]
    reserve_usd: float


@contextlib.contextmanager
def guard(data_dir, provider: str, model: str):
    """Everything sent inside this block is limited and counted."""
    token = _active.set(_Scope(Path(data_dir), provider, model))
    try:
        yield
    finally:
        _active.reset(token)


# ---- the limit, as seen by http.send ---------------------------------------------------


def check(method: str, json_body) -> Pending | None:
    """Refuse the request if it could pass the cap. None when no limit applies."""
    scope = _active.get()
    if scope is None or method != "POST" or not isinstance(json_body, dict):
        return None
    if _broken:
        raise _blocked(f"it could not write down a previous request ({_broken}). Restart the app once the "
                       f"data folder is writable")
    cap, rate = _settings(scope)
    if cap is None:
        return None
    reserve = _cost(_prompt_tokens(json_body), _max_output(json_body), rate)
    ledger = _ledger(scope.data_dir)
    if ledger["usd"] + reserve > cap:
        raise _blocked(f"it would take this month's estimated spend (${ledger['usd']:.2f} so far, plus up to "
                       f"${reserve:.2f} for this request) past your ${cap:.2f} limit. To continue, raise "
                       f"monthly_cap_usd in {scope.data_dir / CONFIG_NAME}, or wait for next month")
    return Pending(scope, rate, reserve)


def record(pending: Pending | None, body) -> None:
    """Add a finished request's cost to this month's tally."""
    global _broken
    if pending is None:
        return
    usage = _usage(body)
    cost = pending.reserve_usd if usage is None else _cost(usage[0], usage[1], pending.rate)
    try:
        with _locked(pending.scope.data_dir):
            ledger = _ledger(pending.scope.data_dir)
            ledger["usd"] = round(ledger["usd"] + cost, 6)
            ledger["calls"] += 1
            _write(pending.scope.data_dir / LEDGER_NAME, ledger)
    except (OSError, LLMError) as e:
        # The request already happened and was paid for. Say nothing to the job,
        # but stop the next one: an unmetered limit is no limit.
        _broken = type(e).__name__
        print(f"[spend limit] could not record ${cost:.4f}: {e}", file=sys.stderr)


def status(data_dir) -> dict:
    """The cap, this month's estimate and what is left. Creates the config if missing."""
    scope = _Scope(Path(data_dir), "", "")
    cap, _rate = _settings(scope)
    ledger = _ledger(scope.data_dir)
    return {"month": ledger["month"], "monthly_cap_usd": cap, "estimated_spend_usd": ledger["usd"],
            "requests": ledger["calls"],
            "remaining_usd": None if cap is None else round(max(cap - ledger["usd"], 0.0), 6),
            "config": str(scope.data_dir / CONFIG_NAME)}


# ---- settings, ledger, lock -------------------------------------------------------------


def _blocked(why: str) -> LLMError:
    return LLMError("no_credits", f"Clips Studio's own monthly AI spending limit stopped this request: {why}. "
                                  "(This is a local safety limit, not your provider's.)")


def _default_config() -> dict:
    return {"_help": _HELP, "monthly_cap_usd": DEFAULT_CAP_USD, "default_rate": dict(DEFAULT_RATE), "rates": {}}


def _settings(scope: _Scope) -> tuple[float | None, tuple[float, float]]:
    path = scope.data_dir / CONFIG_NAME
    if not path.exists():
        try:
            scope.data_dir.mkdir(parents=True, exist_ok=True)
            _write(path, _default_config())
        except OSError:
            pass  # the defaults below still apply; the limit is on even without the file
        config = _default_config()
    else:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as e:
            raise _blocked(f"its settings file {path} can't be read ({e}). Fix or delete it") from e
    try:
        cap = config.get("monthly_cap_usd", DEFAULT_CAP_USD)
        cap = None if cap is None else _positive(cap)
        rates = config.get("rates") if isinstance(config.get("rates"), dict) else {}
        chosen = rates.get(f"{scope.provider}/{scope.model}") or config.get("default_rate") or DEFAULT_RATE
        rate = (_positive(chosen["input"]), _positive(chosen["output"]))
    except (KeyError, TypeError, ValueError) as e:
        raise _blocked(f"a number in {path} is missing or not valid ({e!s})") from e
    return cap, rate


def _positive(value) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{value!r}")
    return number


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _ledger(data_dir: Path) -> dict:
    path = data_dir / LEDGER_NAME
    fresh = {"month": _month(), "usd": 0.0, "calls": 0}
    if not path.exists():
        return fresh
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ledger = {"month": str(data["month"]), "usd": _positive(data["usd"]), "calls": int(data["calls"])}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise _blocked(f"its tally file {path} can't be read ({e}). Delete it to start this month's count over") from e
    return ledger if ledger["month"] == fresh["month"] else fresh


def _write(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


@contextlib.contextmanager
def _locked(data_dir: Path):
    lock = data_dir / LOCK_NAME
    with _thread_lock:
        deadline = time.monotonic() + 10
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                with contextlib.suppress(OSError):
                    if time.time() - lock.stat().st_mtime > 30:  # left behind by a crash
                        lock.unlink()
                        continue
                if time.monotonic() > deadline:
                    raise _blocked("the tally file is busy") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            discard(lock)  # never raises; a lock left behind is swept as stale after 30 s


# ---- counting ----------------------------------------------------------------------------


def _cost(tokens_in: float, tokens_out: float, rate: tuple[float, float]) -> float:
    return (tokens_in * rate[0] + tokens_out * rate[1]) / 1_000_000


def _prompt_tokens(body: dict) -> float:
    return len(json.dumps(body, default=str)) / CHARS_PER_TOKEN


def _max_output(body: dict) -> float:
    config = body.get("generationConfig") if isinstance(body.get("generationConfig"), dict) else {}
    for value in (body.get("max_tokens"), body.get("max_output_tokens"), body.get("max_completion_tokens"),
                  config.get("maxOutputTokens")):
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return float(FALLBACK_MAX_OUTPUT)


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _usage(body) -> tuple[float, float] | None:
    """(input, output) tokens from a response, in any provider's wording. None if absent."""
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if isinstance(usage, dict):
        tokens_in, tokens_out = _num(usage.get("input_tokens")), _num(usage.get("output_tokens"))
        if tokens_in is None and tokens_out is None:
            tokens_in, tokens_out = _num(usage.get("prompt_tokens")), _num(usage.get("completion_tokens"))
        if tokens_in is not None or tokens_out is not None:
            cached = (_num(usage.get("cache_read_input_tokens")) or 0.0) \
                + CACHE_WRITE_FACTOR * (_num(usage.get("cache_creation_input_tokens")) or 0.0)
            return (tokens_in or 0.0) + cached, tokens_out or 0.0
    meta = body.get("usageMetadata")
    if isinstance(meta, dict) and (_num(meta.get("promptTokenCount")) is not None
                                   or _num(meta.get("candidatesTokenCount")) is not None):
        return (_num(meta.get("promptTokenCount")) or 0.0,
                (_num(meta.get("candidatesTokenCount")) or 0.0) + (_num(meta.get("thoughtsTokenCount")) or 0.0))
    return None


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    try:
        print(json.dumps(status(folder), indent=2))
    except LLMError as e:
        print(e.message, file=sys.stderr)
        sys.exit(1)
