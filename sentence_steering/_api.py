"""Rate-limit handling shared by the API-calling scripts (no torch dependency).

Two mechanisms, same as utils.chat uses on the annotation path:

  set_min_interval(s)  minimum seconds between the START of any two calls, enforced across
                       threads. Paces sustained load so the limit is not tripped at all.
  with_backoff(fn)     exponential backoff with jitter on rate-limit errors, honouring
                       Retry-After when the provider sends one.

Pacing is the part that actually prevents 429s; backoff only recovers from them. If you are
still limited after raising the interval, lower the worker count -- more workers with the same
interval does not increase throughput, it just deepens the queue.
"""

import random
import threading
import time

MIN_INTERVAL = 0.0
MAX_ATTEMPTS = 6

_lock = threading.Lock()
_last = [0.0]


def set_min_interval(seconds):
    global MIN_INTERVAL
    MIN_INTERVAL = max(0.0, float(seconds))


def _throttle():
    """Block until MIN_INTERVAL has passed since the last call started.

    The lock is held across the sleep so concurrent workers queue up instead of all waking
    together and firing simultaneously.
    """
    if MIN_INTERVAL <= 0:
        return
    with _lock:
        wait = _last[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def is_rate_limit(exc):
    if getattr(exc, "status_code", None) == 429:
        return True
    if type(exc).__name__ in ("RateLimitError", "APIStatusError"):
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


def retry_after(exc):
    """Seconds the provider asked us to wait, if it said so."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "Retry-After"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                pass
    return None


def with_backoff(fn, *args, label="", max_attempts=None, **kwargs):
    """Call fn(*args, **kwargs), retrying rate limits with exponential backoff."""
    attempts = max_attempts or MAX_ATTEMPTS
    for attempt in range(attempts):
        try:
            _throttle()
            return fn(*args, **kwargs)
        except Exception as exc:
            limited = is_rate_limit(exc)
            if attempt == attempts - 1:
                print(f"giving up after {attempts} attempts{' ' + label if label else ''}: {exc}")
                raise
            # 2, 4, 8, 16, 32s capped at 60, plus jitter so parallel workers do not retry in
            # lockstep. A Retry-After from the provider always wins.
            delay = retry_after(exc) or min(60.0, 2.0 ** (attempt + 1))
            delay += random.uniform(0, 1)
            if limited:
                print(f"rate limited{' ' + label if label else ''}, retrying in {delay:.1f}s "
                      f"({attempt + 1}/{attempts})")
            else:
                print(f"error{' ' + label if label else ''}: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)
