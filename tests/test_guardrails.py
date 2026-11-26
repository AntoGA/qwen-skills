from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest


def test_limiter_window_expiry_and_concurrent_limit(load_module, monkeypatch):
    mod = load_module('05-evaluation-safety/safety_guardrails.py')
    clock = [0.0]
    monkeypatch.setattr(mod, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    limiter = mod.RateLimiter(max_requests=5, time_window=10)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _:limiter.can_proceed(), range(40))) == 5
    clock[0] = 10
    assert limiter.can_proceed()
    with pytest.raises(ValueError): mod.RateLimiter(max_requests=0)


def test_unknown_output_format_fails_explicitly(load_module):
    mod = load_module('05-evaluation-safety/safety_guardrails.py')
    assert mod.OutputValidator.check_format('{"ok":true}', 'json')
    assert not mod.OutputValidator.check_format('invalid', 'json')
    with pytest.raises(ValueError): mod.OutputValidator.check_format('anything', 'unknown')
