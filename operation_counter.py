from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, Iterator


_COUNTERS: DefaultDict[str, float] = defaultdict(float)
# Keep counters disabled unless a benchmark/test explicitly opts in. This
# avoids charging instrumentation overhead to normal code paths.
_ENABLED: bool = False


def increment(name: str, amount: float = 1.0) -> None:
    if not _ENABLED:
        return
    _COUNTERS[name] += amount


def reset() -> None:
    _COUNTERS.clear()


def snapshot() -> Dict[str, float]:
    return dict(_COUNTERS)


def enabled() -> bool:
    return _ENABLED


@contextmanager
def counting_scope() -> Iterator[None]:
    global _ENABLED
    previous = _ENABLED
    reset()
    _ENABLED = True
    try:
        yield
    finally:
        _ENABLED = previous


@contextmanager
def disabled_scope() -> Iterator[None]:
    global _ENABLED
    previous = _ENABLED
    _ENABLED = False
    try:
        yield
    finally:
        _ENABLED = previous


def total(prefix: str = "") -> float:
    if not prefix:
        return float(sum(_COUNTERS.values()))
    return float(
        sum(value for name, value in _COUNTERS.items() if name.startswith(prefix))
    )
