"""Keep Windows from sleeping while the daemon runs (SetThreadExecutionState)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

Setter = Callable[[int], object]


def _default_setter() -> Setter | None:
    if sys.platform != "win32":
        return None
    import ctypes

    return ctypes.windll.kernel32.SetThreadExecutionState  # type: ignore[attr-defined,no-any-return]


def _apply(flags: int, setter: Setter | None) -> bool:
    fn = setter if setter is not None else _default_setter()
    if fn is None:
        return False
    try:
        fn(flags)
    except OSError:
        return False
    return True


def on(setter: Setter | None = None) -> bool:
    """Ask Windows to stay awake (continuous, system required); False where unsupported."""
    return _apply(ES_CONTINUOUS | ES_SYSTEM_REQUIRED, setter)


def off(setter: Setter | None = None) -> bool:
    """Release the request made by `on`."""
    return _apply(ES_CONTINUOUS, setter)


@contextmanager
def keep_awake(setter: Setter | None = None) -> Iterator[None]:
    on(setter)
    try:
        yield
    finally:
        off(setter)
