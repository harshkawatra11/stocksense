"""Server runner tests: the 127.0.0.1-only binding is a security
requirement (this API serves private trading data), so it's tested
directly rather than trusted to a docstring -- both that run() passes
the right host to uvicorn, and that no code path in this module ever
constructs "0.0.0.0"."""

from __future__ import annotations

import inspect
from unittest.mock import patch

from stocksense.server import run as run_module
from stocksense.server.run import LOCALHOST_ONLY, find_free_port, run


def test_localhost_only_constant_is_127_0_0_1() -> None:
    assert LOCALHOST_ONLY == "127.0.0.1"


def test_module_never_constructs_0_0_0_0() -> None:
    source = inspect.getsource(run_module)
    assert "0.0.0.0" not in source


@patch("uvicorn.run")
def test_run_passes_localhost_only_to_uvicorn(mock_uvicorn_run) -> None:
    run(port=8999)
    _, kwargs = mock_uvicorn_run.call_args
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8999


@patch("uvicorn.run")
def test_run_finds_a_port_when_none_given(mock_uvicorn_run) -> None:
    run(port=None)
    _, kwargs = mock_uvicorn_run.call_args
    assert isinstance(kwargs["port"], int)
    assert kwargs["port"] >= 8420


def test_find_free_port_returns_a_bindable_port() -> None:
    import socket

    port = find_free_port(start=18420)
    # verify it's actually free right now by binding to it ourselves
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))  # would raise OSError if not actually free


def test_find_free_port_skips_occupied_port() -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 18500))
        port = find_free_port(start=18500, attempts=5)
        assert port != 18500  # occupied port skipped, a different one returned
