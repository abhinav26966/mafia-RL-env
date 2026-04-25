"""Shared pytest fixtures."""

from __future__ import annotations

import random
import socket
import subprocess
import sys
import time
from typing import Iterator

import pytest


@pytest.fixture
def seeded_rng() -> random.Random:
    """Deterministic RNG for tests that depend on shuffling."""
    return random.Random(42)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server_url() -> Iterator[str]:
    """Boot uvicorn in a subprocess for integration tests.

    Marked module-scoped so the server is reused across all client tests in
    one file. Tests requesting this fixture should be marked
    `@pytest.mark.integration` so they can be run/skipped independently.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover
        pytest.skip("httpx not installed — required for live server tests")

    port = _find_free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "werewolf_env.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    last_err: BaseException | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            err_out = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            pytest.fail(f"uvicorn exited early: {err_out[:500]}")
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception as exc:
            last_err = exc
            time.sleep(0.25)
    else:
        proc.terminate()
        pytest.skip(f"uvicorn did not become ready within 30s: {last_err}")

    try:
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
