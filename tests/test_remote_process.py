# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native process ownership, bounded messages and sanitized worker failures."""

import asyncio
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_execution import make_executor
from worker_fakes import wait_until

from image_embedder import remote_process, remote_protocol, remote_worker
from image_embedder.config import Settings
from image_embedder.input_limits import InputLimitExceeded
from image_embedder.remote_fetch import RemoteFetchError


@pytest.fixture
def anyio_backend():
    return "asyncio"


def options(maximum=3):
    return remote_protocol.RemoteFetchOptions(["images.example"], maximum, 2)


def track_children(monkeypatch):
    children = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(remote_process.subprocess, "Popen", launch)
    return children


def harness(monkeypatch, tmp_path, scenario="dns", port=0, certificate=""):
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).with_name("remote_process_harness.py")),
        str(Path(remote_process.__file__).parent.parent),
        scenario,
        str(port),
        certificate,
        str(tmp_path / "rss.txt"),
    ]
    monkeypatch.setattr(remote_process, "_command", lambda: command)
    return track_children(monkeypatch)


def test_production_worker_blocks_private_destination_and_hides_ambient_secrets(
    monkeypatch,
):
    monkeypatch.setenv("SERVICE_API_KEY", "test-service-credential")
    monkeypatch.setenv("PYTHONPATH", "/nonexistent/python-hooks")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    children = track_children(monkeypatch)
    with pytest.raises(ValueError, match="private address"):
        remote_process.fetch_remote_image(
            "http://127.0.0.1/poster?token=test-url-credential",
            Settings(allow_remote_urls=True),
        )
    assert len(children) == 1 and children[0].poll() == 0
    assert children[0].stdin.closed
    assert "test-url-credential" not in " ".join(children[0].args)
    assert "SERVICE_API_KEY" not in remote_process._environment()


def test_disabled_urls_and_oversized_request_never_launch(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("refused input launched a child")

    monkeypatch.setattr(remote_process.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="disabled"):
        remote_process.fetch_remote_image("https://images.example/a", Settings())
    with pytest.raises(ValueError, match="Invalid image URL"):
        remote_process.fetch_remote_image(
            "https://images.example/" + "x" * 8192, Settings(allow_remote_urls=True)
        )


def test_native_blocked_dns_is_terminated_and_reaped(monkeypatch, tmp_path):
    children = harness(monkeypatch, tmp_path)
    started = time.monotonic()
    with pytest.raises(RemoteFetchError, match="timed out"):
        remote_process.fetch_remote_image(
            "https://images.example/poster?token=test-url-credential",
            Settings(allow_remote_urls=True, remote_fetch_timeout_seconds=0.7),
        )
    assert 0.65 <= time.monotonic() - started < 3
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed


@pytest.mark.anyio
async def test_detached_inference_keeps_permit_until_child_reaped(
    monkeypatch, tmp_path
):
    children = harness(monkeypatch, tmp_path)
    queue, executor = make_executor()
    settings = Settings(allow_remote_urls=True, remote_fetch_timeout_seconds=0.7)
    requester = asyncio.create_task(
        executor.run(
            remote_process.fetch_remote_image, "https://images.example/a", settings
        )
    )
    await wait_until(lambda: bool(children))
    requester.cancel()
    with pytest.raises(asyncio.CancelledError):
        await requester
    assert queue.stats().in_flight == queue.stats().rw_readers == 1
    assert children[0].poll() is None
    assert await executor.close(3)
    assert children[0].poll() is not None
    assert queue.stats().in_flight == queue.stats().rw_readers == 0


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"Zanything",
        b"Vtoken=test-url-credential",
        b"Ftoken=test-url-credential",
        b"Iwrong",
        b"V\xff",
        b"Oabcd",
        b"V" + b"x" * 1024,
    ],
)
def test_malformed_response_is_bounded_and_sanitized(payload):
    with pytest.raises(RemoteFetchError, match="^Unable to fetch remote image$"):
        remote_protocol.decode_response(payload, 3)


@pytest.mark.parametrize(
    "error,kind,detail",
    [
        (ValueError("Invalid image URL"), ValueError, "Invalid image URL"),
        (
            InputLimitExceeded("anything"),
            InputLimitExceeded,
            "Image payload exceeds maximum size",
        ),
        (
            RemoteFetchError("Remote image request failed (HTTP 503)"),
            RemoteFetchError,
            "HTTP 503",
        ),
        (ValueError("token=test-url-credential"), RemoteFetchError, "Unable to fetch"),
        (
            RuntimeError("token=test-url-credential"),
            RemoteFetchError,
            "Unable to fetch",
        ),
    ],
)
def test_worker_errors_preserve_only_reviewed_contract(error, kind, detail):
    encoded = remote_protocol.encode_error(error)
    assert len(encoded) <= remote_protocol.MAX_ERROR_BYTES
    with pytest.raises(kind, match=detail):
        remote_protocol.decode_response(encoded, 3)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"url": "a", "hosts": [], "max_bytes": True, "hop_timeout": 1},
        {"url": "a", "hosts": [1], "max_bytes": 3, "hop_timeout": 1},
        {"url": "a", "hosts": [], "max_bytes": 3, "hop_timeout": False},
    ],
)
def test_invalid_worker_request_fails_closed(value):
    with pytest.raises(ValueError):
        remote_protocol.decode_request(json.dumps(value).encode())


def test_protocol_request_size_and_success_boundaries():
    payload = remote_protocol.encode_request("https://images.example/a", options())
    url, decoded = remote_protocol.decode_request(payload)
    assert url == "https://images.example/a" and decoded == options()
    assert remote_protocol.decode_response(b"Oabc", 3) == b"abc"
    with pytest.raises(ValueError):
        remote_protocol.encode_request(
            "a", remote_protocol.RemoteFetchOptions(["x" * 65536], 3, 2)
        )
    with pytest.raises(ValueError):
        remote_protocol.decode_request(b"x" * 65537)


@pytest.mark.parametrize("failure", [False, True])
def test_worker_entrypoint_reads_bounded_input_and_writes_sanitized_result(
    monkeypatch, failure
):
    source = io.BytesIO(
        remote_protocol.encode_request("https://images.example/a", options())
    )
    output = io.BytesIO()
    monkeypatch.setattr(
        remote_worker.sys, "stdin", type("Input", (), {"buffer": source})()
    )
    monkeypatch.setattr(
        remote_worker.sys, "stdout", type("Output", (), {"buffer": output})()
    )

    def fetch(*_args):
        if failure:
            raise ValueError("Invalid image URL")
        return b"abc"

    monkeypatch.setattr(remote_worker, "fetch_remote_image", fetch)
    remote_worker.main()
    assert output.getvalue() == (b"VInvalid image URL" if failure else b"Oabc")


@pytest.mark.parametrize("behavior", ["failed", "oversized", "startup-budget"])
def test_supervisor_rejects_bad_worker_and_counts_process_startup(
    monkeypatch, behavior
):
    if behavior == "failed":
        script = "import sys;sys.exit(2)"
    elif behavior == "oversized":
        script = (
            "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(b'O'+b'x'*2048)"
        )
    else:
        script = "import time;time.sleep(60)"
    monkeypatch.setattr(
        remote_process, "_command", lambda: [sys.executable, "-I", "-c", script]
    )
    children = track_children(monkeypatch)
    if behavior == "startup-budget":
        launch = remote_process.subprocess.Popen

        def delayed_launch(*args, **kwargs):
            child = launch(*args, **kwargs)
            time.sleep(0.1)
            return child

        monkeypatch.setattr(remote_process.subprocess, "Popen", delayed_launch)
    with pytest.raises(RemoteFetchError):
        remote_process.fetch_remote_image(
            "https://images.example/a",
            Settings(
                allow_remote_urls=True,
                max_image_bytes=3,
                remote_fetch_timeout_seconds=0.05
                if behavior == "startup-budget"
                else 3,
            ),
        )
    assert children[0].poll() is not None and children[0].stdin.closed


def test_process_launch_failure_is_sanitized(monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("token=test-url-credential")

    monkeypatch.setattr(remote_process.subprocess, "Popen", broken)
    with pytest.raises(RemoteFetchError, match="^Unable to fetch remote image$"):
        remote_process.fetch_remote_image(
            "https://images.example/a", Settings(allow_remote_urls=True)
        )


@pytest.mark.parametrize("failure", ["kill", "communication"])
def test_cleanup_failure_still_closes_pipes_and_reaps_before_return(
    monkeypatch, failure
):
    script = "import time;time.sleep(0.2)"
    monkeypatch.setattr(
        remote_process, "_command", lambda: [sys.executable, "-I", "-c", script]
    )
    launch = subprocess.Popen
    children = []

    def broken(*_args, **_kwargs):
        raise OSError("test-private-error-detail")

    def start(*args, **kwargs):
        child = launch(*args, **kwargs)
        children.append(child)
        if failure == "kill":
            child.kill = broken
        else:
            child.communicate = broken
        return child

    monkeypatch.setattr(remote_process.subprocess, "Popen", start)
    started = time.monotonic()
    with pytest.raises(RemoteFetchError, match="^Unable to fetch remote image$"):
        remote_process.fetch_remote_image(
            "https://images.example/a",
            Settings(allow_remote_urls=True, remote_fetch_timeout_seconds=0.05),
        )
    assert children[0].poll() is not None and children[0].stdin.closed
    if failure == "kill":
        assert time.monotonic() - started >= 0.2
