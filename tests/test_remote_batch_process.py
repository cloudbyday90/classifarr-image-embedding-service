# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native shared deadlines kill/reap before detached batch ownership releases."""

import asyncio
import time

import pytest
from test_execution import make_executor
from test_remote_process import harness, track_children
from test_remote_process_network import controlled_server
from worker_fakes import wait_until

from image_embedder import remote_process
from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, ImageEmbedder
from image_embedder.remote_budget import RemoteBatchBudget
from image_embedder.remote_fetch import RemoteFetchError


@pytest.fixture
def anyio_backend():
    return "asyncio"


def batch_inputs(monkeypatch, *, per_image=5, shared=0.9):
    settings = Settings(
        allow_remote_urls=True,
        remote_fetch_timeout_seconds=per_image,
        remote_batch_fetch_timeout_seconds=shared,
    )
    embedder = ImageEmbedder(settings)
    spec = embedder.resolve_model(None)
    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: pytest.fail("failed input loaded model")
    )
    items = [
        BatchItem("https://images.example/a?token=test-private-detail", None, True)
    ] * 32
    return embedder, spec, items


def test_expired_parent_deadline_never_launches_child(monkeypatch):
    monkeypatch.setattr(
        remote_process.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("expired deadline launched child"),
    )
    with pytest.raises(RemoteFetchError, match="^Remote image fetch timed out$"):
        remote_process.fetch_remote_image(
            "https://images.example/a",
            Settings(allow_remote_urls=True),
            total_deadline=time.monotonic() - 1,
        )


def test_native_32_item_dns_batch_uses_one_shared_budget(monkeypatch, tmp_path):
    children = harness(monkeypatch, tmp_path)
    embedder, spec, items = batch_inputs(monkeypatch)
    started = time.monotonic()
    results = embedder.embed_batch(spec, spec.image_size, items)
    assert 0.8 <= time.monotonic() - started < 4
    assert [str(result) for result in results] == ["Remote batch fetch timed out"] * 32
    assert (
        len(children) == 1
        and children[0].poll() is not None
        and children[0].stdin.closed
    )


@pytest.mark.anyio
async def test_detached_batch_keeps_permit_until_active_fetch_reaped(
    monkeypatch, tmp_path
):
    children = harness(monkeypatch, tmp_path)
    embedder, spec, items = batch_inputs(monkeypatch)
    queue, executor = make_executor()
    pending = asyncio.create_task(
        executor.run(embedder.embed_batch, spec, spec.image_size, items)
    )
    try:
        await wait_until(lambda: bool(children))
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert children[0].poll() is None
        assert queue.stats().in_flight == queue.stats().rw_readers == 1
        assert not await executor.close(0.01)
        assert await executor.close(4)
        assert len(children) == 1 and children[0].poll() is not None
        assert children[0].stdin.closed
        assert queue.stats().in_flight == queue.stats().rw_readers == 0
    finally:
        await executor.close(4)


@pytest.mark.parametrize("shorter", ["image", "batch"])
def test_native_earlier_of_per_image_and_shared_deadline(monkeypatch, shorter):
    import sys

    monkeypatch.setattr(
        remote_process,
        "_command",
        lambda: [sys.executable, "-I", "-c", "import time;time.sleep(60)"],
    )
    children = track_children(monkeypatch)
    settings = Settings(
        allow_remote_urls=True,
        remote_fetch_timeout_seconds=0.2 if shorter == "image" else 5,
        remote_batch_fetch_timeout_seconds=5 if shorter == "image" else 0.2,
    )
    started = time.monotonic()
    expected = (
        "Remote image fetch timed out"
        if shorter == "image"
        else "Remote batch fetch timed out"
    )
    with pytest.raises(RemoteFetchError, match=f"^{expected}$"):
        RemoteBatchBudget(settings).fetch("https://images.example/a")
    assert 0.15 <= time.monotonic() - started < 3
    assert children[0].poll() is not None and children[0].stdin.closed


def test_native_success_then_trickle_spends_remaining_shared_budget(
    monkeypatch, tmp_path
):
    with controlled_server(tmp_path) as (port, certificate, seen, _sni):
        children = harness(monkeypatch, tmp_path, "network", port, certificate)
        budget = RemoteBatchBudget(
            Settings(
                allow_remote_urls=True,
                remote_fetch_timeout_seconds=5,
                remote_batch_fetch_timeout_seconds=1.5,
            )
        )
        started = time.monotonic()
        assert budget.fetch(f"http://images.example:{port}/success") == b"abc"
        with pytest.raises(RemoteFetchError, match="^Remote batch fetch timed out$"):
            budget.fetch(f"http://images.example:{port}/body")
        assert 1.4 <= time.monotonic() - started < 4
        with pytest.raises(RemoteFetchError, match="^Remote batch fetch timed out$"):
            budget.fetch(f"http://images.example:{port}/success")
        assert len(children) == 2 and all(
            child.poll() is not None and child.stdin.closed for child in children
        )
        assert [path for path, _headers in seen] == ["/success", "/body"]
