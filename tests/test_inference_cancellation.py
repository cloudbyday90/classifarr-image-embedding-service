# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise real ownership and backend boundaries without publisher weights."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from worker_fakes import wait_until

from image_embedder.batch import BatchWindow, EmbedJob
from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, ImageEmbedder
from image_embedder.execution import InferenceExecutor
from image_embedder.execution_cancellation import (
    InferenceCancelled,
    cancellation_checkpoint,
    cancellation_scope,
)
from image_embedder.queue import EmbedQueue


def prepared_embedder(monkeypatch, *, backend="ov:CPU", size=2, cache=1000):
    embedder = ImageEmbedder(
        Settings(
            warmup_on_startup=False, inference_batch_size=size, embed_cache_size=cache
        )
    )
    spec = embedder.resolve_model("ViT-B-16")
    state = SimpleNamespace(
        images=[], chunks=[], calls=0, processor_hook=None, model_hook=None
    )

    def resolve(_url, value, **_kwargs):
        if value == "fetch-error":
            raise ValueError("controlled input failure")
        return value.encode()

    def decode(data, **_kwargs):
        if data == b"decode-error":
            raise ValueError("controlled decode failure")
        image = Image.new("RGB", (1, 1), (int(data), 0, 0))
        state.images.append(image)
        return image

    def processor(*, images, return_tensors, size):
        markers = [image.getpixel((0, 0))[0] for image in images]
        state.chunks.append(markers)
        if state.processor_hook:
            state.processor_hook()
        values = np.asarray(
            [[float(marker)] * spec.dims for marker in markers], dtype=np.float32
        )
        if return_tensors == "np":
            return {"pixels": values}
        import torch

        return {"pixels": torch.from_numpy(values)}

    def model(*args, **kwargs):
        state.calls += 1
        if state.model_hook:
            state.model_hook()
        if args:
            return {0: args[0]["pixels"]}
        return SimpleNamespace(image_embeds=kwargs["pixels"])

    monkeypatch.setattr(embedder, "_resolve_image_bytes", resolve)
    monkeypatch.setattr(embedder, "_image_from_bytes", decode)
    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: (model, processor, backend)
    )
    return embedder, spec, state


def assert_closed(images):
    for image in images:
        with pytest.raises(ValueError, match="closed"):
            image.getpixel((0, 0))


@pytest.mark.parametrize("backend", ["cpu", "ov:CPU"])
def test_chunks_preserve_order_partial_errors_normalization_and_cache(
    monkeypatch, backend
):
    embedder, spec, state = prepared_embedder(monkeypatch, backend=backend)
    items = [
        BatchItem(None, value, index % 2 == 0)
        for index, value in enumerate(
            ["1", "fetch-error", "2", "decode-error", "3", "4", "5"]
        )
    ]
    result = embedder.embed_batch(spec, 224, items)
    assert state.chunks == [[1, 2], [3, 4], [5]] and state.calls == 3
    for index, item in enumerate(items):
        if index in (1, 3):
            assert isinstance(result[index], ValueError)
        else:
            expected = float(item.image_base64)
            if item.normalize:
                expected = 1 / np.sqrt(spec.dims)
            np.testing.assert_allclose(
                result[index][0], [expected] * spec.dims, rtol=1e-6, atol=1e-6
            )
    assert_closed(state.images)
    again = embedder.embed_batch(spec, 224, items)
    assert state.calls == 3  # Five successes are content-keyed cache hits.
    assert [type(row) for row in again] == [type(row) for row in result]


@pytest.mark.parametrize("backend", ["cpu", "ov:CPU"])
@pytest.mark.parametrize("count", [1, 3])
def test_cancel_keeps_active_chunk_owned_then_skips_future_chunks(
    monkeypatch, backend, count, caplog
):
    embedder, spec, state = prepared_embedder(monkeypatch, backend=backend)
    started, release = threading.Event(), threading.Event()
    phases = []
    trace = SimpleNamespace(mark=phases.append)
    observer = SimpleNamespace(begin=lambda: trace)

    def gate():
        if state.calls == 1:
            started.set()
            assert release.wait(5)

    state.model_hook = gate

    async def scenario():
        queue = EmbedQueue(1, 2, 60)
        executor = InferenceExecutor(queue, observer=observer)
        request = asyncio.create_task(
            executor.run(
                embedder.embed_batch,
                spec,
                224,
                [BatchItem(None, str(i + 1), False) for i in range(count)],
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert queue.stats().in_flight == queue.stats().rw_readers == 1
            assert not await executor.close(0)
            assert all(image.getpixel((0, 0)) is not None for image in state.images)
        finally:
            release.set()
            await executor.close(5)
        assert queue.stats().in_flight == queue.stats().rw_readers == 0
        await wait_until(lambda: "owner_completed" in phases)
        cancellation_checkpoint()  # Parent context is never left cancelled.

    asyncio.run(scenario())
    assert state.calls == 1
    assert_closed(state.images)
    assert ("native_cancelled" in phases) is (count > 2)
    assert ("owner_cancelled" in phases) is (count > 2)
    assert (
        "native_failed" not in phases and "Detached inference failed" not in caplog.text
    )
    assert embedder._embedding_cache.info()["size"] == (1 if count == 1 else 0)


@pytest.mark.parametrize("cancel_all", [False, True])
def test_coalesced_member_cancellation_respects_remaining_waiter(
    monkeypatch, cancel_all
):
    embedder, _spec, state = prepared_embedder(monkeypatch, size=1)
    started, release = threading.Event(), threading.Event()

    def gate():
        if state.calls == 1:
            started.set()
            assert release.wait(5)

    state.model_hook = gate

    async def scenario():
        queue = EmbedQueue(1, 2, 60)
        executor = InferenceExecutor(queue)
        window = BatchWindow(embedder, queue, 100, 2, executor)
        tasks = [
            asyncio.create_task(
                window.submit(EmbedJob(None, str(i + 1), "ViT-B-16", False, None))
            )
            for i in range(2)
        ]
        try:
            assert await asyncio.to_thread(started.wait, 3)
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            if cancel_all:
                tasks[1].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[1]
                await wait_until(lambda: not window._groups)
            assert queue.stats().in_flight == queue.stats().rw_readers == 1
            release.set()
            if not cancel_all:
                result = await tasks[1]
                assert result[0] == [2.0] * 512
        finally:
            release.set()
            await window.stop()
            assert await executor.close(5)
            await asyncio.gather(*tasks, return_exceptions=True)
        assert queue.stats().in_flight == queue.stats().rw_readers == 0

    asyncio.run(scenario())
    assert state.calls == (1 if cancel_all else 2)
    assert_closed(state.images)


@pytest.mark.parametrize("boundary", ["input", "decode", "load", "processor"])
def test_stop_before_forward_closes_prepared_images(monkeypatch, boundary):
    embedder, spec, state = prepared_embedder(monkeypatch)
    stop = threading.Event()
    target = {
        "input": "_resolve_image_bytes",
        "decode": "_image_from_bytes",
        "load": "_load_model",
    }.get(boundary)
    if target:
        original = getattr(embedder, target)

        def finish_then_stop(*args, **kwargs):
            result = original(*args, **kwargs)
            stop.set()
            return result

        monkeypatch.setattr(embedder, target, finish_then_stop)
    else:
        state.processor_hook = stop.set
    with cancellation_scope(stop), pytest.raises(InferenceCancelled):
        embedder.embed_batch(
            spec, 224, [BatchItem(None, "1", False), BatchItem(None, "2", False)]
        )
    assert state.calls == 0
    assert_closed(state.images)
    cancellation_checkpoint()


def test_active_native_error_survives_stop_request(monkeypatch):
    embedder, spec, state = prepared_embedder(monkeypatch)
    stop = threading.Event()

    def fail():
        stop.set()
        raise ValueError("controlled native failure")

    state.model_hook = fail
    with (
        cancellation_scope(stop),
        pytest.raises(ValueError, match="controlled native failure"),
    ):
        embedder.embed_batch(spec, 224, [BatchItem(None, "1", False)])
    assert_closed(state.images)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_native_chunk_size_requires_positive_integer(value):
    with pytest.raises(ValueError, match="inference_batch_size"):
        Settings(inference_batch_size=value)


def test_nested_stop_contexts_restore_after_failure():
    outer, inner = threading.Event(), threading.Event()
    outer.set()
    with cancellation_scope(outer):
        with cancellation_scope(inner):
            cancellation_checkpoint()
        with pytest.raises(InferenceCancelled):
            cancellation_checkpoint()
    cancellation_checkpoint()


@pytest.mark.parametrize(
    "fault", ["none", "missing_checkpoint", "corrupt_completed_chunk"]
)
def test_cooperative_probe_requires_skipped_work_and_private_parity(monkeypatch, fault):
    from capacity_cancellation import check_cooperative_capacity

    from image_embedder import batch_inference

    embedder, spec, state = prepared_embedder(monkeypatch, cache=0)
    subject = SimpleNamespace(
        embedder=embedder,
        models=[spec.name],
        payloads=["1", "2"],
        references={spec.name: [np.ones(spec.dims), np.ones(spec.dims) * 2]},
    )
    if fault == "missing_checkpoint":
        monkeypatch.setattr(batch_inference, "cancellation_checkpoint", lambda: None)
    elif fault == "corrupt_completed_chunk":
        original, processor, device = embedder._load_model(spec)

        def corrupt_first(*args, **kwargs):
            result = original(*args, **kwargs)
            if state.calls == 1:
                result[0][0][0] += 10
            return result

        monkeypatch.setattr(
            embedder, "_load_model", lambda _: (corrupt_first, processor, device)
        )
    if fault != "none":
        with pytest.raises(AssertionError, match="evidence is incomplete"):
            asyncio.run(check_cooperative_capacity(subject))
    else:
        result = asyncio.run(check_cooperative_capacity(subject))["models"][0]
        assert result["skipped_images"] == 30
        assert result["full_cancelled_batch_verified"] is None
        assert result["committed_chunk_and_recovery"] == [
            {"items": 2, "verified": True},
            {"items": 1, "verified": True},
        ]
        assert result["settled"]["in_flight"] == result["settled"]["rw_readers"] == 0
    assert_closed(state.images)


@pytest.mark.parametrize("backend", ["cpu", "ov:CPU"])
def test_single_image_preparation_stop_does_not_forward(monkeypatch, backend):
    embedder, spec, state = prepared_embedder(monkeypatch, backend=backend)
    stop = threading.Event()
    state.processor_hook = stop.set
    # Single-image processor accepts a list only in this fixture; normalize its input.
    _, processor, device = embedder._load_model(spec)
    model = embedder._load_model(spec)[0]
    monkeypatch.setattr(
        embedder,
        "_load_model",
        lambda _: (
            model,
            lambda images, **kwargs: processor(images=[images], **kwargs),
            device,
        ),
    )
    with cancellation_scope(stop), pytest.raises(InferenceCancelled):
        embedder.embed(None, "1", spec.name, False, None)
    assert state.calls == 0
    assert_closed(state.images)


def test_stop_signal_does_not_leak_between_concurrent_owners():
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    completed = []

    def work(index):
        started[index].set()
        assert release.wait(5)
        cancellation_checkpoint()
        completed.append(index)

    async def scenario():
        queue = EmbedQueue(2, 0, 60)
        executor = InferenceExecutor(queue)
        tasks = [asyncio.create_task(executor.run(work, i)) for i in range(2)]
        try:
            assert all(
                await asyncio.gather(
                    *(asyncio.to_thread(event.wait, 3) for event in started)
                )
            )
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert queue.stats().in_flight == 2
            release.set()
            await tasks[1]
        finally:
            release.set()
            await executor.close(5)
            await asyncio.gather(*tasks, return_exceptions=True)
        assert queue.stats().in_flight == 0

    asyncio.run(scenario())
    assert completed == [1]


def test_detach_while_waiting_for_anyio_worker_does_not_start_preparation(monkeypatch):
    from anyio.to_thread import current_default_thread_limiter, run_sync

    embedder, spec, state = prepared_embedder(monkeypatch)
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        started.set()
        assert release.wait(5)

    async def scenario():
        limiter = current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = 1
        holder = asyncio.create_task(run_sync(hold_worker))
        queue = EmbedQueue(1, 0, 60)
        executor = InferenceExecutor(queue)
        request = None
        try:
            assert await asyncio.to_thread(started.wait, 3)
            request = asyncio.create_task(
                executor.run(
                    embedder.embed_batch, spec, 224, [BatchItem(None, "1", False)]
                )
            )
            await wait_until(lambda: limiter.statistics().tasks_waiting == 1)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert queue.stats().in_flight == queue.stats().rw_readers == 1
        finally:
            release.set()
            await holder
            await executor.close(5)
            if request is not None:
                await asyncio.gather(request, return_exceptions=True)
            limiter.total_tokens = previous
        assert queue.stats().in_flight == queue.stats().rw_readers == 0

    asyncio.run(scenario())
    assert state.calls == 0 and state.images == []


def test_native_chunk_environment_setting_is_validated(monkeypatch):
    monkeypatch.setenv("EMBED_INFERENCE_BATCH_SIZE", "3")
    assert Settings().inference_batch_size == 3
    monkeypatch.setenv("EMBED_INFERENCE_BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        Settings()


@pytest.mark.parametrize("backend", ["cpu", "ov:CPU"])
def test_invalid_vector_stays_ordered_and_uncached_across_chunks(monkeypatch, backend):
    embedder, spec, state = prepared_embedder(monkeypatch, backend=backend)
    original, processor, device = embedder._load_model(spec)

    def corrupt_one(*args, **kwargs):
        result = original(*args, **kwargs)
        if state.calls == 1:
            values = result[0] if args else result.image_embeds
            values[1][0] = float("nan")
        return result

    monkeypatch.setattr(
        embedder, "_load_model", lambda _: (corrupt_one, processor, device)
    )
    items = [BatchItem(None, str(i), False) for i in (1, 2, 3)]
    result = embedder.embed_batch(spec, 224, items)
    assert result[0][0] == [1.0] * spec.dims and result[2][0] == [3.0] * spec.dims
    assert isinstance(result[1], ValueError)
    assert embedder._embedding_cache.info()["size"] == 2
    again = embedder.embed_batch(spec, 224, items)
    assert again[1][0] == [2.0] * spec.dims
    assert state.chunks == [[1, 2], [3], [2]]
    assert_closed(state.images)
