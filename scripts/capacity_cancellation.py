# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Controlled committed-forward barrier, private parity and skipped-work proof."""

import asyncio
import threading
import time
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from image_embedder.embedder import BatchItem
from image_embedder.execution import InferenceExecutor
from image_embedder.execution_cancellation import InferenceCancelled
from image_embedder.queue import EmbedQueue


async def check_cooperative_capacity(workload) -> dict:
    settings = workload.embedder.settings
    size = settings.embed_batch_api_max_items
    chunk = settings.inference_batch_size
    if settings.embed_cache_size or not 0 < chunk < size:
        raise ValueError(
            "Cooperative probe needs disabled cache and more than one native chunk"
        )
    records = []
    for name in workload.models:
        records.append(await _check_model(workload, name, size, chunk))
    return {
        "scope": "executor ownership; artificial barrier inside committed forward wrapper; not kernel interruption or HTTP disconnect detection",
        "models": records,
    }


async def _check_model(workload, name, size, chunk):
    embedder = workload.embedder
    spec = embedder.resolve_model(name)
    model, processor, device = embedder._load_model(spec)
    started, release = threading.Event(), threading.Event()
    images, calls, errors, stopped = [], [], [], []
    queue = EmbedQueue(1, 1, 300)
    executor = InferenceExecutor(queue)
    original_batch = embedder.embed_batch

    def checked_processor(**kwargs):
        images.extend(kwargs["images"])
        return processor(**kwargs)

    def committed_forward(*args, **kwargs):
        position = len(calls)
        count = chunk if position == 0 else 1
        calls.append({"items": count, "verified": False})
        try:
            if position == 0:
                started.set()
                if not release.wait(30):
                    raise TimeoutError("Committed-forward barrier was not released")
            raw = model(*args, **kwargs)
            values = (
                raw[0]
                if device.startswith("ov:")
                else raw.image_embeds.detach().cpu().numpy()
            )
            if values.shape != (count, spec.dims) or not np.isfinite(values).all():
                raise AssertionError(
                    "Cooperative native chunk shape/finite check failed"
                )
            for index, value in enumerate(values):
                if not np.allclose(
                    value, workload.references[name][index % 2], rtol=1e-4, atol=1e-4
                ):
                    raise AssertionError("Cooperative native chunk parity failed")
            calls[-1]["verified"] = True
            return raw
        except Exception as error:
            errors.append(type(error).__name__)
            raise

    def checked_batch(*args):
        try:
            return original_batch(*args)
        except InferenceCancelled:
            stopped.append(True)
            raise

    def item(index):
        return BatchItem(None, workload.payloads[index % 2], False)

    first = follower = None
    with (
        patch.object(
            embedder,
            "_load_model",
            return_value=(committed_forward, checked_processor, device),
        ),
        patch.object(embedder, "embed_batch", side_effect=checked_batch),
    ):
        try:
            first = asyncio.create_task(
                executor.run(
                    embedder.embed_batch,
                    spec,
                    spec.image_size,
                    [item(i) for i in range(size)],
                )
            )
            if not await asyncio.to_thread(started.wait, 30):
                raise TimeoutError("No committed native chunk reached the barrier")
            cancelled_at = time.perf_counter()
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            retained = asdict(queue.stats())
            if retained["in_flight"] != 1 or retained["rw_readers"] != 1:
                raise AssertionError("Cancellation released an active native owner")
            follower = asyncio.create_task(
                executor.run(embedder.embed_batch, spec, spec.image_size, [item(0)])
            )
            async with asyncio.timeout(30):
                while queue.stats().waiting != 1:
                    if follower.done():
                        raise AssertionError(
                            "Recovery bypassed active native ownership"
                        )
                    await asyncio.sleep(0.01)
            waiting = asdict(queue.stats())
            release.set()
            result = await follower
            if (
                len(result) != 1
                or not isinstance(result[0], tuple)
                or not np.allclose(
                    result[0][0], workload.references[name][0], rtol=1e-4, atol=1e-4
                )
            ):
                raise AssertionError("Cooperative recovery result failed validation")
            if not await executor.close(300):
                raise TimeoutError("Cooperative native owners did not settle")
            settle_seconds = time.perf_counter() - cancelled_at
            if (
                stopped != [True]
                or errors
                or len(calls) != 2
                or not all(call["verified"] for call in calls)
            ):
                raise AssertionError(
                    "Cooperative native stop, failure or parity evidence is incomplete"
                )
            for image in images:
                try:
                    image.getpixel((0, 0))
                except ValueError:
                    continue
                raise AssertionError("Cooperative inference retained a decoded image")
            settled = asdict(queue.stats())
            if any(settled[key] for key in ("in_flight", "waiting", "rw_readers")):
                raise AssertionError("Cooperative ownership did not settle")
            return {
                "model": name,
                "device": device,
                "batch_size": size,
                "native_chunk_size": chunk,
                "skipped_images": size - chunk,
                "committed_chunk_and_recovery": calls,
                "full_cancelled_batch_verified": None,
                "decoded_images_closed": len(images),
                "retained": retained,
                "recovery_waiting": waiting,
                "settled": settled,
                "cancel_to_recovery_settled_seconds": settle_seconds,
            }
        finally:
            release.set()
            for task in (first, follower):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first, follower) if task is not None),
                return_exceptions=True,
            )
            await executor.close(300)
