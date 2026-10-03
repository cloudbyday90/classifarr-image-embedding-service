# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Production capacity workloads without additional reference-model ownership."""

import asyncio
import base64
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import BytesIO

import numpy as np
from capacity_metrics import MemorySampler
from PIL import Image

from image_embedder.embedder import BatchItem, ImageEmbedder
from image_embedder.execution import InferenceExecutor
from image_embedder.queue import EmbedQueue


def make_payloads(edge: int) -> list[str]:
    y, x = np.indices((edge, edge))
    payloads = []
    for offset in (0, 31):
        pixels = np.stack(
            [(x * 7 + offset) % 256, (y * 11) % 256, ((x + y) * 3) % 256], axis=-1
        ).astype(np.uint8)
        with Image.fromarray(pixels) as image, BytesIO() as output:
            image.save(output, format="PNG")
            payloads.append(base64.b64encode(output.getvalue()).decode("ascii"))
    return payloads


class CapacityWorkload:
    def __init__(
        self,
        embedder: ImageEmbedder,
        models: list[str],
        sizes: list[int],
        payloads: list[str],
        reader: Callable[[], dict],
        emit: Callable[[dict], None],
        interval: float = 0.1,
        repeats: int = 2,
        expected_device: str = "cpu",
        cuda=None,
    ) -> None:
        self.embedder, self.models, self.sizes = embedder, models, sizes
        self.payloads, self.reader, self.emit = payloads, reader, emit
        self.interval, self.repeats = interval, repeats
        self.expected_device = expected_device
        self.cuda = cuda
        self.references: dict[str, list[np.ndarray]] = {}

    def measure(self, name: str, function: Callable[[], object]) -> object:
        if self.cuda is not None:
            self.cuda.begin_phase()
        started = time.monotonic()
        sampler = MemorySampler(self.reader, self.interval)
        self.emit({"event": "phase_start", "phase": name, "memory": self.reader()})
        try:
            with sampler:
                result = function()
                if self.cuda is not None:
                    self.cuda.end_phase()
        except Exception as error:
            self.emit(
                {
                    "event": "phase_error",
                    "phase": name,
                    "error_type": type(error).__name__,
                }
            )
            raise
        self.emit(
            {
                "event": "phase_end",
                "phase": name,
                "elapsed_seconds": time.monotonic() - started,
                "memory": sampler.report,
            }
        )
        return result

    def load(self, overlap: bool) -> None:
        if overlap:

            def concurrent_loads():
                with ThreadPoolExecutor(max_workers=len(self.models)) as pool:
                    list(pool.map(self.embedder.warmup, self.models))

            self.measure("overlapping_model_loads", concurrent_loads)
        else:
            for name in self.models:
                self.measure(f"load/{name}", lambda: self.embedder.warmup(name))
        devices = {
            name: str(self.embedder._load_model(self.embedder.resolve_model(name))[2])
            for name in self.models
        }
        if any(device != self.expected_device for device in devices.values()):
            raise AssertionError("Loaded model backend does not match the experiment")
        self.emit({"event": "models_loaded", "devices": devices})
        for name in self.models:

            def reference():
                self.references[name] = [
                    np.asarray(self.embedder.embed(None, payload, name, False, 224)[0])
                    for payload in self.payloads
                ]

            self.measure(f"single_references/{name}", reference)

    def batch(self, name: str, size: int) -> None:
        spec = self.embedder.resolve_model(name)
        items = [
            BatchItem(None, self.payloads[i % len(self.payloads)], bool(i % 2))
            for i in range(size)
        ]
        results = self.embedder.embed_batch(spec, spec.image_size, items)
        if len(results) != size:
            raise AssertionError("Batch result count changed")
        for index, result in enumerate(results):
            if not isinstance(result, tuple) or result[1:] != (
                spec.dims,
                "local",
                name,
                spec.image_size,
            ):
                raise AssertionError("Batch result failed its metadata contract")
            expected = self.references[name][index % len(self.payloads)]
            if items[index].normalize:
                expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
            np.testing.assert_allclose(result[0], expected, rtol=1e-4, atol=1e-4)

    def serial_batches(self) -> None:
        for name in self.models:
            for size in self.sizes:
                for repeat in range(self.repeats):
                    self.measure(
                        f"batch/{name}/{size}/{repeat + 1}",
                        lambda: self.batch(name, size),
                    )

    async def detached_owner(self) -> dict:
        """Use a controlled dispatch barrier, then run actual maximum-batch inference."""
        queue = EmbedQueue(concurrency=1, max_queue=1, max_wait_seconds=300)
        executor = InferenceExecutor(queue)
        started, release = threading.Event(), threading.Event()
        timings = {}
        detached_model = max(
            self.models, key=lambda name: self.embedder.resolve_model(name).dims
        )
        live_model = next(
            (name for name in self.models if name != detached_model), detached_model
        )
        submitted = time.monotonic()

        def work(
            name: str, size: int, label: str, queued_at: float, held: bool = False
        ):
            dispatched = time.monotonic()
            if held:
                started.set()
                if not release.wait(30):
                    raise TimeoutError("Dispatch barrier was not released")
            native_start = time.monotonic()
            self.batch(name, size)
            timings[label] = {
                "model": name,
                "batch_size": size,
                "submission_to_dispatch_seconds": dispatched - queued_at,
                "native_and_validation_seconds": time.monotonic() - native_start,
            }

        first = asyncio.create_task(
            executor.run(
                work, detached_model, max(self.sizes), "detached", submitted, True
            )
        )
        follower = None
        try:
            if not await asyncio.to_thread(started.wait, 30):
                raise TimeoutError("No worker reached the dispatch barrier")
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            detached_stats = asdict(queue.stats())
            if detached_stats["in_flight"] != 1:
                raise AssertionError("Canceled caller released a native owner permit")
            follower = asyncio.create_task(
                executor.run(
                    work, live_model, max(self.sizes), "live", time.monotonic()
                )
            )
            async with asyncio.timeout(30):
                while queue.stats().waiting != 1:
                    if follower.done():
                        await follower
                        raise AssertionError("Live work bypassed the retained permit")
                    await asyncio.sleep(0.01)
            waiting_stats = asdict(queue.stats())
            release.set()
            await follower
            if not await executor.close(300):
                raise TimeoutError("Native owner outlived the drain budget")
            final_stats = asdict(queue.stats())
            if set(timings) != {"detached", "live"}:
                raise AssertionError(
                    "A native computation failed after caller cancellation"
                )
            if (
                final_stats["in_flight"]
                or final_stats["waiting"]
                or final_stats["rw_readers"]
            ):
                raise AssertionError("Queue ownership did not settle")
            return {
                "detached": detached_stats,
                "live_waiting": waiting_stats,
                "settled": final_stats,
                "timings": timings,
            }
        finally:
            release.set()
            if not first.done():
                first.cancel()
            if follower is not None and not follower.done():
                follower.cancel()
                await asyncio.gather(follower, return_exceptions=True)
            await asyncio.gather(first, return_exceptions=True)
            await executor.close(300)

    def run(self, scenario: str) -> None:
        self.load(overlap=scenario == "overlap")
        self.serial_batches()
        if scenario == "detached":
            result = self.measure(
                "detached_owner_and_live_waiter",
                lambda: asyncio.run(self.detached_owner()),
            )
            self.emit({"event": "ownership", "result": result})
        elif scenario == "api":
            from capacity_api import check_api_deadline

            result = self.measure(
                "authenticated_api_deadline",
                lambda: asyncio.run(
                    check_api_deadline(
                        self.embedder,
                        self.embedder.settings,
                        self.models,
                        self.payloads[0],
                        self.embedder.settings.embed_batch_api_max_items,
                        self.references,
                    )
                ),
            )
            self.emit({"event": "api_deadline", "result": result})
        elif scenario == "mixed":
            from capacity_mixed import check_mixed_capacity

            result = self.measure(
                "concurrent_mixed_and_detached_inputs",
                lambda: asyncio.run(check_mixed_capacity(self)),
            )
            self.emit({"event": "mixed_inputs", "result": result})
        elif scenario == "socket":
            from capacity_socket import check_socket_capacity

            result = self.measure(
                "native_socket_upload_capacity",
                lambda: asyncio.run(check_socket_capacity(self)),
            )
            self.emit({"event": "socket_upload_capacity", "result": result})
