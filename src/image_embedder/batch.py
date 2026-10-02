# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Async batch-window coalescer for embed requests.

When ``embed_batch_window_ms > 0``, incoming embed requests are collected for
up to that many milliseconds (or until ``embed_batch_max_size`` is reached),
then dispatched together.  Requests sharing the same ``(model, image_size)``
bucket are sent to the model as a single batched tensor call; ``normalize`` is
applied per-item post-forward so requests with different settings can coexist
in the same window.

When ``batch_window_ms == 0`` (default) the ``BatchWindow`` is disabled and
``app.state.batch_window`` is ``None``; the route falls back to the standard
single-request path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, ParamSpec, Tuple, TypeVar

from .execution import ExecutionClosedError, InferenceExecutor
from .logging_config import get_logger

if TYPE_CHECKING:
    from .embedder import BatchItem, ImageEmbedder, ModelSpec
    from .queue import EmbedQueue

logger = get_logger(__name__)
P = ParamSpec("P")
T = TypeVar("T")

# (embedding, dims, provider, model_name, image_size)
EmbedResult = Tuple[List[float], int, str, str, int]


@dataclass
class EmbedJob:
    """One pending embed request submitted to a BatchWindow."""

    image_url: Optional[str]
    image_base64: Optional[str]
    model: Optional[str]
    normalize: bool
    image_size: Optional[int]
    # Set by bind(); not part of __init__ so callers don't have to provide it.
    _future: "asyncio.Future[EmbedResult]" = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def bind(self, loop: asyncio.AbstractEventLoop) -> "asyncio.Future[EmbedResult]":
        self._future = loop.create_future()
        return self._future


class BatchWindow:
    """Async batch-window coalescer sitting in front of EmbedQueue.

    Usage::

        bw = BatchWindow(embedder, queue, batch_window_ms=50, batch_max_size=8)
        await bw.start()
        result = await bw.submit(job)   # raises on embed/queue errors
        await bw.stop()
    """

    def __init__(
        self,
        embedder: "ImageEmbedder",
        queue: "EmbedQueue",
        batch_window_ms: int,
        batch_max_size: int,
        executor: InferenceExecutor | None = None,
    ) -> None:
        self._embedder = embedder
        self._executor = executor or InferenceExecutor(queue)
        self._owns_executor = executor is None
        self._window_ms = batch_window_ms
        self._max_size = max(1, batch_max_size)
        self._pending: asyncio.Queue[EmbedJob] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._active: List[EmbedJob] = []
        self._stopping = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="batch-window")
        logger.info(
            f"BatchWindow started: window_ms={self._window_ms}, max_size={self._max_size}"
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

        # Collection and dispatch remove jobs from _pending; they must also
        # be settled if the dispatcher is canceled before delivering results.
        for job in self._active:
            if not job._future.done():
                job._future.cancel()
        self._active = []

        # Cancel any jobs still waiting in the queue.
        while True:
            try:
                job = self._pending.get_nowait()
                if not job._future.done():
                    job._future.cancel()
            except asyncio.QueueEmpty:
                break

        if self._owns_executor:
            await self._executor.close(timeout_seconds=30.0)

        logger.info("BatchWindow stopped")

    async def submit(self, job: EmbedJob) -> EmbedResult:
        """Add *job* to the pending queue and return its result (or raise)."""
        if self._stopping:
            raise ExecutionClosedError("service is shutting down")
        future = job.bind(asyncio.get_running_loop())
        await self._pending.put(job)
        return await future

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                first = await self._pending.get()
            except asyncio.CancelledError:
                return

            batch: List[EmbedJob] = [first]
            self._active = batch

            if self._window_ms > 0:
                deadline = asyncio.get_event_loop().time() + self._window_ms / 1000.0
                while len(batch) < self._max_size:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        job = await asyncio.wait_for(
                            self._pending.get(), timeout=remaining
                        )
                        batch.append(job)
                    except (asyncio.TimeoutError, TimeoutError):
                        break

            if len(batch) > 1:
                logger.debug(f"BatchWindow dispatching {len(batch)} requests")

            await self._dispatch(batch)
            self._active = []

    async def _dispatch(self, batch: List[EmbedJob]) -> None:
        """Group batch by (resolved_model, image_size) then embed each group."""
        # Group jobs by (resolved model name, resolved image_size).
        groups: dict[tuple, list[EmbedJob]] = {}
        for job in batch:
            if job._future.done():
                continue
            spec: ModelSpec = self._embedder.resolve_model(job.model)
            target_size: int = spec.image_size if job.image_size is None else job.image_size
            key = (spec.name, target_size)
            groups.setdefault(key, []).append(job)

        for (model_name, image_size), jobs in groups.items():
            jobs = [job for job in jobs if not job._future.done()]
            if not jobs:
                continue
            try:
                if len(jobs) == 1:
                    j = jobs[0]
                    result: EmbedResult = await self._execute_jobs(
                        jobs,
                        self._embedder.embed,
                        j.image_url,
                        j.image_base64,
                        j.model,
                        j.normalize,
                        j.image_size,
                    )
                    if not j._future.done():
                        j._future.set_result(result)
                else:
                    # Import the runtime batch type only on the batched path.
                    from .embedder import BatchItem

                    spec = self._embedder.resolve_model(jobs[0].model)
                    batch_items: List[BatchItem] = [
                        BatchItem(j.image_url, j.image_base64, j.normalize)
                        for j in jobs
                    ]
                    per_item = await self._execute_jobs(
                        jobs, self._embedder.embed_batch, spec, image_size, batch_items
                    )
                    if len(per_item) != len(jobs):
                        raise RuntimeError(
                            "embed_batch returned "
                            f"{len(per_item)} results for {len(jobs)} jobs"
                        )
                    for job, outcome in zip(jobs, per_item):
                        if not job._future.done():
                            if isinstance(outcome, Exception):
                                job._future.set_exception(outcome)
                            else:
                                job._future.set_result(outcome)

            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                # All clients abandoned this group. The executor cancels its
                # admission or retains its running thread; collect the next.
            except Exception as exc:
                for job in jobs:
                    if not job._future.done():
                        job._future.set_exception(exc)

    async def _execute_jobs(
        self, jobs: List[EmbedJob], function: Callable[P, T],
        *args: P.args, **kwargs: P.kwargs,
    ) -> T:
        dispatch = asyncio.create_task(
            self._executor.run(function, *args, **kwargs), name="batch-dispatch"
        )

        def cancel_if_abandoned(_future) -> None:
            if (
                not dispatch.done()
                and not dispatch.cancelling()
                and all(job._future.done() for job in jobs)
            ):
                dispatch.cancel()

        for job in jobs:
            job._future.add_done_callback(cancel_if_abandoned)
        try:
            return await dispatch
        finally:
            for job in jobs:
                job._future.remove_done_callback(cancel_if_abandoned)
