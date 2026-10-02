# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded per-model batch collection using shared inference admission."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from .admission import AdmissionTicket
from .execution import ExecutionClosedError, InferenceExecutor
from .logging_config import get_logger

if TYPE_CHECKING:
    from .embedder import ImageEmbedder, ModelSpec
    from .queue import EmbedQueue

logger = get_logger(__name__)
EmbedResult = tuple[list[float], int, str, str, int]
GroupResult = list[EmbedResult | Exception]


@dataclass(eq=False)
class EmbedJob:
    """One client; its payload is retained only after admission succeeds."""

    image_url: str | None
    image_base64: str | None
    model: str | None
    normalize: bool
    image_size: int | None
    _future: asyncio.Future[EmbedResult] = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def bind(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future[EmbedResult]:
        if self._future is not None:
            raise ValueError("an embed job may only be submitted once")
        self._future = loop.create_future()
        return self._future


@dataclass(eq=False, slots=True)
class _Group:
    key: tuple[str, int]
    spec: ModelSpec
    ticket: AdmissionTicket
    jobs: list[EmbedJob] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class BatchWindow:
    """Collect bounded compatible groups without a second waiting room.

    Waiting members count individually against EmbedQueue's waiting budget.
    Admitted collecting groups reserve one computation slot; compatible joins
    share that slot up to batch_max_size and never extend its collection window.
    """

    def __init__(
        self,
        embedder: ImageEmbedder,
        queue: EmbedQueue,
        batch_window_ms: int,
        batch_max_size: int,
        executor: InferenceExecutor | None = None,
    ) -> None:
        self._embedder = embedder
        self._queue = queue
        self._executor = executor or InferenceExecutor(queue)
        self._owns_executor = executor is None
        self._window_ms = batch_window_ms
        self._max_size = max(1, batch_max_size) if batch_window_ms > 0 else 1
        self._open: dict[tuple[str, int], _Group] = {}
        self._groups: set[_Group] = set()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._stop_event = asyncio.Event()

    @property
    def _active(self) -> list[EmbedJob]:
        return [job for group in self._groups for job in group.jobs]

    async def start(self) -> None:
        if self._stopping:
            raise ExecutionClosedError("service is shutting down")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="batch-window")
            logger.info(
                "BatchWindow started: window_ms=%s, max_size=%s",
                self._window_ms,
                self._max_size,
            )

    async def stop(self) -> None:
        self._stopping = True
        self._stop_event.set()
        if self._task is not None:
            await self._task
        else:
            await self._stop_groups()
        if self._owns_executor:
            await self._executor.close(timeout_seconds=30.0)

    async def submit(self, job: EmbedJob) -> EmbedResult:
        if self._stopping:
            raise ExecutionClosedError("service is shutting down")
        if job._future is not None:
            raise ValueError("an embed job may only be submitted once")
        spec = self._embedder.resolve_model(job.model)
        size = spec.image_size if job.image_size is None else job.image_size
        key = (spec.name, size)
        group = self._open.get(key)
        if group is not None and not group.ticket.finished:
            group.ticket.add_member()
        else:
            group = _Group(key, spec, self._queue.reserve())
            self._groups.add(group)
            self._open[key] = group
        future = job.bind(asyncio.get_running_loop())
        group.jobs.append(job)
        group.changed.set()
        if len(group.jobs) >= self._max_size:
            self._seal(group)
        if group.task is None:
            group.task = asyncio.create_task(self._run_group(group), name="batch-group")
            group.task.add_done_callback(partial(self._group_completed, group))
        try:
            return await future
        finally:
            group.jobs.remove(job)
            group.ticket.remove_member()
            group.changed.set()
            if not group.jobs:
                self._seal(group)
                self._cancel_group(group)

    def _seal(self, group: _Group) -> None:
        if self._open.get(group.key) is group:
            del self._open[group.key]

    def _group_completed(self, group: _Group, task: asyncio.Task[None]) -> None:
        self._groups.discard(group)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.warning("Batch group failed (%s)", type(error).__name__)

    @staticmethod
    def _cancel_group(group: _Group) -> None:
        if (
            group.task is not None
            and not group.task.done()
            and not group.task.cancelling()
        ):
            group.task.cancel()

    async def _run(self) -> None:
        try:
            await self._stop_event.wait()
        finally:
            await self._stop_groups()

    async def _stop_groups(self) -> None:
        groups = tuple(self._groups)
        for group in groups:
            self._cancel_group(group)
            # An immediately canceled task may never enter its coroutine.
            self._seal(group)
            for job in group.jobs:
                if not job._future.done():
                    job._future.cancel()
            group.ticket.discard()
        await asyncio.gather(
            *(group.task for group in groups if group.task is not None),
            return_exceptions=True,
        )
        self._groups.difference_update(groups)

    async def _run_group(self, group: _Group) -> None:
        dispatched: list[EmbedJob] = []
        try:
            await group.ticket.wait()
            deadline = asyncio.get_running_loop().time() + self._window_ms / 1000.0
            while (
                self._open.get(group.key) is group and len(group.jobs) < self._max_size
            ):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0 or not group.jobs:
                    break
                group.changed.clear()
                try:
                    await asyncio.wait_for(group.changed.wait(), remaining)
                except TimeoutError:
                    break
            self._seal(group)
            results = await self._executor.run_admitted(
                group.ticket, partial(self._prepare, group, dispatched)
            )
            if len(results) != len(dispatched):
                raise RuntimeError(
                    f"embed_batch returned {len(results)} results for {len(dispatched)} jobs"
                )
            for job, outcome in zip(dispatched, results):
                if not job._future.done():
                    if isinstance(outcome, Exception):
                        job._future.set_exception(outcome)
                    else:
                        job._future.set_result(outcome)
        except asyncio.CancelledError:
            for job in group.jobs:
                if not job._future.done():
                    job._future.cancel()
        except Exception as error:
            for job in group.jobs:
                if not job._future.done():
                    job._future.set_exception(error)
        finally:
            self._seal(group)
            group.ticket.discard()

    def _prepare(
        self, group: _Group, dispatched: list[EmbedJob]
    ) -> Callable[[], GroupResult]:
        """Run on the event loop after shared-lock acquisition, before dispatch."""
        dispatched.extend(job for job in group.jobs if not job._future.done())
        if not dispatched:
            raise asyncio.CancelledError
        if len(dispatched) == 1:
            job = dispatched[0]
            embed = partial(
                self._embedder.embed,
                job.image_url,
                job.image_base64,
                job.model,
                job.normalize,
                job.image_size,
            )
            return lambda: [embed()]

        from .embedder import BatchItem

        items = [
            BatchItem(job.image_url, job.image_base64, job.normalize)
            for job in dispatched
        ]
        return partial(self._embedder.embed_batch, group.spec, group.key[1], items)
