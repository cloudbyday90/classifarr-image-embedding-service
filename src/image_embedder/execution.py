# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own inference permits independently of the HTTP task awaiting a result."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, ParamSpec, TypeVar, cast

from anyio.to_thread import run_sync

from .admission import AdmissionTicket
from .execution_observation import (
    ExecutionObserver,
    ExecutionTrace,
    begin_observation,
    mark_observation,
)
from .logging_config import get_logger
from .queue import EmbedQueue

P = ParamSpec("P")
T = TypeVar("T")
logger = get_logger(__name__)


class ExecutionClosedError(RuntimeError):
    """The application has stopped accepting inference work."""


@dataclass(slots=True)
class _Work:
    dispatched: bool = False
    detached: bool = False
    trace: ExecutionTrace | None = None


@dataclass(frozen=True, slots=True)
class _Outcome(Generic[T]):
    value: T | None = None
    error: Exception | None = None

    def unwrap(self) -> T:
        if self.error is not None:
            raise self.error
        return cast(T, self.value)


class InferenceExecutor:
    """Track admission and retain capacity until synchronous work completes.

    Request cancellation cancels admission, but never a dispatched owner task.
    Call ``close`` during application teardown; a drain timeout deliberately
    leaves running owners and their queue permits intact.
    """

    def __init__(self, queue: EmbedQueue, *, observer: ExecutionObserver | None = None) -> None:
        self._queue = queue
        self._observer = observer
        self._tasks: dict[asyncio.Task[_Outcome[Any]], _Work] = {}
        self._closing = False

    async def run(
        self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return await self._submit(partial(function, *args, **kwargs))

    async def run_admitted(
        self, ticket: AdmissionTicket, prepare: Callable[[], Callable[[], T]]
    ) -> T:
        """Transfer reserved admission; prepare live payloads just before dispatch."""
        if not self._queue.owns(ticket):
            raise ValueError("admission ticket belongs to another queue")
        return await self._submit(prepare, ticket)

    async def _submit(
        self, function: Callable[[], Any], ticket: AdmissionTicket | None = None
    ) -> Any:
        if self._closing:
            raise ExecutionClosedError("service is shutting down")

        work = _Work(trace=begin_observation(self._observer))
        task = asyncio.create_task(
            self._execute(work, function, ticket),
            name="inference-owner",
        )
        self._tasks[task] = work
        task.add_done_callback(self._completed)

        try:
            outcome = await asyncio.shield(task)
            return outcome.unwrap()
        except asyncio.CancelledError as canceled:
            requester = asyncio.current_task()
            if (
                self._closing
                and not work.dispatched
                and requester is not None
                and not requester.cancelling()
            ):
                # close can cancel an owner before its coroutine ever starts,
                # bypassing _execute's cancellation-to-outcome conversion.
                raise ExecutionClosedError("service is shutting down") from canceled
            work.detached = True
            mark_observation(work.trace, "waiter_detached")
            if not work.dispatched:
                self._cancel_admission(task)
                # Settle admission and its partial acquisitions before the
                # request returns. Repeated caller cancellation cannot cancel
                # this cleanup through the shield.
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
            raise

    async def _execute(
        self, work: _Work, function: Callable[[], Any], ticket: AdmissionTicket | None
    ) -> _Outcome[Any]:
        # Ordinary failures travel as values until an attached caller unwraps
        # them. Python 3.14 shield otherwise reports detached exceptions to the
        # loop even when a separate done callback has retrieved the exception.
        try:
            return _Outcome(value=await self._run_work(work, function, ticket))
        except asyncio.CancelledError:
            if self._closing and not work.dispatched:
                return _Outcome(error=ExecutionClosedError("service is shutting down"))
            raise
        except Exception as error:
            return _Outcome(error=error)

    async def _run_work(
        self, work: _Work, function: Callable[[], Any], ticket: AdmissionTicket | None
    ) -> Any:
        acquired = False
        shared = False
        try:
            mark_observation(work.trace, "admission_started")
            if ticket is None:
                await self._queue.acquire()
            else:
                await ticket.wait()
                ticket.claim()
            acquired = True
            mark_observation(work.trace, "admitted")
            await self._queue.acquire_shared()
            shared = True
            mark_observation(work.trace, "read_lock_acquired")
            if ticket is not None:
                function = function()
            # No suspension between committing dispatch and scheduling work.
            # After this point only this owner may release the permits.
            work.dispatched = True
            mark_observation(work.trace, "dispatched")
            if work.trace is None:
                return await run_sync(function)

            def observed_call():
                mark_observation(work.trace, "native_started")
                try:
                    return function()
                except BaseException:
                    mark_observation(work.trace, "native_failed")
                    raise
                finally:
                    mark_observation(work.trace, "native_finished")

            return await run_sync(observed_call)
        finally:
            if shared:
                await self._queue.release_shared()
            if acquired:
                if ticket is None:
                    await self._queue.release()
                else:
                    await self._queue.release_admission(ticket)

    def _completed(self, task: asyncio.Task[_Outcome[Any]]) -> None:
        work = self._tasks.pop(task)
        if task.cancelled():
            mark_observation(work.trace, "owner_cancelled")
        if not task.cancelled():
            error = task.result().error
            if error is not None:
                mark_observation(work.trace, "owner_failed")
            if error is not None and work.detached:
                # Do not include callable arguments or potentially sensitive
                # exception text from remote requests in detached-work logs.
                logger.warning("Detached inference failed (%s)", type(error).__name__)
        mark_observation(work.trace, "owner_completed")

    @staticmethod
    def _cancel_admission(task: asyncio.Task[_Outcome[Any]]) -> None:
        # A second cancellation can interrupt partial-permit cleanup while it
        # waits on a contended condition. Caller expiry and close may overlap.
        if not task.cancelling():
            task.cancel()

    async def close(self, timeout_seconds: float) -> bool:
        """Reject new work, cancel admission, and drain without canceling threads.

        Return False if dispatched work outlives the drain budget. Callers must
        not free its model/GPU resources until those owners finish.
        """
        self._closing = True
        self._queue.close_admission(ExecutionClosedError("service is shutting down"))
        tasks = tuple(self._tasks)
        for task in tasks:
            if not self._tasks[task].dispatched:
                self._cancel_admission(task)
        if not tasks:
            return True
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_seconds))
        return not pending
