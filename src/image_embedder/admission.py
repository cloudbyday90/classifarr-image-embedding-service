# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Event-loop-owned FIFO admission with bounded, weighted waiting entries."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field


class QueueFullError(RuntimeError):
    pass


class QueueWaitTimeoutError(TimeoutError):
    pass


@dataclass(eq=False, slots=True)
class AdmissionTicket:
    """One computation permit, optionally shared by a bounded batch group.

    All mutations run on the application's event loop. Once claimed, only the
    inference owner may release the ticket; client departure only drops weight.
    """

    _pool: AdmissionPool
    _ready: asyncio.Future[None]
    members: int = 1
    admitted: bool = False
    claimed: bool = False
    finished: bool = False
    _deadline: float | None = None
    _timer: asyncio.TimerHandle | None = field(default=None, repr=False)

    async def wait(self) -> None:
        try:
            await self._ready
        except BaseException:
            self.discard()
            raise

    def add_member(self) -> None:
        self._pool.add_member(self)

    def remove_member(self) -> None:
        if self.members <= 0:
            raise RuntimeError("admission member released twice")
        self.members -= 1
        if not self.finished and not self.admitted:
            self._pool.waiting -= 1
        if self.members == 0:
            self.discard()

    def claim(self) -> None:
        if not self.admitted or self.finished or self.claimed:
            raise RuntimeError("admission ticket is not available for transfer")
        self.claimed = True

    def discard(self) -> None:
        if not self.claimed:
            self._pool.release(self)

    def release(self) -> None:
        self._pool.release(self)


class AdmissionPool:
    """One waiting budget shared by direct requests and batch-window groups."""

    def __init__(self, capacity: int, max_queue: int, max_wait_seconds: float):
        self.capacity = capacity
        self.max_queue = max_queue
        self.max_wait_seconds = max_wait_seconds
        self.in_flight = 0
        self.waiting = 0
        self._waiters: OrderedDict[AdmissionTicket, None] = OrderedDict()
        self._closed_error: tuple[type[Exception], tuple[object, ...]] | None = None

    def reserve(self) -> AdmissionTicket:
        if self._closed_error is not None:
            raise self._new_closed_error()
        loop = asyncio.get_running_loop()
        ticket = AdmissionTicket(self, loop.create_future())
        if self.in_flight < self.capacity and not self._waiters:
            self._admit(ticket)
        else:
            if self.waiting >= self.max_queue:
                raise QueueFullError("service is busy (queue full)")
            if self.max_wait_seconds <= 0:
                raise self._timeout_error()
            self.waiting += 1
            self._waiters[ticket] = None
            ticket._deadline = loop.time() + self.max_wait_seconds
            ticket._timer = loop.call_at(ticket._deadline, self._expire, ticket)
        return ticket

    def add_member(self, ticket: AdmissionTicket) -> None:
        if self._closed_error is not None:
            raise self._new_closed_error()
        if ticket.finished or ticket.claimed:
            raise RuntimeError("cannot join closed admission")
        if not ticket.admitted:
            if self.waiting >= self.max_queue:
                raise QueueFullError("service is busy (queue full)")
            self.waiting += 1
        ticket.members += 1

    def release(self, ticket: AdmissionTicket) -> None:
        # Abandoned entries must observe failures even if their owner coroutine
        # was canceled before it could begin awaiting the readiness future.
        if ticket._ready.done() and not ticket._ready.cancelled():
            ticket._ready.exception()
        if ticket.finished:
            return
        ticket.finished = True
        if ticket.admitted:
            self.in_flight -= 1
        else:
            self._waiters.pop(ticket, None)
            self.waiting -= ticket.members
        self._cancel_timer(ticket)
        if not ticket._ready.done():
            ticket._ready.cancel()
        self._promote()

    def close(self, error: Exception) -> None:
        if self._closed_error is not None:
            return
        # Retain the error contract, not a traceback that can accumulate caller
        # frames and payloads whenever shutdown rejects another submission.
        self._closed_error = (type(error), error.args)
        for ticket in tuple(self._waiters):
            self.waiting -= ticket.members
            ticket.finished = True
            self._cancel_timer(ticket)
            if not ticket._ready.done():
                ticket._ready.set_exception(self._new_closed_error())
        self._waiters.clear()

    def _new_closed_error(self) -> Exception:
        assert self._closed_error is not None
        error_type, arguments = self._closed_error
        return error_type(*arguments)

    def _admit(self, ticket: AdmissionTicket) -> None:
        self._cancel_timer(ticket)
        ticket.admitted = True
        self.in_flight += 1
        ticket._ready.set_result(None)

    def _promote(self) -> None:
        while self.in_flight < self.capacity and self._waiters:
            ticket, _ = self._waiters.popitem(last=False)
            self.waiting -= ticket.members
            self._cancel_timer(ticket)
            if ticket._ready.cancelled():
                ticket.finished = True
                continue
            if (
                ticket._deadline is not None
                and asyncio.get_running_loop().time() >= ticket._deadline
            ):
                ticket.finished = True
                ticket._ready.set_exception(self._timeout_error())
                continue
            self._admit(ticket)

    def _expire(self, ticket: AdmissionTicket) -> None:
        if ticket.finished or ticket.admitted:
            return
        self._waiters.pop(ticket)
        self.waiting -= ticket.members
        ticket.finished = True
        ticket._timer = None
        if not ticket._ready.done():
            ticket._ready.set_exception(self._timeout_error())

    def _timeout_error(self) -> QueueWaitTimeoutError:
        return QueueWaitTimeoutError(
            f"timed out waiting for a slot after {self.max_wait_seconds}s"
        )

    @staticmethod
    def _cancel_timer(ticket: AdmissionTicket) -> None:
        if ticket._timer is not None:
            ticket._timer.cancel()
            ticket._timer = None
