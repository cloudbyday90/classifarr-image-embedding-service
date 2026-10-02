# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from .admission import (
    AdmissionPool,
    AdmissionTicket,
    QueueFullError,
    QueueWaitTimeoutError,
)

__all__ = ["EmbedQueue", "QueueStats", "QueueFullError", "QueueWaitTimeoutError", "RWLock"]


@dataclass(frozen=True)
class QueueStats:
    concurrency: int
    in_flight: int
    waiting: int
    max_queue: int
    max_wait_seconds: int
    rw_readers: int
    rw_writer: bool
    rw_writer_waiters: int


class RWLock:
    """
    Async read/write lock with writer preference.

    Writer preference matters for "exclusive" operations (future: model load/update):
    if an exclusive operation starts waiting, new readers should queue behind it.
    """

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._writer_waiters = 0

    async def acquire_shared(self) -> None:
        async with self._cond:
            while self._writer or self._writer_waiters > 0:
                await self._cond.wait()
            self._readers += 1

    async def release_shared(self) -> None:
        async with self._cond:
            if self._readers <= 0:
                raise RuntimeError("RWLock: release_shared called without a matching acquire_shared")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_exclusive(self) -> None:
        async with self._cond:
            self._writer_waiters += 1
            try:
                while self._writer or self._readers > 0:
                    await self._cond.wait()
                self._writer = True
            finally:
                self._writer_waiters -= 1

    async def release_exclusive(self) -> None:
        async with self._cond:
            if not self._writer:
                raise RuntimeError("RWLock: release_exclusive called without a matching acquire_exclusive")
            self._writer = False
            self._cond.notify_all()

    def stats(self) -> tuple[int, bool, int]:
        return self._readers, self._writer, self._writer_waiters


class EmbedQueue:
    """
    Concurrency limiter + bounded waiting room for embedding work.

    Semantics:
    - concurrency: maximum concurrent in-flight embed computations.
    - max_queue: maximum number of requests allowed to wait for a slot.
      max_queue = 0 means "no waiting allowed" (fail fast with 429 mapping).
    - max_wait_seconds: maximum time a request is allowed to wait for a slot.
    """

    def __init__(self, concurrency: int, max_queue: int, max_wait_seconds: int) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be >= 1")
        if max_queue < 0:
            raise ValueError("max_queue must be >= 0")
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be >= 0")

        self._capacity = concurrency
        self._max_queue = max_queue
        self._max_wait_seconds = max_wait_seconds

        self._cond = asyncio.Condition()
        self._admission = AdmissionPool(concurrency, max_queue, max_wait_seconds)
        self._acquired: deque[AdmissionTicket] = deque()

        self._rwlock = RWLock()

    async def acquire(self) -> None:
        ticket = self.reserve()
        await ticket.wait()
        ticket.claim()
        self._acquired.append(ticket)

    def reserve(self) -> AdmissionTicket:
        """Reserve before retaining a batch job; callers must transfer or discard."""
        return self._admission.reserve()

    def owns(self, ticket: AdmissionTicket) -> bool:
        return ticket._pool is self._admission

    def close_admission(self, error: Exception) -> None:
        self._admission.close(error)

    async def release_admission(self, ticket: AdmissionTicket) -> None:
        if not self.owns(ticket):
            raise RuntimeError("admission ticket belongs to another queue")
        async with self._cond:
            ticket.release()

    async def release(self) -> None:
        async with self._cond:
            if not self._acquired:
                raise RuntimeError("release called without a matching acquire")
            self._acquired.popleft().release()

    async def acquire_shared(self) -> None:
        await self._rwlock.acquire_shared()

    async def release_shared(self) -> None:
        await self._rwlock.release_shared()

    async def acquire_exclusive(self) -> None:
        await self._rwlock.acquire_exclusive()

    async def release_exclusive(self) -> None:
        await self._rwlock.release_exclusive()

    def stats(self) -> QueueStats:
        readers, writer, writer_waiters = self._rwlock.stats()
        return QueueStats(
            concurrency=self._capacity,
            in_flight=self._admission.in_flight,
            waiting=self._admission.waiting,
            max_queue=self._max_queue,
            max_wait_seconds=self._max_wait_seconds,
            rw_readers=readers,
            rw_writer=writer,
            rw_writer_waiters=writer_waiters,
        )

