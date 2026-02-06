# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import asyncio
from dataclasses import dataclass


class QueueFullError(RuntimeError):
    pass


class QueueWaitTimeoutError(TimeoutError):
    pass


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
        self._in_flight = 0
        self._waiting = 0

        self._rwlock = RWLock()

    async def acquire(self) -> None:
        async with self._cond:
            # Fast-path: available slot
            if self._in_flight < self._capacity:
                self._in_flight += 1
                return

            # No waiting allowed
            if self._max_queue == 0:
                raise QueueFullError("service is busy")

            # Waiting room full
            if self._waiting >= self._max_queue:
                raise QueueFullError("service is busy (queue full)")

            self._waiting += 1
            try:
                try:
                    await asyncio.wait_for(
                        self._cond.wait_for(lambda: self._in_flight < self._capacity),
                        timeout=self._max_wait_seconds,
                    )
                except TimeoutError as exc:
                    raise QueueWaitTimeoutError(
                        f"timed out waiting for a slot after {self._max_wait_seconds}s"
                    ) from exc

                self._in_flight += 1
            finally:
                self._waiting -= 1

    async def release(self) -> None:
        async with self._cond:
            if self._in_flight <= 0:
                raise RuntimeError("release called without a matching acquire")
            self._in_flight -= 1
            self._cond.notify(1)

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
            in_flight=self._in_flight,
            waiting=self._waiting,
            max_queue=self._max_queue,
            max_wait_seconds=self._max_wait_seconds,
            rw_readers=readers,
            rw_writer=writer,
            rw_writer_waiters=writer_waiters,
        )

