# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Watch one parsed request only while its embedding result is awaited."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio
from starlette.types import Receive

T = TypeVar("T")


class RequestDisconnected(Exception):
    """The caller left before its embedding waiter completed."""


async def await_connected(receive: Receive, operation: Callable[[], Awaitable[T]]) -> T:
    """Require a completely consumed body and exclusive receive ownership.

    Cancel the waiter, never its separately tracked inference owner. Both helper
    tasks are settled before returning to response serialization or unwinding.
    """
    async def watch() -> None:
        message = await receive()
        if message["type"] != "http.disconnect":
            raise RuntimeError("Expected disconnect after the complete request body")

    async def run() -> T:
        return await operation()

    watcher = asyncio.create_task(watch(), name="embedding-disconnect")
    waiter = asyncio.create_task(run(), name="embedding-waiter")
    try:
        await asyncio.wait((waiter, watcher), return_when=asyncio.FIRST_COMPLETED)
        # A ready result (including its error) wins a simultaneous close. No
        # receive task survives into send, where completion also means disconnect.
        if waiter.done():
            return waiter.result()
        watcher.result()
        raise RequestDisconnected()
    finally:
        for task in (waiter, watcher):
            if not task.done():
                task.cancel()
        # Parent AnyIO cancellation must not interrupt waiter/admission cleanup.
        # InferenceExecutor retains dispatched owners independently of this wait.
        with anyio.CancelScope(shield=True):
            settled = asyncio.gather(waiter, watcher, return_exceptions=True)
            interrupted = None
            while not settled.done():
                try:
                    await asyncio.shield(settled)
                except asyncio.CancelledError as exc:
                    interrupted = exc
            if interrupted is not None:
                raise interrupted
