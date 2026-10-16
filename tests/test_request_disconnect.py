# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Receive ownership, result races and cancellation cleanup for parsed requests."""

import asyncio

import anyio
import pytest

from image_embedder.request_disconnect import RequestDisconnected, await_connected


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [False, True])
async def test_ready_operation_wins_simultaneous_disconnect_and_settles_watcher(failure):
    closed = []

    async def receive():
        try:
            return {"type": "http.disconnect"}
        finally:
            closed.append(True)

    async def operation():
        if failure:
            raise ValueError("operation failure")
        return 42

    if failure:
        with pytest.raises(ValueError, match="operation failure"):
            await await_connected(receive, operation)
    else:
        assert await await_connected(receive, operation) == 42
    assert closed == [True]


@pytest.mark.anyio
@pytest.mark.parametrize("message", ["http.disconnect", "http.request", "failure"])
async def test_receive_completion_cancels_and_settles_waiter(message):
    started, settled = asyncio.Event(), asyncio.Event()

    async def receive():
        await started.wait()
        if message == "failure":
            raise OSError("controlled receive failure")
        return {"type": message}

    async def operation():
        try:
            started.set()
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            settled.set()

    error = {"http.disconnect": RequestDisconnected, "http.request": RuntimeError, "failure": OSError}[message]
    with pytest.raises(error):
        await await_connected(receive, operation)
    assert settled.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize("cancellation", ["asyncio", "anyio", "timeout"])
async def test_parent_cancellation_settles_both_helpers(cancellation):
    started, settled = asyncio.Event(), []

    async def receive():
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            settled.append("receive")

    async def operation():
        try:
            started.set()
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            settled.append("operation")

    if cancellation == "anyio":
        with anyio.CancelScope() as scope:
            async def cancel():
                await started.wait()
                scope.cancel()
            async with anyio.create_task_group() as group:
                group.start_soon(cancel)
                await await_connected(receive, operation)
    else:
        task = asyncio.create_task(await_connected(receive, operation))
        await started.wait()
        if cancellation == "timeout":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.01)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert sorted(settled) == ["operation", "receive"]


@pytest.mark.anyio
async def test_repeated_parent_cancellation_cannot_abandon_helper_cleanup():
    started, cleaning, finish, settled = (asyncio.Event() for _ in range(4))

    async def receive():
        await asyncio.Future()

    async def operation():
        try:
            started.set()
            await asyncio.Future()
        finally:
            cleaning.set()
            await finish.wait()
            settled.set()

    task = asyncio.create_task(await_connected(receive, operation))
    await started.wait()
    task.cancel()
    await cleaning.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set()
