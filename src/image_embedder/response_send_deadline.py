# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A total response budget with server-owned abort and final-drain callbacks."""

import logging
from collections.abc import Awaitable, Callable

import anyio
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .deadlines import positive_duration

TransportOperation = Callable[[], Awaitable[None]]
logger = logging.getLogger(__name__)


class ResponseSendDeadline:
    def __init__(
        self,
        app: ASGIApp,
        timeout_seconds: float,
        abort: TransportOperation,
        drain: TransportOperation,
    ) -> None:
        self.app = app
        self.timeout_seconds = positive_duration(
            timeout_seconds, "response_send_timeout_seconds"
        )
        self.abort = abort
        self.drain = drain

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started, completed, aborted = anyio.Event(), anyio.Event(), anyio.Event()
        deadline = 0.0
        expired = False
        trailers = False
        error: Exception | None = None

        with anyio.CancelScope() as request:

            async def expire() -> None:
                nonlocal expired
                if not expired:
                    expired = True
                    # Close before cancellation can release the downstream ingress owner.
                    with anyio.CancelScope(shield=True):
                        await self.abort()
                        aborted.set()
                    logger.warning(
                        "Response send deadline exceeded; connection aborted"
                    )
                    request.cancel()
                else:
                    with anyio.CancelScope(shield=True):
                        await aborted.wait()

            async def watch() -> None:
                await started.wait()
                with anyio.move_on_after(
                    max(0.0, deadline - anyio.current_time())
                ) as timer:
                    await completed.wait()
                if timer.cancel_called:
                    await expire()

            async def check_expiry() -> None:
                if (
                    started.is_set()
                    and not completed.is_set()
                    and anyio.current_time() >= deadline
                ):
                    await expire()
                if expired:
                    with anyio.CancelScope(shield=True):
                        await aborted.wait()
                    raise anyio.get_cancelled_exc_class()()

            async def limited_send(message: Message) -> None:
                nonlocal deadline, trailers
                await check_expiry()
                if message["type"] == "http.response.start" and not started.is_set():
                    deadline = anyio.current_time() + self.timeout_seconds
                    trailers = message.get("trailers", False)
                    started.set()
                try:
                    await send(message)
                    terminal = (
                        message["type"] == "http.response.body"
                        and not message.get("more_body", False)
                        and not trailers
                    ) or (
                        message["type"] == "http.response.trailers"
                        and not message.get("more_trailers", False)
                    )
                    if terminal:
                        await self.drain()
                    await check_expiry()
                    if terminal:
                        completed.set()
                except BaseException:
                    with anyio.CancelScope(shield=True):
                        await self.abort()
                    raise

            async with anyio.create_task_group() as group:
                group.start_soon(watch)
                try:
                    await self.app(scope, receive, limited_send)
                except anyio.get_cancelled_exc_class():
                    if not expired:
                        raise
                except Exception as exc:
                    # Preserve the original application error rather than an ExceptionGroup.
                    error = exc
                finally:
                    if started.is_set() and not completed.is_set():
                        with anyio.CancelScope(shield=True):
                            await self.abort()
                    group.cancel_scope.cancel()
        if error is not None:
            raise error
