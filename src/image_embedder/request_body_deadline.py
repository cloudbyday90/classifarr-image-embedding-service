# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A total cooperative upload budget, disabled after body or response completion."""

import math

import anyio
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .deadlines import positive_duration


class RequestBodyDeadlineMiddleware:
    def __init__(self, app: ASGIApp, timeout_seconds: float) -> None:
        self.app = app
        self.timeout_seconds = positive_duration(
            timeout_seconds, "request_body_timeout_seconds"
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False
        with anyio.CancelScope(
            deadline=anyio.current_time() + self.timeout_seconds
        ) as budget:

            async def limited_receive() -> Message:
                message = await receive()
                if message["type"] == "http.disconnect" or (
                    message["type"] == "http.request"
                    and not message.get("more_body", False)
                ):
                    budget.deadline = math.inf
                return message

            async def limited_send(message: Message) -> None:
                nonlocal started
                if budget.cancel_called:
                    raise anyio.get_cancelled_exc_class()()
                if message["type"] == "http.response.start":
                    started = True
                    budget.deadline = math.inf
                await send(message)

            await self.app(scope, limited_receive, limited_send)
        if budget.cancel_called and not started:
            headers = (
                {"Connection": "close"}
                if scope.get("http_version") in {"1.0", "1.1"}
                else None
            )
            await JSONResponse(
                {"detail": "Request body timed out"}, status_code=408, headers=headers
            )(scope, receive, send)
