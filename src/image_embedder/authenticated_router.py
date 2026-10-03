# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Authenticate matched protected routes before FastAPI reads their body."""

from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.routing import APIRoute

AuthDependency = Callable[[Request, str | None, str | None], Awaitable[None]]


def authenticated_router(auth: AuthDependency) -> APIRouter:
    class AuthenticatedRoute(APIRoute):
        def get_route_handler(
            self,
        ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            downstream = super().get_route_handler()

            async def handle(request: Request) -> Response:
                try:
                    await auth(
                        request,
                        request.headers.get("x-api-key"),
                        request.headers.get("authorization"),
                    )
                except HTTPException as exc:
                    # An unread HTTP/1 body leaves no usable keep-alive connection.
                    headers = dict(exc.headers or {})
                    if request.scope.get("http_version") in {"1.0", "1.1"}:
                        headers["Connection"] = "close"
                    raise HTTPException(exc.status_code, exc.detail, headers) from None
                return await downstream(request)

            return handle

    # Retain the dependency for schema security and ordinary validation. Both
    # stages call the same policy; no mutable request-state bypass is introduced.
    return APIRouter(route_class=AuthenticatedRoute, dependencies=[Depends(auth)])
