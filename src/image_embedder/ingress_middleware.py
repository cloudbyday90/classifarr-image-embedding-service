# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reject excess HTTP work before downstream body receive or parsing."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .ingress import IngressAdmission


class IngressAdmissionMiddleware:
    def __init__(self, app: ASGIApp, admission: IngressAdmission) -> None:
        self.app = app
        self.admission = admission

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (
            scope["method"] in {"GET", "HEAD"}
            and scope["path"] in {"/health", "/ready"}
        ):
            await self.app(scope, receive, send)
            return
        if not self.admission.try_acquire():
            headers = {"Retry-After": "1"}
            if scope.get("http_version") in {"1.0", "1.1"}:
                headers["Connection"] = "close"
            response = JSONResponse(
                {"detail": "HTTP ingress capacity exceeded"},
                status_code=503,
                headers=headers,
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.admission.release()
