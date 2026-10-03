# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Narrow adapter for the locked Uvicorn HTTP/1 transport contract."""

import asyncio
from typing import Any

import anyio
from starlette.types import ASGIApp
from uvicorn.protocols.http.auto import AutoHTTPProtocol
from uvicorn.protocols.http.flow_control import FlowControl
from uvicorn.protocols.http.h11_impl import H11Protocol

from .response_send_deadline import ResponseSendDeadline


class _ResponseTransport(asyncio.Protocol):
    app: ASGIApp
    transport: asyncio.Transport
    flow: FlowControl

    def __init__(self, *args: Any, timeout_seconds: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._closed = asyncio.Event()
        self.app = ResponseSendDeadline(
            self.app, timeout_seconds, self._abort, self._drain
        )

    async def _abort(self) -> None:
        if not self._closed.is_set():
            self.transport.abort()
            await self._closed.wait()

    async def _drain(self) -> None:
        # A final Uvicorn write need not cross its high watermark before returning.
        # Low-water resumption is also not equivalent to an empty output buffer.
        while not self._closed.is_set() and self._pending_write_bytes():
            if self.flow.write_paused:
                await self.flow.drain()
            else:
                await anyio.sleep(0.01)

    def _pending_write_bytes(self) -> int:
        if self._closed.is_set():
            return 0
        pending = self.transport.get_write_buffer_size()
        # CPython's SSL transport excludes ciphertext already queued on its
        # underlying socket. Opaque TLS transports cannot establish final drain.
        ssl_protocol = getattr(self.transport, "_ssl_protocol", None)
        underlying = getattr(ssl_protocol, "_transport", None)
        if underlying is not None:
            pending += underlying.get_write_buffer_size()
        elif (
            not self._closed.is_set()
            and self.transport.get_extra_info("ssl_object") is not None
        ):
            raise RuntimeError(
                "Response deadlines require an observable TLS socket buffer; use the asyncio event loop"
            )
        return pending

    def connection_lost(self, exc: Exception | None) -> None:
        try:
            super().connection_lost(exc)
        finally:
            self._closed.set()


class ResponseDeadlineHTTPProtocol(_ResponseTransport, AutoHTTPProtocol):
    """Keep Uvicorn's httptools selection, falling back to h11 when unavailable."""


class ResponseDeadlineH11Protocol(_ResponseTransport, H11Protocol):
    """Explicit h11 adapter for deployments choosing the pure Python parser."""
