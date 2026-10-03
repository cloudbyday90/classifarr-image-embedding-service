# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Probe-only ASGI receive counters; never retain image bytes or credentials."""

from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass
class ReceiveRecord:
    bytes_received: int = 0
    disconnected: bool = False
    completed: bool = False
    status: int | None = None


class ReceiveObserver:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.records: dict[str, ReceiveRecord] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        label = next(
            (
                value.decode("ascii")
                for key, value in scope["headers"]
                if key == b"x-capacity-id"
            ),
            None,
        )
        if label is None:
            await self.app(scope, receive, send)
            return
        if label in self.records:
            raise ValueError("Duplicate capacity request label")
        record = self.records[label] = ReceiveRecord()

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.request":
                record.bytes_received += len(message.get("body", b""))
            elif message["type"] == "http.disconnect":
                record.disconnected = True
            return message

        async def observed_send(message):
            if message["type"] == "http.response.start":
                record.status = message["status"]
            await send(message)

        try:
            await self.app(scope, observed_receive, observed_send)
        finally:
            record.completed = True
