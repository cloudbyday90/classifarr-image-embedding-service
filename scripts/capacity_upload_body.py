# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Repeatable, bounded upload fixtures without per-client full-body copies."""

import asyncio
import base64
import json
import math
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from image_embedder.config import Settings


def validate_socket_limits(settings: Settings) -> None:
    if not (
        1 <= settings.max_http_requests <= 16
        and 1 <= settings.max_image_bytes <= 16 * 1024**2
        and 1 <= settings.max_image_pixels <= 32_000_000
        and 1 <= settings.max_request_body_bytes <= 32 * 1024**2
        and 2 <= settings.embed_batch_api_max_items <= 32
        and settings.server_concurrency > settings.max_http_requests + 2
    ):
        raise ValueError("Socket experiment exceeds bounded fixture/connection limits")


def ceiling_image(settings: Settings) -> tuple[str, int]:
    edge = math.isqrt(settings.max_image_pixels)
    with Image.new("RGB", (edge, edge), (17, 33, 77)) as image, BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    if len(data) > settings.max_image_bytes:
        raise ValueError("Pixel fixture exceeds the configured image byte ceiling")
    padded = data + b"\0" * (settings.max_image_bytes - len(data))
    return base64.b64encode(padded).decode("ascii"), edge


@dataclass(frozen=True)
class UploadBody:
    encoded_json: bytes
    length: int

    @classmethod
    def create(cls, request: dict, length: int) -> "UploadBody":
        encoded = json.dumps(request, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > length:
            raise ValueError("Valid JSON exceeds the configured request body ceiling")
        return cls(encoded, length)

    def chunks(self, stop: int, chunk_size: int = 65536) -> Iterator[bytes]:
        if not 0 <= stop <= self.length or chunk_size <= 0:
            raise ValueError("Invalid bounded upload range")
        prefix_end = min(stop, len(self.encoded_json))
        for offset in range(0, prefix_end, chunk_size):
            yield self.encoded_json[offset : min(offset + chunk_size, prefix_end)]
        remaining = stop - prefix_end
        padding = b" " * min(remaining, chunk_size)
        while remaining:
            amount = min(remaining, chunk_size)
            yield padding[:amount]
            remaining -= amount

    async def stream(
        self, release: asyncio.Event | None = None
    ) -> AsyncIterator[bytes]:
        for chunk in self.chunks(self.length - 1):
            yield chunk
        if release is not None:
            await release.wait()
        yield b" " if self.length > len(self.encoded_json) else self.encoded_json[-1:]
