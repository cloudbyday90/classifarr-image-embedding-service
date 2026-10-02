# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Allocation-free inline checks and accounting owned by one inference batch."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .config import Settings


class InputLimitExceeded(ValueError):
    """An image input exceeds a configured resource ceiling."""


def base64_size(value: str, max_bytes: int) -> int:
    """Bound the ASCII copy/decoder allocation before strict base64 validation.

    For valid padded base64 the estimate is exact. Malformed inputs remain the
    decoder's responsibility; extra padding cannot bypass the encoded ceiling.
    """
    if len(value) > 4 * ((max_bytes + 2) // 3):
        raise InputLimitExceeded("Image payload exceeds maximum size")
    padding = int(value.endswith("=")) + int(value.endswith("=="))
    size = max(0, ((len(value) + 3) // 4) * 3 - padding)
    if size > max_bytes:
        raise InputLimitExceeded("Image payload exceeds maximum size")
    return size


def validate_inline_images(
    values: Iterable[str | None], settings: Settings, *, batch: bool = False
) -> None:
    """Reject known oversized payloads before retaining a queue/batch job."""
    total = 0
    for value in values:
        if value is not None:
            total += base64_size(value, settings.max_image_bytes)
            if batch and total > settings.max_batch_image_bytes:
                raise InputLimitExceeded(
                    "Batch image payloads exceed maximum aggregate size"
                )


@dataclass(slots=True)
class BatchInputBudget:
    """Local to one synchronous batch; rejection never consumes capacity."""

    max_bytes: int
    max_pixels: int
    used_bytes: int = field(default=0, init=False)
    used_pixels: int = field(default=0, init=False)

    def add_bytes(self, size: int) -> None:
        if self.used_bytes + size > self.max_bytes:
            raise InputLimitExceeded(
                "Batch image payloads exceed maximum aggregate size"
            )
        self.used_bytes += size

    def add_pixels(self, pixels: int) -> None:
        if self.used_pixels + pixels > self.max_pixels:
            raise InputLimitExceeded(
                "Batch images exceed maximum aggregate pixel budget"
            )
        self.used_pixels += pixels

    def release_pixels(self, pixels: int) -> None:
        self.used_pixels -= pixels
