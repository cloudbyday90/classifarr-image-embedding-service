# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strict base64 decoding and dimension checks before Pillow allocation."""

import base64
import binascii
import io

from PIL import Image

from .input_limits import BatchInputBudget, InputLimitExceeded, base64_size


def decode_base64(value: str, max_bytes: int) -> bytes:
    base64_size(value, max_bytes)
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    if len(data) > max_bytes:
        raise InputLimitExceeded("Image payload exceeds maximum size")
    return data


def image_pixel_cost(
    size: tuple[int, int], target_size: int | None, max_pixels: int
) -> int:
    """Charge source and CLIP shortest-edge resize, before center cropping.

    Ceiling integer arithmetic conservatively covers the processor's rounding.
    """
    width, height = size
    if width <= 0 or height <= 0 or (target_size is not None and target_size <= 0):
        raise ValueError("Image dimensions and target size must be positive")
    source = width * height
    if source > max_pixels:
        raise InputLimitExceeded("Image exceeds maximum pixel count")
    resized = 0
    if target_size is not None:
        shorter, longer = sorted(size)
        resized = target_size * ((longer * target_size + shorter - 1) // shorter)
        if resized > max_pixels:
            raise InputLimitExceeded("Image resize exceeds maximum pixel count")
    return source + resized


def load_rgb(
    data: bytes,
    max_pixels: int,
    *,
    target_size: int | None = None,
    budget: BatchInputBudget | None = None,
) -> Image.Image:
    """Read only dimensions before reserving pixels and loading the first frame.

    Keep Pillow's own decompression-bomb defenses; never change global policy.
    The caller owns and must close the returned, detached RGB image.
    """
    reserved = 0
    try:
        with Image.open(io.BytesIO(data)) as source:
            cost = image_pixel_cost(source.size, target_size, max_pixels)
            if budget is not None:
                budget.add_pixels(cost)
                reserved = cost
            return source.convert("RGB")
    except Exception as exc:
        if reserved and budget is not None:
            budget.release_pixels(reserved)
        if isinstance(exc, InputLimitExceeded):
            raise
        if isinstance(exc, Image.DecompressionBombError):
            raise InputLimitExceeded("Image exceeds Pillow safety limit") from exc
        raise ValueError("Unable to decode image bytes") from exc
