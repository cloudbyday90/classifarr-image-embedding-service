# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixed, encoded poster/landscape fixtures for the existing capacity probe."""

import base64
import hashlib
import json
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, features


@dataclass(frozen=True)
class ImageFixture:
    name: str
    payloads: tuple[str, str]
    metadata: dict


def make_fixtures() -> list[ImageFixture]:
    fixtures = []
    for name, width, height, mode, codec, options in (
        ("poster-jpeg", 500, 750, "RGB", "JPEG", {"quality": 85, "progressive": True}),
        ("poster-png", 500, 750, "RGBA", "PNG", {"compress_level": 6}),
        ("landscape-webp", 750, 500, "RGB", "WEBP", {"quality": 85, "method": 4}),
        ("noise-png", 224, 224, "RGB", "PNG", {"compress_level": 6}),
    ):
        payloads, variants = [], []
        y, x = np.indices((height, width))
        for variant in range(2):
            if name == "noise-png":
                pixels = np.random.default_rng(20261003 + variant).integers(
                    0, 256, (height, width, 3), dtype=np.uint8
                )
            else:
                channels = [
                    (x * 7 + variant * 31) % 256,
                    (y * 11) % 256,
                    ((x + y) * 3) % 256,
                ]
                if mode == "RGBA":
                    channels.append((x + y + variant * 17) % 256)
                pixels = np.stack(channels, axis=-1).astype(np.uint8)
            with Image.fromarray(pixels) as image, BytesIO() as output:
                image.save(output, format=codec, **options)
                encoded = output.getvalue()
            with Image.open(BytesIO(encoded)) as decoded:
                decoded.load()
                if (
                    decoded.size != (width, height)
                    or decoded.mode != mode
                    or decoded.format != codec
                ):
                    raise AssertionError("Encoded fixture changed shape, mode or codec")
            payloads.append(base64.b64encode(encoded).decode("ascii"))
            variants.append(
                {
                    "variant": variant,
                    "encoded_bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        fixtures.append(
            ImageFixture(
                name,
                (payloads[0], payloads[1]),
                {
                    "name": name,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "codec": codec,
                    "save_options": options,
                    "variants": variants,
                },
            )
        )
    return fixtures


def codec_versions() -> dict:
    return {name: features.version(name) for name in ("jpg", "zlib", "webp")}


def batch_body(
    fixture: ImageFixture, model: str, size: int, normalize: bool = False
) -> bytes:
    return json.dumps(
        {
            "model": model,
            "normalize": normalize,
            "items": [
                {"image_base64": fixture.payloads[index % 2]} for index in range(size)
            ],
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def validate_fixtures(fixtures, settings, models, sizes, clients, repeats) -> None:
    """Validate every allocation ceiling before creating a model owner."""
    from image_embedder.input_limits import validate_inline_images

    if not 1 <= clients <= min(8, settings.max_http_requests):
        raise ValueError("Representative clients must fit 1..8 and HTTP admission")
    if not 1 <= repeats <= 2 or not sizes or min(sizes) < 1 or max(sizes) > 32:
        raise ValueError("Representative repeats/batches exceed experiment bounds")
    for fixture in fixtures:
        pixels = fixture.metadata["width"] * fixture.metadata["height"]
        if pixels > settings.max_image_pixels:
            raise ValueError("Representative fixture exceeds image pixel ceiling")
        for model in models:
            if (
                max(sizes) * (pixels + model.image_size**2)
                > settings.max_batch_image_pixels
            ):
                raise ValueError("Representative fixture exceeds batch pixel ceiling")
            for size in sizes:
                validate_inline_images(
                    (fixture.payloads[index % 2] for index in range(size)),
                    settings,
                    batch=True,
                )
                if (
                    len(batch_body(fixture, model.name, size))
                    > settings.max_request_body_bytes
                ):
                    raise ValueError(
                        "Representative fixture exceeds request body ceiling"
                    )
