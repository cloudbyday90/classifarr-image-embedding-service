# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded forward passes; active calls finish before the next stop checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .execution_cancellation import cancellation_checkpoint

if TYPE_CHECKING:
    from .embedder import BatchItem, EmbedResult, ImageEmbedder
    from .model_catalog import ModelSpec


def infer_batch(
    embedder: ImageEmbedder,
    spec: ModelSpec,
    target_size: int,
    items: list[BatchItem],
    images: list[Image.Image],
) -> tuple[list[EmbedResult | Exception], str]:
    cancellation_checkpoint()
    model, processor, device = embedder._load_model(spec)
    outcomes: list[EmbedResult | Exception] = []
    size = embedder.settings.inference_batch_size
    for start in range(0, len(images), size):
        cancellation_checkpoint()
        chunk = images[start : start + size]
        chunk_items = items[start : start + size]
        inputs = processor(  # type: ignore[operator]
            images=chunk,
            return_tensors="np" if device.startswith("ov:") else "pt",
            size={"shortest_edge": target_size},
        )
        cancellation_checkpoint()
        if device.startswith("ov:"):
            raw = model(dict(inputs))  # type: ignore[operator]
            features = raw[0].astype(np.float32)
            if features.ndim != 2 or features.shape != (len(chunk), spec.dims):
                raise ValueError(
                    f"OpenVINO model returned unexpected output shape {features.shape!r} "
                    f"(expected ({len(chunk)}, {spec.dims})) for model={spec.name}"
                )
            for item, feature in zip(chunk_items, features, strict=True):
                try:
                    if item.normalize:
                        feature = embedder._normalize_embedding_np(feature)
                    outcomes.append(
                        _validated(embedder, spec, target_size, feature.tolist())
                    )
                except ValueError as error:
                    outcomes.append(error)
        else:
            import torch

            inputs = {key: value.to(device) for key, value in inputs.items()}
            cancellation_checkpoint()
            with torch.no_grad():
                features = model(**inputs).image_embeds  # type: ignore[operator]
            for position, item in enumerate(chunk_items):
                feature = features[position]
                if item.normalize:
                    feature = torch.nn.functional.normalize(
                        feature.unsqueeze(0), p=2, dim=-1
                    ).squeeze(0)
                vector = feature.detach().cpu().numpy().astype(np.float32).tolist()
                try:
                    outcomes.append(_validated(embedder, spec, target_size, vector))
                except ValueError as error:
                    outcomes.append(error)
    return outcomes, device


def _validated(
    embedder: ImageEmbedder,
    spec: ModelSpec,
    target_size: int,
    vector: Any,
) -> EmbedResult:
    embedder._validate_embedding_result(spec, vector, len(vector))
    return vector, len(vector), "local", spec.name, target_size
