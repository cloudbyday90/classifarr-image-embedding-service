# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify pinned publisher assets before loading models or preprocessing."""

import re
from pathlib import Path

from .artifact_integrity import sha256_file
from .model_catalog import ModelSpec
from .model_initialization import serialized_initialization


def verified_asset(spec: ModelSpec, filename: str) -> Path:
    digest = spec.asset_digest(filename)
    if not re.fullmatch(r"[0-9a-f]{40}", spec.revision) or not re.fullmatch(
        r"[0-9a-f]{64}", digest
    ):
        raise ValueError(f"Unverified model identity: {spec.name}")
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=spec.hf_id,
            filename=filename,
            revision=spec.revision,
            token=False,
        )
    )
    if sha256_file(path) != digest:
        raise ValueError(f"Model asset digest mismatch: {spec.name}/{filename}")
    return path


@serialized_initialization
def load_processor(spec: ModelSpec):
    from transformers import CLIPImageProcessorPil

    path = verified_asset(spec, "preprocessor_config.json")
    return CLIPImageProcessorPil.from_pretrained(
        str(path.parent), local_files_only=True
    )


@serialized_initialization
def load_vision_model(spec: ModelSpec):
    from transformers import CLIPVisionModelWithProjection

    config = verified_asset(spec, "config.json")
    weights = verified_asset(spec, spec.weights_file)
    if config.parent != weights.parent:
        raise ValueError(f"Model assets are not in the same snapshot: {spec.name}")
    model = CLIPVisionModelWithProjection.from_pretrained(
        str(config.parent),
        local_files_only=True,
        use_safetensors=spec.weights_file == "model.safetensors",
        weights_only=True,
        trust_remote_code=False,
    )
    if (
        model.config.projection_dim != spec.dims
        or model.config.image_size != spec.image_size
    ):
        raise ValueError(
            f"Model dimensions do not match the pinned contract: {spec.name}"
        )
    model.eval()
    return model
