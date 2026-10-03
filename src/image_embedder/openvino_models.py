# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenVINO export/reload ownership with explicit preprocessing and precision."""

import os
from importlib.metadata import version
from pathlib import Path

from .ir_cache import IRCache
from .model_catalog import ModelSpec
from .model_initialization import serialized_initialization
from .model_loading import load_processor, load_vision_model


def ir_contract(spec: ModelSpec) -> dict:
    return {
        "schema_version": 1,
        "source": {
            "repository": spec.hf_id,
            "revision": spec.revision,
            "assets": dict(spec.assets),
        },
        "shape": {"image_size": spec.image_size, "projection_dim": spec.dims},
        "preprocessing": {"class": "CLIPImageProcessorPil", "policy_version": 1},
        "runtime": {
            name: version(name) for name in ("torch", "transformers", "openvino")
        },
        "export": {"format": "openvino_ir", "compress_to_fp16": False},
        "inference": {"execution_mode": "ACCURACY"},
    }


@serialized_initialization
def load_openvino_model(spec: ModelSpec, device: str):
    import openvino as ov

    processor = load_processor(spec)
    cache = IRCache(Path(os.environ.get("OV_MODEL_CACHE", "/app/.cache/ov_ir")))

    def export(xml_path: Path) -> None:
        import torch

        model = load_vision_model(spec)
        inputs = {"pixel_values": torch.zeros(1, 3, spec.image_size, spec.image_size)}
        converted = ov.convert_model(model, example_input=inputs)
        ov.save_model(converted, str(xml_path), compress_to_fp16=False)

    xml_path = cache.get_or_create(ir_contract(spec), export)
    core = ov.Core()
    # Compile the published representation on first load as well as every reload.
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(
        model, device.removeprefix("ov:"), {"EXECUTION_MODE_HINT": "ACCURACY"}
    )
    return compiled, processor, device
