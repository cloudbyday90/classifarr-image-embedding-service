# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offline native dependency and tiny CLIP projection checks for container CI."""

import os
import re
from collections import Counter
from importlib import import_module
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory


def validate_torch_backend(backend: str, cuda: str | None, hip: str | None) -> None:
    """Reject a CPU/GPU wheel substitution, even on hosts without an accelerator."""
    if backend not in {"cpu", "cuda", "cuda-legacy", "openvino"}:
        raise ValueError(f"Unknown backend: {backend}")
    expected_cuda = {"cuda": "13.0", "cuda-legacy": "12.6"}.get(backend)
    if cuda != expected_cuda or hip is not None:
        raise RuntimeError(
            f"Wrong Torch profile for {backend}: CUDA={cuda}, HIP={hip}; "
            f"expected CUDA={expected_cuda}, HIP=None"
        )


def validate_package_inventory() -> None:
    """Refuse stale metadata left by overlaying an environment from another image."""
    names = (
        re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        for dist in distributions()
    )
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate installed package metadata: {', '.join(duplicates)}")


def probe_backend(backend: str, cache_root: Path, *, require_gpu: bool = False) -> dict[str, object]:
    """Exercise installed wheels without production weights or external requests."""
    validate_package_inventory()
    import torch
    from PIL import Image
    from transformers import (
        CLIPImageProcessorPil,
        CLIPVisionConfig,
        CLIPVisionModelWithProjection,
    )

    validate_torch_backend(backend, torch.version.cuda, torch.version.hip)
    runtime_cuda = os.environ.get("CUDA_VERSION", "")
    if backend.startswith("cuda") and ".".join(runtime_cuda.split(".")[:2]) != torch.version.cuda:
        raise RuntimeError(f"CUDA base {runtime_cuda!r} does not match Torch {torch.version.cuda}")
    if require_gpu and (not backend.startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("GPU execution was required but CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    processor = CLIPImageProcessorPil(size=16, crop_size=16)
    pixels = processor(images=Image.new("RGB", (16, 16), "navy"), return_tensors="pt")
    model: torch.nn.Module = CLIPVisionModelWithProjection(
        CLIPVisionConfig.from_dict({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 16,
            "patch_size": 8,
            "projection_dim": 8,
        })
    )
    model.eval()
    with torch.inference_mode():
        expected = model(**pixels).image_embeds
    if tuple(expected.shape) != (1, 8) or not torch.isfinite(expected).all():
        raise RuntimeError("Tiny CLIP projection produced invalid output")

    result: dict[str, object] = {
        "backend": backend,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "projection_shape": list(expected.shape),
        "gpu_inference": False,
    }
    if backend.startswith("cuda") and torch.cuda.is_available():
        torch.nn.Module.cuda(model)
        with torch.inference_mode():
            actual = model(**{key: value.to("cuda") for key, value in pixels.items()}).image_embeds.cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        result.update(gpu_inference=True, gpu_name=torch.cuda.get_device_name(0))
    # Each check creates and removes only its own subdirectory in the model cache.
    cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="backend-smoke-", dir=cache_root) as cache_dir:
        Path(cache_dir, "writable").write_text("ok", encoding="utf-8")
        if backend == "openvino":
            result.update(probe_openvino(model, pixels, expected, Path(cache_dir)))
    return result


def probe_openvino(model, pixels, expected, cache_dir: Path) -> dict[str, object]:
    """Check first-export and cached-IR inference with actual OpenVINO bindings."""
    import numpy as np

    ov = import_module("openvino")

    if not ov.__version__.startswith("2026.4.0"):
        raise RuntimeError(f"OpenVINO bindings do not match the base: {ov.__version__}")
    converted = ov.convert_model(model, example_input=dict(pixels))
    xml_path = cache_dir / "model.xml"
    ov.save_model(converted, str(xml_path), compress_to_fp16=False)
    core = ov.Core()
    reloaded = core.read_model(str(xml_path))
    compiled = core.compile_model(reloaded, "CPU")
    # Production uses the first output as the visual projection embedding.
    actual = compiled({"pixel_values": pixels["pixel_values"].numpy()})[compiled.output(0)]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
    return {"openvino": ov.__version__, "openvino_device": "CPU", "ir_reload": True}
