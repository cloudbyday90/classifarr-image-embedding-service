# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offline publisher metadata and independent PIL preprocessing contracts."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from image_embedder.artifact_integrity import sha256_file
from image_embedder.model_catalog import MODEL_CATALOG
from image_embedder.model_loading import (
    load_processor,
    load_vision_model,
    verified_asset,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models"


@pytest.mark.parametrize("name", MODEL_CATALOG)
def test_official_metadata_matches_immutable_catalog(name):
    spec = MODEL_CATALOG[name]
    source = next(
        s
        for s in json.loads((FIXTURES / "sources.json").read_text())
        if s["name"] == name
    )
    assert spec.hf_id == source["repo_id"] and spec.revision == source["revision"]
    for filename, digest in spec.assets:
        assert digest == source["assets"][filename]["sha256"]
        if filename.endswith(".json"):
            assert sha256_file(FIXTURES / name / filename) == digest
    config = json.loads((FIXTURES / name / "config.json").read_text())
    assert config["projection_dim"] == spec.dims
    assert config["vision_config"]["image_size"] == spec.image_size
    with pytest.raises(FrozenInstanceError):
        spec.revision = "0" * 40
    with pytest.raises(TypeError):
        MODEL_CATALOG[name] = spec


@pytest.mark.parametrize("name", MODEL_CATALOG)
@pytest.mark.parametrize("shape", [(83, 129), (128, 64)])
def test_production_preprocessing_matches_independent_pil_math(name, shape):
    from transformers import CLIPImageProcessorPil

    h, w = shape
    y, x = np.indices((h, w))
    pixels = np.stack(
        [(x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256], axis=-1
    ).astype(np.uint8)
    processor = CLIPImageProcessorPil.from_pretrained(
        str(FIXTURES / name), local_files_only=True
    )
    with Image.fromarray(pixels) as image:
        actual = processor(images=image, return_tensors="np")["pixel_values"]
        resized = image.resize(
            (int(w * 224 / min(h, w)), int(h * 224 / min(h, w))),
            Image.Resampling.BICUBIC,
        )
        with resized:
            left, top = (resized.width - 224) // 2, (resized.height - 224) // 2
            with resized.crop((left, top, left + 224, top + 224)) as cropped:
                expected = np.asarray(cropped).astype(np.float32) / 255
    mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    expected = ((expected - mean) / std).transpose(2, 0, 1)[None]
    assert actual.shape == (1, 3, 224, 224) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_source_verification_uses_pinned_unauthenticated_download(monkeypatch):
    import huggingface_hub

    spec = MODEL_CATALOG["ViT-B-16"]
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(FIXTURES / spec.name / kwargs["filename"])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert verified_asset(spec, "config.json") == FIXTURES / spec.name / "config.json"
    assert calls == [
        {
            "repo_id": spec.hf_id,
            "filename": "config.json",
            "revision": spec.revision,
            "token": False,
        }
    ]
    assert load_processor(spec).size == {"shortest_edge": 224}


@pytest.mark.parametrize("failure", ["revision", "digest", "tampered", "missing"])
def test_unverified_assets_fail_before_model_deserialization(
    monkeypatch, tmp_path, failure
):
    import huggingface_hub
    import transformers

    spec = MODEL_CATALOG["ViT-B-16"]
    path = tmp_path / "config.json"
    path.write_text("{}")
    if failure == "revision":
        spec = replace(spec, revision="main")
    if failure == "digest":
        spec = replace(spec, assets=(("config.json", "bad"),))
    if failure == "missing":
        spec = replace(spec, assets=())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(path))
    monkeypatch.setattr(
        transformers.CLIPVisionModelWithProjection,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail(
            "unverified assets reached deserialization"
        ),
    )
    with pytest.raises(ValueError, match="Unverified|digest mismatch"):
        verified_asset(spec, "config.json")
    with pytest.raises(ValueError, match="Unverified|digest mismatch"):
        load_vision_model(spec)


@pytest.mark.parametrize("name", MODEL_CATALOG)
def test_local_weight_loader_is_restricted_and_preserves_dimensions(monkeypatch, name):
    import transformers

    spec = MODEL_CATALOG[name]
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(projection_dim=spec.dims, image_size=224),
        eval=lambda: None,
    )

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return model

    monkeypatch.setattr(
        "image_embedder.model_loading.verified_asset",
        lambda _s, filename: FIXTURES / name / filename,
    )
    monkeypatch.setattr(
        transformers.CLIPVisionModelWithProjection, "from_pretrained", load
    )
    assert load_vision_model(spec) is model
    assert calls == [
        (
            str(FIXTURES / name),
            {
                "local_files_only": True,
                "use_safetensors": name == "ViT-L-14",
                "weights_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    model.config.projection_dim += 1
    with pytest.raises(ValueError, match="dimensions"):
        load_vision_model(spec)
    monkeypatch.setattr(
        "image_embedder.model_loading.verified_asset",
        lambda _s, filename: (
            Path(filename) if filename.endswith(".bin") else FIXTURES / name / filename
        ),
    )
    if name == "ViT-B-16":
        with pytest.raises(ValueError, match="same snapshot"):
            load_vision_model(spec)
