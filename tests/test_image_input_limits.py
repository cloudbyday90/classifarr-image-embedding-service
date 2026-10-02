# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real encoded images reject expansion before decoder/processor allocation."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from image_embedder import config
from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, ImageEmbedder, ModelSpec
from image_embedder.image_input import decode_base64, image_pixel_cost, load_rgb
from image_embedder.input_limits import BatchInputBudget, InputLimitExceeded


def png(size=(2, 2)):
    with Image.new("RGB", size, "navy") as image:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6])
def test_exact_byte_ceiling_and_padding(length):
    value = base64.b64encode(b"x" * length).decode("ascii")
    assert decode_base64(value, length) == b"x" * length
    if length > 1:
        with pytest.raises(InputLimitExceeded):
            decode_base64(value, length - 1)


def test_oversized_base64_never_calls_decoder(monkeypatch):
    monkeypatch.setattr(
        base64, "b64decode", lambda *_a, **_k: pytest.fail("decoder allocated")
    )
    with pytest.raises(InputLimitExceeded):
        decode_base64("A" * 100, 3)


@pytest.mark.parametrize("value", ["!!!!", "a===", "a", "é", "AA=!"])
def test_invalid_base64_remains_strict(value):
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_base64(value, 100)


def test_source_pixel_limit_precedes_rgb_conversion(monkeypatch):
    data = png((11, 10))
    monkeypatch.setattr(
        Image.Image, "convert", lambda *_a, **_k: pytest.fail("RGB allocated")
    )
    with pytest.raises(InputLimitExceeded, match="maximum pixel count"):
        load_rgb(data, 100)


def test_extreme_aspect_ratio_precedes_rgb_conversion(monkeypatch):
    data = png((1, 100))
    monkeypatch.setattr(
        Image.Image, "convert", lambda *_a, **_k: pytest.fail("RGB allocated")
    )
    with pytest.raises(InputLimitExceeded, match="resize"):
        load_rgb(data, 1000, target_size=224)


def test_pixel_cost_conservatively_covers_resize_rounding():
    assert image_pixel_cost((7, 3), 2, 21) == 31
    with load_rgb(png((3, 3)), 9, target_size=3) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (0, 0, 128)


def test_conversion_failure_rolls_back_pixel_reservation(monkeypatch):
    budget = BatchInputBudget(1000, 100)
    monkeypatch.setattr(
        Image.Image,
        "convert",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("bad pixels")),
    )
    with pytest.raises(ValueError, match="Unable to decode"):
        load_rgb(png(), 100, target_size=2, budget=budget)
    assert budget.used_pixels == 0


@pytest.mark.parametrize(
    "field",
    [
        "max_request_body_bytes",
        "max_image_bytes",
        "max_image_pixels",
        "max_batch_image_bytes",
        "max_batch_image_pixels",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True])
def test_nonpositive_or_boolean_limits_fail_closed(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        Settings(**{field: value})


@pytest.mark.parametrize("value", [True, 1.5, "100", 0, -1])
def test_toml_limits_are_positive_integers(monkeypatch, value):
    monkeypatch.delenv("MAX_IMAGE_PIXELS", raising=False)
    monkeypatch.setattr(config, "_TOML", {"image": {"max_image_pixels": value}})
    with pytest.raises(ValueError, match="positive integer"):
        Settings()


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_invalid_environment_limits_fail_startup(monkeypatch, value):
    monkeypatch.setenv("MAX_IMAGE_PIXELS", value)
    with pytest.raises(ValueError, match="positive integer"):
        Settings()


def test_environment_limit_overrides_toml(monkeypatch):
    monkeypatch.setattr(config, "_TOML", {"image": {"max_image_pixels": 12}})
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "24")
    assert Settings().max_image_pixels == 24


def install_model(monkeypatch, embedder, seen, *, fail=False):
    def processor(*, images, **_kwargs):
        seen.extend(images if isinstance(images, list) else [images])
        if fail:
            raise RuntimeError("processor failed")
        return {"images": images}

    def model(inputs):
        images = inputs["images"]
        count = len(images) if isinstance(images, list) else 1
        return [np.ones((count, 2), dtype=np.float32)]

    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: (model, processor, "ov:CPU")
    )


@pytest.mark.parametrize("cache", [0, 8])
def test_batch_pixel_exhaustion_keeps_order_and_allows_smaller_later_item(
    monkeypatch, cache
):
    embedder = ImageEmbedder(
        Settings(embed_cache_size=cache, max_image_pixels=4, max_batch_image_pixels=13)
    )
    spec = ModelSpec("tiny", "unused", 2, 2)
    values = [png(), png(), png((1, 1))]
    items = [BatchItem(None, base64.b64encode(data).decode(), False) for data in values]
    seen = []
    install_model(monkeypatch, embedder, seen)
    results = embedder.embed_batch(spec, 2, items)
    assert results[0][0] == [1.0, 1.0]
    assert isinstance(results[1], InputLimitExceeded)
    assert results[2][0] == [1.0, 1.0]
    assert len(seen) == 2
    for image in seen:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize("cache", [0, 8])
def test_batch_byte_exhaustion_precedes_retention_and_accepts_later_small_item(
    monkeypatch, cache
):
    small = png()
    values = [small, small + b"x" * 100, small]
    embedder = ImageEmbedder(
        Settings(embed_cache_size=cache, max_batch_image_bytes=2 * len(small))
    )
    spec = ModelSpec("tiny", "unused", 2, 2)
    items = [BatchItem(None, base64.b64encode(data).decode(), False) for data in values]
    seen = []
    install_model(monkeypatch, embedder, seen)
    results = embedder.embed_batch(spec, 2, items)
    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], InputLimitExceeded)
    assert not isinstance(results[2], Exception)
    assert len(seen) == 2


def test_cached_batch_inputs_cannot_bypass_byte_budget_or_trigger_model(monkeypatch):
    data = png()
    embedder = ImageEmbedder(
        Settings(
            embed_cache_size=8, max_batch_image_bytes=len(data), embed_cleanup_every_n=1
        )
    )
    spec = ModelSpec("tiny", "unused", 2, 2)
    key = embedder._embedding_cache.make_key(data, spec.name, 2, False)
    expected = ([1.0, 1.0], 2, "local", spec.name, 2)
    embedder._embedding_cache.put(key, expected)
    monkeypatch.setattr(
        embedder, "_load_model", lambda *_a: pytest.fail("cached batch loaded model")
    )
    item = BatchItem(None, base64.b64encode(data).decode(), False)
    results = embedder.embed_batch(spec, 2, [item, item])
    assert results[0] == expected
    assert isinstance(results[1], InputLimitExceeded)


@pytest.mark.parametrize("batch", [False, True])
def test_processor_failure_closes_decoded_images(monkeypatch, batch):
    embedder = ImageEmbedder(Settings(embed_cache_size=0))
    spec = embedder.resolve_model(None)
    seen = []
    install_model(monkeypatch, embedder, seen, fail=True)
    value = base64.b64encode(png()).decode()
    with pytest.raises(RuntimeError, match="processor failed"):
        if batch:
            embedder.embed_batch(spec, spec.image_size, [BatchItem(None, value, False)])
        else:
            embedder.embed(None, value, None, False, None)
    assert len(seen) == 1
    with pytest.raises(ValueError, match="closed image"):
        seen[0].getpixel((0, 0))


def test_all_invalid_images_skip_model_loading(monkeypatch):
    embedder = ImageEmbedder(Settings())
    spec = embedder.resolve_model(None)
    monkeypatch.setattr(
        embedder, "_load_model", lambda *_a: pytest.fail("invalid image loaded model")
    )
    results = embedder.embed_batch(
        spec, spec.image_size, [BatchItem(None, "AAAA", False)]
    )
    assert isinstance(results[0], ValueError)


def test_pillow_bomb_error_is_refused_without_changing_global_policy(monkeypatch):
    data = png()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(InputLimitExceeded, match="Pillow safety limit"):
        load_rgb(data, 100)
    assert Image.MAX_IMAGE_PIXELS == 1


def test_real_tiny_clip_single_and_batch_preserve_embeddings(monkeypatch):
    import torch
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionConfig,
        CLIPVisionModelWithProjection,
    )

    torch.manual_seed(0)
    spec = ModelSpec("tiny", "unused", 8, 16)
    model = CLIPVisionModelWithProjection(
        CLIPVisionConfig.from_dict(
            {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 16,
                "patch_size": 8,
                "projection_dim": 8,
            }
        )
    )
    model.eval()
    processor = CLIPImageProcessor(size=16, crop_size=16)
    embedder = ImageEmbedder(
        Settings(embed_cache_size=0, max_image_pixels=1000, max_batch_image_pixels=1000)
    )
    monkeypatch.setattr(embedder, "resolve_model", lambda _name: spec)
    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: (model, processor, "cpu")
    )
    data = png((3, 4))
    value = base64.b64encode(data).decode()
    with load_rgb(data, 1000, target_size=16) as image, torch.inference_mode():
        expected = (
            model(**processor(images=image, return_tensors="pt"))
            .image_embeds[0]
            .numpy()
        )
        expected_batch = model(
            **processor(images=[image, image], return_tensors="pt")
        ).image_embeds.numpy()
    single = embedder.embed(None, value, None, False, None)
    batch = embedder.embed_batch(spec, 16, [BatchItem(None, value, False)] * 2)
    np.testing.assert_allclose(single[0], expected, rtol=1e-6, atol=1e-7)
    for result, expected_row in zip(batch, expected_batch, strict=True):
        np.testing.assert_allclose(result[0], expected_row, rtol=1e-6, atol=1e-7)
        assert result[1:] == single[1:]
    assert single[1:] == (8, "local", "tiny", 16)
