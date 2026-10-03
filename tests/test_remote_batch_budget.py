# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fetch timing preserves ordered input, cache and inference contracts."""

import base64
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import _png_bytes

from image_embedder import remote_budget
from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, ImageEmbedder
from image_embedder.remote_budget import RemoteBatchBudget
from image_embedder.remote_fetch import RemoteFetchError


def clock(monkeypatch):
    current = [100.0]
    monkeypatch.setattr(
        remote_budget, "time", SimpleNamespace(monotonic=lambda: current[0])
    )
    return current


def prepared_embedder(monkeypatch, *, cache=0, **overrides):
    settings = Settings(
        allow_remote_urls=True,
        remote_batch_fetch_timeout_seconds=1,
        embed_cache_size=cache,
        **overrides,
    )
    embedder = ImageEmbedder(settings)
    spec = embedder.resolve_model(None)
    calls = []

    def processor(*, images, **_kwargs):
        assert all(image.mode == "RGB" for image in images)
        calls.append(tuple(images))
        return {"pixels": np.ones((len(images), spec.dims), dtype=np.float32)}

    def model(inputs):
        return [inputs["pixels"]]

    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: (model, processor, "ov:CPU")
    )
    return embedder, spec, calls


def inline_item():
    return BatchItem(None, base64.b64encode(_png_bytes()).decode(), False)


def test_deadline_starts_lazily_and_never_resets(monkeypatch):
    current = clock(monkeypatch)
    settings = Settings(remote_batch_fetch_timeout_seconds=1)
    budget = RemoteBatchBudget(settings)
    current[0] += 50  # Queue/inline work before first remote fetch is excluded.
    deadlines = []

    def fetch(_url, received_settings, *, total_deadline):
        assert received_settings is settings
        deadlines.append(total_deadline)
        current[0] += 0.4
        return b"image"

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    assert budget.fetch("https://images.example/a") == b"image"
    assert budget.fetch("https://images.example/b") == b"image"
    current[0] = 151  # Exact boundary; no third worker call.
    with pytest.raises(RemoteFetchError, match="^Remote batch fetch timed out$"):
        budget.fetch("https://images.example/c?token=test-private-detail")
    assert deadlines == [151, 151]


@pytest.mark.parametrize("late_error", [False, True])
def test_late_completion_or_cleanup_cannot_admit_remote_bytes(monkeypatch, late_error):
    current = clock(monkeypatch)
    budget = RemoteBatchBudget(Settings(remote_batch_fetch_timeout_seconds=1))

    def fetch(*_args, **_kwargs):
        current[0] += 1
        if late_error:
            raise RemoteFetchError("Remote image fetch timed out")
        return b"late image"

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    with pytest.raises(RemoteFetchError, match="^Remote batch fetch timed out$"):
        budget.fetch("https://images.example/a")


def test_early_per_image_error_does_not_exhaust_or_reset_shared_budget(monkeypatch):
    current = clock(monkeypatch)
    budget = RemoteBatchBudget(Settings(remote_batch_fetch_timeout_seconds=1))
    deadlines = []

    def fetch(_url, _settings, *, total_deadline):
        deadlines.append(total_deadline)
        current[0] += 0.2
        if len(deadlines) == 1:
            raise RemoteFetchError("Remote image fetch timed out")
        return b"image"

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    with pytest.raises(RemoteFetchError, match="^Remote image fetch timed out$"):
        budget.fetch("https://images.example/a")
    assert budget.fetch("https://images.example/b") == b"image"
    assert deadlines == [101, 101]


@pytest.mark.parametrize("cache", [0, 8])
def test_mixed_batch_keeps_order_partial_success_and_remote_cache_freshness(
    monkeypatch, cache
):
    current = clock(monkeypatch)
    embedder, spec, calls = prepared_embedder(monkeypatch, cache=cache)
    payload = _png_bytes()
    cached = (
        [1.0] + [0.0] * (spec.dims - 1),
        spec.dims,
        "local",
        spec.name,
        spec.image_size,
    )
    if embedder._embedding_cache is not None:
        embedder._embedding_cache.put(
            embedder._embedding_cache.make_key(
                payload, spec.name, spec.image_size, True
            ),
            cached,
        )
    fetched = []

    def fetch(url, _settings, *, total_deadline):
        fetched.append((url, total_deadline))
        current[0] += 0.4 if len(fetched) == 1 else 0.6
        return payload

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    items = [
        inline_item(),
        BatchItem("https://images.example/a", None, True),
        BatchItem("https://images.example/b", None, False),
        inline_item(),
        BatchItem("https://images.example/c?token=test-private-detail", None, False),
    ]
    results = embedder.embed_batch(spec, spec.image_size, items)
    assert len(results) == len(items)
    assert [isinstance(result, Exception) for result in results] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert str(results[2]) == str(results[4]) == "Remote batch fetch timed out"
    assert results[0] == results[3]
    if cache:
        assert results[1] == cached
    else:
        assert len(results[1][0]) == spec.dims
    assert [deadline for _url, deadline in fetched] == [101, 101]
    assert len(calls) == 1
    # PIL close invalidates access; both successful uncached inputs are closed.
    for image in calls[0]:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


def test_inline_work_between_remote_inputs_consumes_elapsed_fetch_phase(monkeypatch):
    current = clock(monkeypatch)
    embedder, spec, _calls = prepared_embedder(monkeypatch)
    fetched = []

    def fetch(url, *_args, **_kwargs):
        fetched.append(url)
        return _png_bytes()

    decode = embedder._decode_base64

    def delayed_decode(payload):
        current[0] += 1
        return decode(payload)

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    monkeypatch.setattr(embedder, "_decode_base64", delayed_decode)
    result = embedder.embed_batch(
        spec,
        spec.image_size,
        [
            BatchItem("https://images.example/a", None, False),
            inline_item(),
            BatchItem("https://images.example/b", None, False),
        ],
    )
    assert not isinstance(result[0], Exception) and not isinstance(result[1], Exception)
    assert str(result[2]) == "Remote batch fetch timed out"
    assert fetched == ["https://images.example/a"]


def test_independent_invocations_do_not_share_budget_or_modify_settings(monkeypatch):
    current = clock(monkeypatch)
    embedder, spec, _calls = prepared_embedder(monkeypatch)
    original_settings = vars(embedder.settings).copy()
    deadlines = []

    def fetch(*_args, total_deadline):
        deadlines.append(total_deadline)
        current[0] += 1
        raise RemoteFetchError("Remote image fetch timed out")

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    monkeypatch.setattr(
        embedder,
        "_load_model",
        lambda _spec: pytest.fail("all-failed batch loaded model"),
    )
    item = BatchItem("https://images.example/a", None, False)
    for _ in range(2):
        results = embedder.embed_batch(spec, spec.image_size, [item, item])
        assert [str(result) for result in results] == [
            "Remote batch fetch timed out"
        ] * 2
    assert deadlines == [101, 102]
    assert vars(embedder.settings) == original_settings


def test_inline_only_batch_does_not_touch_remote_clock(monkeypatch):
    embedder, spec, _calls = prepared_embedder(monkeypatch)
    monkeypatch.setattr(
        remote_budget,
        "time",
        SimpleNamespace(
            monotonic=lambda: pytest.fail("inline batch started remote clock")
        ),
    )
    assert not isinstance(
        embedder.embed_batch(spec, spec.image_size, [inline_item()])[0], Exception
    )


def test_model_execution_retains_its_independent_lifetime(monkeypatch):
    current = clock(monkeypatch)
    embedder, spec, _calls = prepared_embedder(monkeypatch)
    deadlines = []

    def fetch(*_args, total_deadline):
        deadlines.append(total_deadline)
        current[0] += 0.2
        return _png_bytes()

    model, processor, device = embedder._load_model(spec)

    def slow_model(inputs):
        current[0] += 10  # Fetch budget is not a native inference kill boundary.
        return model(inputs)

    monkeypatch.setattr(remote_budget, "fetch_remote_image", fetch)
    monkeypatch.setattr(
        embedder, "_load_model", lambda _spec: (slow_model, processor, device)
    )
    result = embedder.embed_batch(
        spec, spec.image_size, [BatchItem("https://images.example/a", None, False)]
    )
    assert not isinstance(result[0], Exception)
    assert len(result[0][0]) == spec.dims
    assert current[0] > deadlines[0]
