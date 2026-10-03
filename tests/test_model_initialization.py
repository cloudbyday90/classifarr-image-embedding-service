# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder
from image_embedder.model_catalog import MODEL_CATALOG
from image_embedder.model_initialization import (
    initialization_guard,
    serialized_initialization,
)


def test_initialization_is_process_shared_reentrant_and_released_after_failure():
    entered, release, contender_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    events = []

    @serialized_initialization
    def first():
        with initialization_guard():
            events.append("first")
            entered.set()
            assert release.wait(2)
            raise ValueError("fixture")

    @serialized_initialization
    def second():
        events.append("second")
        return "loaded"

    def contender():
        contender_started.set()
        return second()

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(first)
        assert entered.wait(2)
        waiting = pool.submit(contender)
        try:
            assert contender_started.wait(2)
            assert not waiting.done()
        finally:
            release.set()
        with pytest.raises(ValueError):
            owner.result(timeout=2)
        assert waiting.result(timeout=2) == "loaded"
    assert events == ["first", "second"]


def test_warm_model_returns_while_another_owner_is_initializing(monkeypatch):
    warm, cold = (
        ImageEmbedder(Settings(device="cpu")),
        ImageEmbedder(Settings(device="cpu")),
    )
    spec = MODEL_CATALOG["ViT-L-14"]
    cached = (object(), object(), "cpu")
    warm._models[spec.name] = cached
    entered, release = threading.Event(), threading.Event()

    def processor(_spec):
        entered.set()
        assert release.wait(2)
        return object()

    monkeypatch.setattr("image_embedder.embedder.load_processor", processor)
    monkeypatch.setattr("image_embedder.embedder.load_vision_model", lambda *_: Mock())
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(cold._load_model, MODEL_CATALOG["ViT-B-16"])
        try:
            assert entered.wait(2)
            assert pool.submit(warm._load_model, spec).result(timeout=1) is cached
        finally:
            release.set()
        pending.result(timeout=2)


@pytest.mark.parametrize("separate_owners", [False, True])
def test_cold_model_aliases_and_owners_do_not_overlap(monkeypatch, separate_owners):
    first = ImageEmbedder(Settings(device="cpu"))
    second = ImageEmbedder(Settings(device="cpu")) if separate_owners else first
    entered, release, contender_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    guard = threading.Lock()
    active = peak = calls = 0

    def processor(spec):
        nonlocal active, peak, calls
        with guard:
            active += 1
            peak = max(peak, active)
            calls += 1
        try:
            if spec.name == "ViT-L-14":
                entered.set()
                assert release.wait(2)
            return object()
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr("image_embedder.embedder.load_processor", processor)
    monkeypatch.setattr("image_embedder.embedder.load_vision_model", lambda *_: Mock())

    def contender():
        contender_started.set()
        return second._load_model(MODEL_CATALOG["ViT-B-16"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(first._load_model, MODEL_CATALOG["ViT-L-14"])
        assert entered.wait(2)
        waiting = pool.submit(contender)
        try:
            assert contender_started.wait(2)
            assert not waiting.done()
        finally:
            release.set()
        assert owner.result(timeout=2)[2] == waiting.result(timeout=2)[2] == "cpu"
    assert peak == 1 and calls == 2 and active == 0
