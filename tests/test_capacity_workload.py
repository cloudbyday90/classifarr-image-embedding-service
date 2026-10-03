# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import base64
from io import BytesIO

import capacity_probe
import numpy as np
import pytest
from capacity_workload import CapacityWorkload, make_payloads
from PIL import Image

from image_embedder.model_catalog import MODEL_CATALOG


class FixtureEmbedder:
    def _load_model(self, spec):
        return None, None, "cpu"

    def warmup(self, name):
        assert name in MODEL_CATALOG

    def resolve_model(self, name):
        return MODEL_CATALOG[name]

    def embed(self, url, payload, name, normalize, size):
        spec = MODEL_CATALOG[name]
        vector = np.zeros(spec.dims)
        vector[:2] = [3, 4]
        if normalize:
            vector /= 5
        return vector.tolist(), spec.dims, "local", name, size

    def embed_batch(self, spec, size, items):
        return [
            self.embed(
                item.image_url, item.image_base64, spec.name, item.normalize, size
            )
            for item in items
        ]


def fixture_memory():
    return {
        "process_rss_bytes": 10,
        "process_peak_rss_bytes": 20,
        "cgroup": {"current_bytes": 30, "limit_bytes": 65536, "events": {}},
    }


def workload(events):
    return CapacityWorkload(
        FixtureEmbedder(),
        list(MODEL_CATALOG),
        [1, 8, 32],
        ["first", "second"],
        fixture_memory,
        events.append,
        repeats=1,
    )


def test_synthetic_payloads_are_deterministic_distinct_square_images():
    payloads = make_payloads(32)
    assert payloads == make_payloads(32) and payloads[0] != payloads[1]
    for payload in payloads:
        with Image.open(BytesIO(base64.b64decode(payload, validate=True))) as image:
            assert image.size == (32, 32) and image.mode == "RGB"


@pytest.mark.parametrize("scenario", ["serial", "overlap", "detached"])
def test_workloads_preserve_vectors_and_native_owner_capacity(scenario):
    events = []
    workload(events).run(scenario)
    assert (
        sum(
            event["event"] == "phase_end" and event["phase"].startswith("batch/")
            for event in events
        )
        == 6
    )
    if scenario == "detached":
        owner = next(
            event["result"] for event in events if event["event"] == "ownership"
        )
        assert owner["detached"]["in_flight"] == 1
        assert owner["live_waiting"]["waiting"] == 1
        assert owner["settled"]["in_flight"] == owner["settled"]["waiting"] == 0
        assert set(owner["timings"]) == {"detached", "live"}


def test_failed_phase_records_only_error_type():
    events = []

    def failure():
        raise ValueError("private image payload")

    with pytest.raises(ValueError):
        workload(events).measure("fixture", failure)
    assert events[-1] == {
        "event": "phase_error",
        "phase": "fixture",
        "error_type": "ValueError",
    }
    assert "private image payload" not in str(events)


def test_bad_batch_metadata_fails_probe(monkeypatch):
    subject = workload([])
    subject.load(False)
    monkeypatch.setattr(
        subject.embedder, "embed_batch", lambda *_: [ValueError("fixture")]
    )
    with pytest.raises(AssertionError, match="metadata"):
        subject.batch(subject.models[0], 1)


def test_loaded_backend_must_match_declared_experiment(monkeypatch):
    subject = workload([])
    monkeypatch.setattr(
        subject.embedder, "_load_model", lambda *_: (None, None, "ov:CPU")
    )
    with pytest.raises(AssertionError, match="backend does not match"):
        subject.load(False)


@pytest.mark.parametrize("timeout", [False, True])
def test_real_api_capacity_contract_and_deadline(monkeypatch, timeout):
    import threading

    import capacity_api
    import httpx

    from image_embedder.config import Settings

    subject = workload([])
    subject.load(False)
    settings = Settings(
        warmup_on_startup=False, cleanup_on_shutdown=False, request_timeout_seconds=1
    )
    release = threading.Event()
    original_batch = subject.embedder.embed_batch

    def held_batch(*args):
        if not release.wait(5):
            raise TimeoutError("fixture barrier")
        return original_batch(*args)

    async def response_hook(response):
        if response.status_code == 504:
            release.set()

    original_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return original_client(**kwargs, event_hooks={"response": [response_hook]})

    if timeout:
        monkeypatch.setattr(subject.embedder, "embed_batch", held_batch)
        monkeypatch.setattr(capacity_api.httpx, "AsyncClient", client_factory)
    try:
        result = asyncio.run(
            capacity_api.check_api_deadline(
                subject.embedder,
                settings,
                subject.models,
                make_payloads(32)[0],
                32,
                subject.references,
            )
        )
        assert result["batch_http_status"] == (504 if timeout else 200)
        assert result["unauthenticated_status"] == 401
        assert result["oversized_status"] == 413
        assert result["live_http_status"] == 200
        assert result["settled"]["in_flight"] == result["settled"]["waiting"] == 0
        if timeout:
            assert result["in_flight_header_at_batch_response"] == "1"
    finally:
        release.set()


def test_detached_native_failure_is_not_hidden_by_canceled_caller(monkeypatch):
    subject = workload([])
    subject.load(False)

    def batch(name, size):
        if name == "ViT-L-14":
            raise ValueError("fixture")

    monkeypatch.setattr(subject, "batch", batch)
    with pytest.raises(AssertionError, match="failed after caller cancellation"):
        asyncio.run(subject.detached_owner())


@pytest.mark.parametrize("reason", ["offline", "unbounded", "pixels", "interval"])
def test_cli_refuses_invalid_experiments_before_model_load(monkeypatch, reason):
    monkeypatch.setattr(capacity_probe.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    memory = fixture_memory()
    argv = ["capacity_probe.py", "--backend", "cpu"]
    if reason == "offline":
        monkeypatch.delenv("HF_HUB_OFFLINE")
    elif reason == "unbounded":
        memory["cgroup"]["limit_bytes"] = None
    elif reason == "pixels":
        argv += ["--image-edge", "1000000000"]
    else:
        argv += ["--sample-interval", "nan"]
    monkeypatch.setattr(capacity_probe, "MemoryReader", lambda: lambda: memory)
    monkeypatch.setattr(
        capacity_probe,
        "ImageEmbedder",
        lambda *_: pytest.fail("Unexpected model owner"),
    )
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        capacity_probe.main()


def test_private_cold_cache_is_restored_and_cleaned(monkeypatch, tmp_path):
    monkeypatch.setattr(capacity_probe.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("OV_MODEL_CACHE", "original-cache")
    monkeypatch.setattr(capacity_probe, "MemoryReader", lambda: fixture_memory)
    monkeypatch.setattr(capacity_probe, "ImageEmbedder", lambda *_: FixtureEmbedder())
    seen = []

    def run(_self, _scenario):
        import os
        from pathlib import Path

        seen.append(Path(os.environ["OV_MODEL_CACHE"]))
        assert seen[-1].is_dir() and seen[-1].parent == tmp_path

    monkeypatch.setattr(capacity_probe.CapacityWorkload, "run", run)
    monkeypatch.setattr(
        "sys.argv", ["capacity_probe.py", "--backend", "openvino", "--cold-ir"]
    )
    monkeypatch.setattr(capacity_probe, "version", lambda *_: "fixture")
    capacity_probe.main()
    assert not seen[0].exists()
    import os

    assert os.environ["OV_MODEL_CACHE"] == "original-cache"
