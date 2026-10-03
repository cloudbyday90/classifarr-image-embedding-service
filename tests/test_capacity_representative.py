# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import base64
import hashlib
from dataclasses import replace
from io import BytesIO

import capacity_probe
import numpy as np
import pytest
from capacity_image_fixtures import make_fixtures, validate_fixtures
from capacity_representative import check_representative
from capacity_representative_client import request_case
from capacity_workload import CapacityWorkload
from PIL import Image
from test_capacity_workload import FixtureEmbedder, fixture_memory

from image_embedder.config import Settings
from image_embedder.model_catalog import MODEL_CATALOG


def test_encoded_fixtures_cover_operator_shapes_codecs_and_entropy():
    fixtures = make_fixtures()
    assert [fixture.metadata for fixture in fixtures] == [
        fixture.metadata for fixture in make_fixtures()
    ]
    assert {fixture.metadata["codec"] for fixture in fixtures} == {
        "PNG",
        "JPEG",
        "WEBP",
    }
    assert {fixture.metadata["mode"] for fixture in fixtures} == {"RGB", "RGBA"}
    for fixture in fixtures:
        assert fixture.payloads[0] != fixture.payloads[1]
        for variant, payload in zip(
            fixture.metadata["variants"], fixture.payloads, strict=True
        ):
            encoded = base64.b64decode(payload, validate=True)
            assert len(encoded) == variant["encoded_bytes"]
            assert hashlib.sha256(encoded).hexdigest() == variant["sha256"]
            with Image.open(BytesIO(encoded)) as image:
                image.load()
                assert image.size == (
                    fixture.metadata["width"],
                    fixture.metadata["height"],
                )
    assert (
        min(variant["encoded_bytes"] for variant in fixtures[-1].metadata["variants"])
        > 100_000
    )
    validate_fixtures(
        fixtures, Settings(), list(MODEL_CATALOG.values()), [1, 8, 32], 2, 1
    )


@pytest.mark.parametrize(
    "settings,clients,repeats",
    [
        (Settings(max_image_bytes=100), 2, 1),
        (Settings(max_batch_image_bytes=100), 2, 1),
        (Settings(max_image_pixels=100), 2, 1),
        (Settings(max_batch_image_pixels=100), 2, 1),
        (Settings(max_request_body_bytes=100), 2, 1),
        (Settings(max_http_requests=1), 2, 1),
        (Settings(), 9, 1),
        (Settings(), 2, 3),
    ],
)
def test_fixture_or_pressure_overflow_refused(settings, clients, repeats):
    with pytest.raises(ValueError):
        validate_fixtures(
            make_fixtures(),
            settings,
            list(MODEL_CATALOG.values()),
            [1, 8, 32],
            clients,
            repeats,
        )


@pytest.mark.parametrize(
    "option,value", [("--clients", "9"), ("--repeats", "3"), ("--batch-sizes", "3")]
)
def test_representative_cli_refuses_pressure_before_model_ownership(
    monkeypatch, option, value
):
    monkeypatch.setattr(capacity_probe.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(capacity_probe, "MemoryReader", lambda: fixture_memory)
    monkeypatch.setattr(
        capacity_probe, "ImageEmbedder", lambda *_: pytest.fail("Unexpected owner")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "capacity_probe.py",
            "--backend",
            "cpu",
            "--scenario",
            "representative",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit):
        capacity_probe.main()


class EncodedFixtureEmbedder(FixtureEmbedder):
    def embed(self, url, payload, name, normalize, size):
        vector = np.zeros(self.resolve_model(name).dims)
        vector[:4] = list(
            hashlib.sha256(base64.b64decode(payload, validate=True)).digest()[:4]
        )
        if normalize:
            vector /= np.linalg.norm(vector)
        return vector.tolist(), len(vector), "local", name, size


def representative_workload():
    embedder = EncodedFixtureEmbedder()
    embedder.settings = Settings(warmup_on_startup=False, cleanup_on_shutdown=False)
    return CapacityWorkload(
        embedder,
        list(MODEL_CATALOG),
        [1, 8, 32],
        [],
        fixture_memory,
        lambda _: None,
        repeats=1,
    )


def test_native_socket_child_uses_encoded_refs_and_settles_all_owners(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("SERVICE_API_KEY", "ambient-private-value")
    subject = representative_workload()
    settings = subject.embedder.settings
    result = asyncio.run(check_representative(subject, 2))
    assert subject.embedder.settings is settings
    assert settings.rate_limit_embed == "30/minute"
    assert result["client"]["reaped"] and result["client"]["returncode"] == 0
    assert len(result["client"]["records"]) == 48
    assert {record["normalize"] for record in result["client"]["records"]} == {
        True,
        False,
    }
    assert all(
        record["request_bytes"] == record["server_received_bytes"]
        for record in result["client"]["records"]
    )
    assert (
        result["unauthenticated_status"] == 401
        and result["unauthenticated_received_bytes"] == 0
    )
    assert (
        result["settled_ingress"]["active"] == result["settled_queue"]["in_flight"] == 0
    )
    assert (
        result["settled_queue"]["waiting"] == result["settled_queue"]["rw_readers"] == 0
    )
    assert "ambient-private-value" not in str(result)


def test_child_parity_failure_is_not_counted_as_success(monkeypatch):
    subject = representative_workload()
    subject.sizes = [1]
    original = subject.embedder.embed_batch

    def corrupt(*args):
        results = original(*args)
        results[0][0][0] += 10
        return results

    monkeypatch.setattr(subject.embedder, "embed_batch", corrupt)
    events = []
    subject.emit = events.append
    with pytest.raises(RuntimeError, match="client failed"):
        asyncio.run(check_representative(subject, 2))
    result = events[-1]["result"]
    assert not result["client"]["passed"] and result["client"]["reaped"]
    assert any(not record["passed"] for record in result["client"]["records"])
    assert (
        result["settled_ingress"]["active"] == result["settled_queue"]["in_flight"] == 0
    )


def test_representative_scenario_does_not_run_square_stress(monkeypatch):
    subject = representative_workload()
    monkeypatch.setattr(
        subject, "serial_batches", lambda: pytest.fail("Unrelated stress workload")
    )
    subject.run("representative")
    assert subject.references == {}


def test_failed_deadline_records_status_and_settled_native_owner(monkeypatch):
    import time

    subject = representative_workload()
    subject.sizes = [1]
    subject.embedder.settings = replace(
        subject.embedder.settings, embedding_timeout_seconds=0.05
    )
    original = subject.embedder.embed_batch

    def slow(*args):
        time.sleep(0.2)
        return original(*args)

    monkeypatch.setattr(subject.embedder, "embed_batch", slow)
    events = []
    subject.emit = events.append
    with pytest.raises(RuntimeError, match="client failed"):
        asyncio.run(check_representative(subject, 2))
    result = events[-1]["result"]
    assert result["client"]["reaped"] and not result["client"]["passed"]
    assert any(record["status"] == 504 for record in result["client"]["records"])
    assert (
        result["settled_queue"]["in_flight"] == result["settled_queue"]["waiting"] == 0
    )
    assert (
        result["settled_queue"]["rw_readers"]
        == result["settled_ingress"]["active"]
        == 0
    )


@pytest.mark.parametrize(
    "status,payload,error",
    [(429, b"refused", AssertionError), (200, b"x" * 2000, ValueError)],
)
def test_http_failure_or_oversized_response_is_recorded(
    monkeypatch, status, payload, error
):
    import capacity_representative_client
    import httpx

    monkeypatch.setattr(capacity_representative_client, "MAX_RESPONSE_BYTES", 1024)
    records = []

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status, content=payload)
            ),
            base_url="http://127.0.0.1:1234",
        ) as client:
            with pytest.raises(error):
                await request_case(
                    client,
                    {"name": "ViT-B-16", "dims": 2, "image_size": 224},
                    make_fixtures()[0],
                    1,
                    0,
                    0,
                    False,
                    [[3, 4], [4, 3]],
                    records,
                )

    asyncio.run(scenario())
    assert records[0]["status"] == status and not records[0]["passed"]
    assert records[0]["response_bytes"] <= 1024
