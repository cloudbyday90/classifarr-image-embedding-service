# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import base64

import httpx
import pytest
from capacity_mixed import check_mixed_capacity, validate_batch
from capacity_remote_fixture import remote_fixture
from capacity_workload import make_payloads
from test_capacity_workload import FixtureEmbedder, workload

from image_embedder import remote_process
from image_embedder.config import Settings


@pytest.mark.parametrize("detached_failure", [False, True])
def test_real_children_mixed_api_cancel_and_restore_command(detached_failure):
    subject = workload([])
    settings = Settings(
        allow_remote_urls=True,
        allowed_remote_hosts=["capacity.example"],
        max_image_bytes=65536,
        embed_batch_api_max_items=8,
        warmup_on_startup=False,
        cleanup_on_shutdown=False,
    )

    class RemoteEmbedder(FixtureEmbedder):
        large_batches = 0

        def embed_batch(self, spec, size, items):
            for item in items:
                if item.image_url:
                    data = remote_process.fetch_remote_image(item.image_url, settings)
                    assert len(data) == 65536 and data.startswith(
                        base64.b64decode(subject.payloads[int(item.image_url[-1])])
                    )
            result = super().embed_batch(spec, size, items)
            if spec.name == "ViT-L-14":
                self.large_batches += 1
                if detached_failure and self.large_batches == 2:
                    raise ValueError("detached fixture failure")
            return result

    subject.embedder = RemoteEmbedder()
    subject.embedder.settings = settings
    subject.payloads = make_payloads(32)
    subject.load(False)
    command = remote_process._command
    if detached_failure:
        with pytest.raises(AssertionError, match="failed after caller cancellation"):
            asyncio.run(check_mixed_capacity(subject))
        assert remote_process._command is command
        return
    result = asyncio.run(check_mixed_capacity(subject))
    assert result["child_count"] == 4 and result["children_reaped"]
    assert result["validated_native_batches"] == 4
    assert result["queued"]["waiting"] == 1
    assert result["detached_owner"]["in_flight"] == 1
    assert result["settled"]["in_flight"] == result["settled"]["waiting"] == 0
    assert remote_process._command is command


def test_fixture_rejects_excess_before_starting_server():
    with pytest.raises(ValueError, match="exceeds"):
        with remote_fixture(make_payloads(32), 1):
            pytest.fail("oversized fixture started")


@pytest.mark.parametrize("status", [401, 413, 504])
def test_mixed_probe_requires_success(status):
    with pytest.raises(AssertionError, match="HTTP"):
        validate_batch(httpx.Response(status), "fixture", [], {})
