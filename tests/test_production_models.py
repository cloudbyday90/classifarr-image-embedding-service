# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Opt-in real publisher weights; the routine suite never downloads them."""

import os

import pytest
from production_model_probe import probe_model

from image_embedder.model_catalog import MODEL_CATALOG

pytestmark = [
    pytest.mark.production_model,
    pytest.mark.skipif(
        os.environ.get("RUN_PRODUCTION_MODEL_TESTS") != "1",
        reason="explicit pinned model cache required",
    ),
]


@pytest.mark.parametrize("name", MODEL_CATALOG)
def test_pinned_cpu_model_single_batch_and_protected_route(name, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    result = probe_model(name, "cpu")
    assert (
        result["single_batch_direct_parity"]
        and result["authenticated_schema_unchanged"]
    )
