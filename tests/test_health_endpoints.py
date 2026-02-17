# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from unittest.mock import patch, MagicMock

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder, MODEL_CATALOG


def test_get_device_info_cpu():
    settings = Settings(device="cpu")
    embedder = ImageEmbedder(settings=settings)
    info = embedder.get_device_info()
    assert info["type"] == "cpu"


def test_get_model_status():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    status = embedder.get_model_status()
    assert len(status) == len(MODEL_CATALOG)
    for entry in status:
        assert "name" in entry
        assert "loaded" in entry
        assert entry["loaded"] is False


def test_get_model_status_with_loaded_model():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    
    mock_model = MagicMock()
    mock_processor = MagicMock()
    embedder._models["ViT-L-14"] = (mock_model, mock_processor, "cpu")
        
    status = embedder.get_model_status()
    vit_l_status = next(s for s in status if s["name"] == "ViT-L-14")
    assert vit_l_status["loaded"] is True


def test_get_memory_info_cpu_returns_none():
    settings = Settings(device="cpu")
    embedder = ImageEmbedder(settings=settings)
    info = embedder.get_memory_info()
    assert info is None


def test_is_default_model_loaded_false_initially():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    assert embedder.is_default_model_loaded() is False


def test_is_default_model_loaded_true_after_direct_load():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)
    
    mock_model = MagicMock()
    mock_processor = MagicMock()
    embedder._models["ViT-L-14"] = (mock_model, mock_processor, "cpu")
        
    assert embedder.is_default_model_loaded() is True
