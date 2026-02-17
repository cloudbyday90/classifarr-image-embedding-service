# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder, MODEL_CATALOG
from image_embedder.main import create_app


def test_warmup_loads_default_model():
    settings = Settings(default_model="ViT-B-16")
    embedder = ImageEmbedder(settings=settings)

    with patch.object(embedder, "_load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
        spec = embedder.warmup()
        assert spec.name == "ViT-B-16"
        mock_load.assert_called_once()


def test_warmup_loads_specified_model():
    settings = Settings(default_model="ViT-L-14")
    embedder = ImageEmbedder(settings=settings)

    with patch.object(embedder, "_load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
        spec = embedder.warmup("ViT-B-16")
        assert spec.name == "ViT-B-16"
        mock_load.assert_called_once()


def test_warmup_returns_model_spec():
    settings = Settings()
    embedder = ImageEmbedder(settings=settings)

    with patch.object(embedder, "_load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
        spec = embedder.warmup()
        assert spec.name in MODEL_CATALOG
        assert hasattr(spec, "dims")
        assert hasattr(spec, "hf_id")


def test_config_warmup_on_startup_default():
    settings = Settings()
    assert settings.warmup_on_startup is True


def test_config_warmup_on_startup_can_be_disabled():
    settings = Settings(warmup_on_startup=False)
    assert settings.warmup_on_startup is False


class FakeEmbedderWithWarmup:
    def __init__(self):
        self.warmed_up = False
        self.models = [
            {"name": "ViT-L-14", "dims": 768, "image_size": 224},
            {"name": "ViT-B-16", "dims": 512, "image_size": 224}
        ]

    def list_models(self):
        class Spec:
            def __init__(self, name, dims, image_size):
                self.name = name
                self.dims = dims
                self.image_size = image_size

        return [Spec(m["name"], m["dims"], m["image_size"]) for m in self.models]

    def embed(self, image_url, image_base64, model, normalize, image_size):
        return [0.1, 0.2, 0.3], 3, "local", model or "ViT-L-14", image_size or 224

    def warmup(self, model_name=None):
        self.warmed_up = True

    def get_device_info(self):
        return {"type": "cpu"}

    def get_model_status(self):
        return [{"name": "ViT-L-14", "loaded": self.warmed_up}]

    def get_memory_info(self):
        return None

    def is_default_model_loaded(self):
        return self.warmed_up


def test_app_warmup_called_on_startup_when_enabled():
    embedder = FakeEmbedderWithWarmup()
    settings = Settings(warmup_on_startup=True)
    with patch("image_embedder.main.Settings", return_value=settings):
        app = create_app(embedder=embedder)
        with TestClient(app) as client:
            client.get("/health")
        assert embedder.warmed_up is True


def test_app_warmup_skipped_when_disabled():
    embedder = FakeEmbedderWithWarmup()
    settings = Settings(warmup_on_startup=False)
    with patch("image_embedder.main.Settings", return_value=settings):
        app = create_app(embedder=embedder)
        with TestClient(app) as client:
            client.get("/health")
        assert embedder.warmed_up is False
