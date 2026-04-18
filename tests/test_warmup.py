# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder, MODEL_CATALOG
from image_embedder.main import create_app
from fakes import FakeEmbedder


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


class FakeEmbedderWithWarmup(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.warmed_up = False

    def warmup(self, model_name=None):
        self.warmed_up = True

    def get_model_status(self):
        return [{"name": "ViT-L-14", "loaded": self.warmed_up}]

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
