# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from image_embedder.config import Settings
from image_embedder.logging_config import setup_logging, JsonFormatter, get_logger
from image_embedder.memory import cleanup_gpu_memory, get_memory_usage, check_memory_health, force_cleanup
from image_embedder.main import create_app


class FakeEmbedderWithCleanup:
    def __init__(self):
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
        pass

    def get_device_info(self):
        return {"type": "cpu"}

    def get_model_status(self):
        return [
            {"name": "ViT-L-14", "loaded": True},
            {"name": "ViT-B-16", "loaded": False},
        ]

    def get_memory_info(self):
        return None

    def is_default_model_loaded(self):
        return True


def test_setup_logging_creates_logger():
    logger = setup_logging(level="DEBUG")
    assert logger is not None
    assert logger.name == "image_embedder"
    assert logger.level == 10


def test_setup_logging_with_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="INFO", log_file=str(log_file))
    logger.info("Test message")
    
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content


def test_json_formatter():
    formatter = JsonFormatter()
    import logging
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    import json
    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["message"] == "Test message"


def test_get_logger():
    logger = get_logger("test.module")
    assert logger.name == "test.module"


def test_cleanup_gpu_memory_cpu():
    result = cleanup_gpu_memory(device="cpu")
    assert "gc_collected" in result
    assert "gpu_freed_mb" in result
    assert isinstance(result["gc_collected"], int)


def test_get_memory_usage():
    usage = get_memory_usage()
    assert "process_rss_mb" in usage
    assert "gpu_allocated_mb" in usage
    assert "gpu_reserved_mb" in usage


def test_check_memory_health_no_limits():
    is_healthy, issues = check_memory_health()
    assert is_healthy is True
    assert len(issues) == 0


def test_check_memory_health_with_limits():
    is_healthy, issues = check_memory_health(max_process_mb=1)
    if usage := get_memory_usage():
        if usage["process_rss_mb"] and usage["process_rss_mb"] > 1:
            assert is_healthy is False
            assert len(issues) > 0


def test_force_cleanup():
    result = force_cleanup()
    assert "gc_collected" in result
    assert "gpu_freed_mb" in result


def test_config_logging_settings():
    settings = Settings(
        log_level="DEBUG",
        log_max_bytes=1024,
        log_backup_count=3,
        log_json_format=True,
    )
    assert settings.log_level == "DEBUG"
    assert settings.log_max_bytes == 1024
    assert settings.log_backup_count == 3
    assert settings.log_json_format is True


def test_config_memory_settings():
    settings = Settings(
        memory_cleanup_interval_seconds=60,
        max_process_memory_mb=1024,
        max_gpu_memory_mb=512,
        cleanup_on_shutdown=False,
    )
    assert settings.memory_cleanup_interval_seconds == 60
    assert settings.max_process_memory_mb == 1024
    assert settings.max_gpu_memory_mb == 512
    assert settings.cleanup_on_shutdown is False


def test_config_shutdown_settings():
    settings = Settings(shutdown_timeout_seconds=45)
    assert settings.shutdown_timeout_seconds == 45


def test_cleanup_endpoint():
    embedder = FakeEmbedderWithCleanup()
    settings = Settings(warmup_on_startup=False)
    with patch("image_embedder.main.Settings", return_value=settings):
        app = create_app(embedder=embedder)
        with TestClient(app) as client:
            response = client.post("/admin/cleanup")
            assert response.status_code == 200
            data = response.json()
            assert "gc_collected" in data
            assert "gpu_freed_mb" in data


def test_global_exception_handler():
    embedder = FakeEmbedderWithCleanup()
    
    def boom(*args, **kwargs):
        raise RuntimeError("Test error")
    
    embedder.embed = boom
    
    settings = Settings(warmup_on_startup=False)
    with patch("image_embedder.main.Settings", return_value=settings):
        app = create_app(embedder=embedder)
        with TestClient(app) as client:
            response = client.post("/embed-image", json={
                "image_url": "https://example.com/test.jpg",
            })
            assert response.status_code == 500
