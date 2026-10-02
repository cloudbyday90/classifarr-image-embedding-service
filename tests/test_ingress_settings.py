# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Worker and ingress budgets reject invalid configuration and respect precedence."""

import pytest

from image_embedder import config
from image_embedder.config import Settings

LIMITS = [
    ("max_http_requests", "MAX_HTTP_REQUESTS", "max_http_requests"),
    ("server_workers", "IMAGE_EMBEDDER_WORKERS", "workers"),
    ("server_concurrency", "IMAGE_EMBEDDER_SERVER_CONCURRENCY", "limit_concurrency"),
    ("server_backlog", "IMAGE_EMBEDDER_SERVER_BACKLOG", "backlog"),
]


@pytest.mark.parametrize("field,env,key", LIMITS)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_constructor_and_toml_fail_closed(monkeypatch, field, env, key, value):
    with pytest.raises(ValueError, match="positive integer"):
        Settings(**{field: value})
    monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(config, "_TOML", {"server": {key: value}})
    with pytest.raises(ValueError, match="positive integer"):
        Settings()


@pytest.mark.parametrize("field,env,key", LIMITS)
def test_environment_wins_over_toml_and_ignores_web_concurrency(
    monkeypatch, field, env, key
):
    monkeypatch.setattr(config, "_TOML", {"server": {key: 3}})
    monkeypatch.setenv("WEB_CONCURRENCY", "100")
    monkeypatch.setenv(env, "2")
    assert getattr(Settings(), field) == 2
    monkeypatch.delenv(env)
    settings = Settings()
    assert getattr(settings, field) == 3
    if field != "server_workers":
        assert settings.server_workers == 1
    monkeypatch.setenv(env, "1.5")
    with pytest.raises(ValueError, match="positive integer"):
        Settings()
