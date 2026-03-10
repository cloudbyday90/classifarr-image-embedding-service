# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared test fixtures."""

import pytest
from image_embedder.config import Settings


@pytest.fixture()
def no_auth_settings(**kwargs):
    """Return a Settings object with auth disabled for unit tests that test non-auth behaviour."""
    def _factory(**overrides):
        s = Settings()
        s.require_api_key = False
        s.warmup_on_startup = False
        for k, v in overrides.items():
            setattr(s, k, v)
        return s
    return _factory
