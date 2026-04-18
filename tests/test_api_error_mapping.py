# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from fastapi.testclient import TestClient

from image_embedder.main import create_app
from fakes import _no_auth_settings


class EmbedderValueError:
    def list_models(self):
        return []

    def embed(self, *args, **kwargs):
        raise ValueError("bad request")


class EmbedderException:
    def list_models(self):
        return []

    def embed(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_embed_image_value_error_maps_to_400():
    app = create_app(embedder=EmbedderValueError(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.post("/embed-image", json={"image_url": "https://example.com/poster.jpg"})
    assert response.status_code == 400
    assert response.json()["detail"] == "bad request"


def test_embed_image_generic_exception_maps_to_500():
    app = create_app(embedder=EmbedderException(), settings=_no_auth_settings())
    client = TestClient(app)
    response = client.post("/embed-image", json={"image_url": "https://example.com/poster.jpg"})
    assert response.status_code == 500
    # Internal exception details must NOT be leaked to the client (OWASP A05 / security hardening).
    assert response.json()["detail"] == "Internal server error"
    assert "boom" not in response.text

