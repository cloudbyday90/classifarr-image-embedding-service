# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from fastapi.testclient import TestClient

from image_embedder.main import create_app


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
    app = create_app(embedder=EmbedderValueError())
    client = TestClient(app)
    response = client.post("/embed-image", json={"image_url": "https://example.com/poster.jpg"})
    assert response.status_code == 400
    assert response.json()["detail"] == "bad request"


def test_embed_image_generic_exception_maps_to_500():
    app = create_app(embedder=EmbedderException())
    client = TestClient(app)
    response = client.post("/embed-image", json={"image_url": "https://example.com/poster.jpg"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"

