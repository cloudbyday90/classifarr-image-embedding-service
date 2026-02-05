# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from fastapi.testclient import TestClient

from image_embedder.main import create_app


class FakeEmbedder:
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


def test_health():
    app = create_app(embedder=FakeEmbedder())
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    app = create_app(embedder=FakeEmbedder())
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["id"] == "ViT-L-14"


def test_embed_image():
    app = create_app(embedder=FakeEmbedder())
    client = TestClient(app)
    response = client.post("/embed-image", json={
        "image_url": "https://example.com/poster.jpg",
        "model": "ViT-L-14",
        "normalize": True,
        "image_size": 512
    })
    assert response.status_code == 200
    data = response.json()
    assert data["dims"] == 3
    assert data["model"] == "ViT-L-14"


def test_embed_image_requires_payload():
    app = create_app(embedder=FakeEmbedder())
    client = TestClient(app)
    response = client.post("/embed-image", json={
        "model": "ViT-L-14"
    })
    assert response.status_code == 400
