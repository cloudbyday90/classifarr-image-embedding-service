# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from .config import Settings


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    dims: int
    image_size: int


MODEL_CATALOG: Dict[str, ModelSpec] = {
    "ViT-L-14": ModelSpec(
        name="ViT-L-14",
        hf_id="openai/clip-vit-large-patch14",
        dims=768,
        image_size=224
    ),
    "ViT-B-16": ModelSpec(
        name="ViT-B-16",
        hf_id="openai/clip-vit-base-patch16",
        dims=512,
        image_size=224
    )
}


class ImageEmbedder:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._models: Dict[str, Tuple[object, object, str]] = {}

    def list_models(self) -> List[ModelSpec]:
        return list(MODEL_CATALOG.values())

    def resolve_model(self, model_name: Optional[str]) -> ModelSpec:
        candidate = model_name or self.settings.default_model
        default_spec = MODEL_CATALOG.get(self.settings.default_model)
        if default_spec is None:
            default_spec = next(iter(MODEL_CATALOG.values()))
        return MODEL_CATALOG.get(candidate, default_spec)

    def _resolve_device(self):
        import torch

        if self.settings.device.lower() == "cpu":
            return torch.device("cpu")
        if self.settings.device.lower() == "cuda":
            return torch.device("cuda")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self, spec: ModelSpec):
        if spec.name in self._models:
            return self._models[spec.name]

        from transformers import CLIPModel, CLIPProcessor

        device = self._resolve_device()
        model = CLIPModel.from_pretrained(spec.hf_id)
        processor = CLIPProcessor.from_pretrained(spec.hf_id)
        model.to(device)
        model.eval()

        self._models[spec.name] = (model, processor, str(device))
        return self._models[spec.name]

    def _fetch_image_bytes(self, image_url: str) -> bytes:
        if not self.settings.allow_remote_urls:
            raise ValueError("Remote image URLs are disabled")

        response = requests.get(
            image_url,
            timeout=self.settings.request_timeout_seconds,
            stream=True
        )
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > self.settings.max_image_bytes:
            raise ValueError("Image payload exceeds maximum size")

        data = response.content
        if len(data) > self.settings.max_image_bytes:
            raise ValueError("Image payload exceeds maximum size")

        return data

    def _decode_base64(self, image_base64: str) -> bytes:
        try:
            data = base64.b64decode(image_base64)
            if len(data) > self.settings.max_image_bytes:
                raise ValueError("Image payload exceeds maximum size")
            return data
        except Exception as exc:
            raise ValueError("Invalid base64 image payload") from exc

    def _load_image(self, image_url: Optional[str], image_base64: Optional[str]) -> Image.Image:
        if image_base64:
            data = self._decode_base64(image_base64)
        elif image_url:
            data = self._fetch_image_bytes(image_url)
        else:
            raise ValueError("image_url or image_base64 is required")

        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise ValueError("Unable to decode image bytes") from exc

    def embed(
        self,
        image_url: Optional[str],
        image_base64: Optional[str],
        model: Optional[str],
        normalize: bool,
        image_size: Optional[int]
    ) -> Tuple[List[float], int, str, str, int]:
        spec = self.resolve_model(model)
        model_obj, processor, device = self._load_model(spec)

        image = self._load_image(image_url, image_base64)
        target_size = image_size or spec.image_size

        inputs = processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": target_size}
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            features = model_obj.get_image_features(**inputs)
            if normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=-1)

        embedding = features[0].detach().cpu().numpy().astype(np.float32).tolist()
        dims = len(embedding)
        return embedding, dims, "local", spec.name, target_size
