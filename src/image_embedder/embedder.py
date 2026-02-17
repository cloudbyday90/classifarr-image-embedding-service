# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import binascii
import ipaddress
import io
import socket
import threading
from dataclasses import dataclass
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from .config import Settings

if TYPE_CHECKING:
    import torch
    from transformers import CLIPModel, CLIPProcessor

ModelTuple = Tuple[Any, Any, str]


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
        self._models: Dict[str, ModelTuple] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._model_locks_guard = threading.Lock()

    def list_models(self) -> List[ModelSpec]:
        return list(MODEL_CATALOG.values())

    def get_device_info(self) -> dict:
        try:
            device = self._resolve_device()
            info = {"type": str(device).split(":")[0]}
            if str(device).startswith("cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        info["name"] = torch.cuda.get_device_name(device)
                except Exception:
                    pass
            return info
        except Exception:
            return {"type": "unknown"}

    def get_model_status(self) -> List[dict]:
        result = []
        for spec in MODEL_CATALOG.values():
            result.append({
                "name": spec.name,
                "loaded": spec.name in self._models,
            })
        return result

    def get_memory_info(self) -> Optional[dict]:
        try:
            import torch
            device = self._resolve_device()
            if str(device).startswith("cuda") and torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
                }
        except Exception:
            pass
        return None

    def is_default_model_loaded(self) -> bool:
        spec = self.resolve_model(None)
        return spec.name in self._models

    def warmup(self, model_name: Optional[str] = None) -> ModelSpec:
        spec = self.resolve_model(model_name or self.settings.default_model)
        self._load_model(spec)
        return spec

    def resolve_model(self, model_name: Optional[str]) -> ModelSpec:
        candidate = model_name or self.settings.default_model
        default_spec = MODEL_CATALOG.get(self.settings.default_model)
        if default_spec is None:
            default_spec = next(iter(MODEL_CATALOG.values()))
        return MODEL_CATALOG.get(candidate, default_spec)

    def _resolve_device(self):
        import torch

        device_setting = (self.settings.device or "auto").strip().lower()
        if device_setting == "cpu":
            return torch.device("cpu")
        if device_setting == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available")
            return torch.device("cuda")
        if device_setting != "auto":
            raise ValueError(f"Unsupported DEVICE value: {self.settings.device}")

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self, spec: ModelSpec):
        if spec.name in self._models:
            return self._models[spec.name]

        with self._model_locks_guard:
            lock = self._model_locks.get(spec.name)
            if lock is None:
                lock = threading.Lock()
                self._model_locks[spec.name] = lock

        with lock:
            if spec.name in self._models:
                return self._models[spec.name]

        from transformers import CLIPModel, CLIPProcessor

        device = self._resolve_device()
        model = CLIPModel.from_pretrained(spec.hf_id)
        processor = CLIPProcessor.from_pretrained(spec.hf_id)
        model.to(device)  # type: ignore[arg-type]
        model.eval()  # type: ignore[union-attr]

        self._models[spec.name] = (model, processor, str(device))
        return self._models[spec.name]

    def _is_public_ip(self, ip_str: str) -> bool:
        ip = ipaddress.ip_address(ip_str)
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )

    def _validate_remote_url(self, image_url: str) -> None:
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http(s) image URLs are supported")
        if not parsed.hostname:
            raise ValueError("Invalid image URL")

        host = parsed.hostname.lower()

        if self.settings.allowed_remote_hosts:
            allowed = {h.strip().lower() for h in self.settings.allowed_remote_hosts if h.strip()}
            if host not in allowed:
                raise ValueError("Remote image host is not allowlisted")

        # Block obvious localhost aliases early.
        if host in {"localhost"}:
            raise ValueError("Remote image host resolves to a private address")

        # If it's a literal IP, enforce public-only.
        try:
            ipaddress.ip_address(host)
            if not self._is_public_ip(host):
                raise ValueError("Remote image host resolves to a private address")
            return
        except ValueError:
            pass

        # Resolve DNS and block private/reserved ranges.
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("Unable to resolve remote image host") from exc

        for info in infos:
            sockaddr = info[4]
            ip_str = sockaddr[0]
            if not self._is_public_ip(str(ip_str)):  # type: ignore[arg-type]
                raise ValueError("Remote image host resolves to a private address")

    def _fetch_image_bytes(self, image_url: str) -> bytes:
        if not self.settings.allow_remote_urls:
            raise ValueError("Remote image URLs are disabled")

        self._validate_remote_url(image_url)

        response = requests.get(
            image_url,
            timeout=self.settings.request_timeout_seconds,
            stream=True,
        )

        with closing(response):
            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.settings.max_image_bytes:
                raise ValueError("Image payload exceeds maximum size")

            buf = io.BytesIO()
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > self.settings.max_image_bytes:
                    raise ValueError("Image payload exceeds maximum size")
                buf.write(chunk)

            return buf.getvalue()

    def _decode_base64(self, image_base64: str) -> bytes:
        try:
            # validate=True rejects non-base64 characters instead of silently ignoring them.
            data = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image payload") from exc

        if len(data) > self.settings.max_image_bytes:
            raise ValueError("Image payload exceeds maximum size")
        return data

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
        target_size = spec.image_size if image_size is None else image_size
        if target_size <= 0:
            raise ValueError("image_size must be a positive integer")

        model_obj, processor, device = self._load_model(spec)
        image = self._load_image(image_url, image_base64)

        inputs = processor(  # type: ignore[operator]
            images=image,
            return_tensors="pt",
            size={"shortest_edge": target_size}
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            features = model_obj.get_image_features(**inputs)  # type: ignore[union-attr]
            if normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=-1)

        embedding = features[0].detach().cpu().numpy().astype(np.float32).tolist()
        dims = len(embedding)
        return embedding, dims, "local", spec.name, target_size
