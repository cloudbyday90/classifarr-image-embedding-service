# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import base64
import binascii
import hashlib
import ipaddress
import io
import math
import socket
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from .config import Settings

if TYPE_CHECKING:
    import torch
    from transformers import CLIPVisionModelWithProjection, CLIPProcessor

ModelTuple = Tuple[Any, Any, str]
EmbedResult = Tuple[List[float], int, str, str, int]


@dataclass(frozen=True, slots=True)
class CachedEmbedding:
    """Immutable cache payload for one embedding result."""

    embedding: Tuple[float, ...]
    dims: int
    source: str
    model: str
    image_size: int

    @classmethod
    def from_result(cls, value: EmbedResult) -> "CachedEmbedding":
        embedding, dims, source, model, image_size = value
        return cls(tuple(embedding), dims, source, model, image_size)

    def to_result(self) -> EmbedResult:
        return (list(self.embedding), self.dims, self.source, self.model, self.image_size)


class EmbeddingLRUCache:
    """Thread-safe in-memory LRU cache for computed embeddings.

    Cache keys encode all axes that affect the output:
    ``sha256(image_bytes) | model_name | image_size | normalize``
    where *image_bytes* is the effective image content embedded by the service.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def make_key(
        image_bytes: bytes,
        model_name: str,
        image_size: int,
        normalize: bool,
    ) -> str:
        """Return a deterministic, hashable cache key for the given request."""
        content_hash = hashlib.sha256(image_bytes, usedforsecurity=False).hexdigest()
        return f"{content_hash}|{model_name}|{image_size}|{normalize}"

    def get(self, key: str) -> Optional[EmbedResult]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key].to_result()
            self._misses += 1
            return None

    def put(self, key: str, value: Union[CachedEmbedding, EmbedResult]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                    self._evictions += 1
            if isinstance(value, CachedEmbedding):
                self._cache[key] = value
            else:
                self._cache[key] = CachedEmbedding.from_result(value)

    def info(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0


@dataclass
class BatchItem:
    """Descriptor for a single image within a batch embed call."""

    image_url: Optional[str]
    image_base64: Optional[str]
    normalize: bool


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
        self._embed_count = 0
        self._embed_count_lock = threading.Lock()
        self._embedding_cache: Optional[EmbeddingLRUCache] = (
            EmbeddingLRUCache(self.settings.embed_cache_size)
            if self.settings.embed_cache_size > 0
            else None
        )

    def list_models(self) -> List[ModelSpec]:
        return list(MODEL_CATALOG.values())

    def get_device_info(self) -> dict:
        try:
            device = self._resolve_device()
            device_str = str(device)
            if device_str.startswith("ov:"):
                ov_device = device_str[len("ov:"):]
                info: dict = {"type": "openvino", "ov_device": ov_device}
                try:
                    import openvino as ov
                    core = ov.Core()
                    available = core.available_devices
                    info["available_devices"] = available
                except Exception:
                    pass
                return info
            info = {"type": device_str.split(":")[0]}
            if device_str.startswith("cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        info["name"] = torch.cuda.get_device_name(device)
                        if getattr(torch.version, "hip", None) is not None:
                            info["type"] = "rocm"
                            info["hip_version"] = torch.version.hip
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
            device = self._resolve_device()
            device_str = str(device)
            if device_str.startswith("ov:"):
                # OpenVINO does not expose a Python API for GPU memory stats.
                return None
            import torch
            if device_str.startswith("cuda") and torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
                }
        except Exception:
            pass
        return None

    def get_cache_info(self) -> Optional[dict]:
        """Return cache statistics, or None when caching is disabled."""
        if self._embedding_cache is None:
            return None
        return self._embedding_cache.info()

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
        device_setting = (self.settings.device or "auto").strip().lower()

        # OpenVINO device family: "openvino", "openvino:CPU", "openvino:GPU",
        # "openvino:AUTO", "openvino:GPU.0", …
        # Return a plain string prefixed with "ov:" so the rest of the code can
        # distinguish OV paths from torch device objects without importing torch.
        if device_setting == "openvino" or device_setting.startswith("openvino:"):
            if device_setting == "openvino":
                ov_device = "AUTO"
            else:
                ov_device = self.settings.device.split(":", 1)[1].upper()  # type: ignore[union-attr]
            return f"ov:{ov_device}"

        if device_setting == "rocm":
            if sys.platform == "win32":
                raise ValueError(
                    "ROCm (AMD GPU) is not supported on Windows. "
                    "ROCm requires a Linux host. "
                    "See the README for AMD GPU setup instructions."
                )
            import torch
            if not torch.cuda.is_available():
                raise ValueError("ROCm requested but no ROCm-capable AMD GPU is available")
            return torch.device("cuda")

        import torch

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

            device = self._resolve_device()

            if isinstance(device, str) and device.startswith("ov:"):
                return self._load_model_openvino(spec, device)

            from transformers import CLIPVisionModelWithProjection, CLIPProcessor

            model = CLIPVisionModelWithProjection.from_pretrained(spec.hf_id)
            processor = CLIPProcessor.from_pretrained(spec.hf_id)
            model.to(device)  # type: ignore[arg-type]
            model.eval()  # type: ignore[union-attr]

            self._models[spec.name] = (model, processor, str(device))
            return self._models[spec.name]

    def _load_model_openvino(self, spec: ModelSpec, ov_device_str: str):
        """Load (or export-then-cache) a CLIP model for OpenVINO inference.

        On the first call the HuggingFace model is loaded via PyTorch, converted
        to OpenVINO IR format (.xml + .bin) with ``openvino.convert_model()``,
        saved to *OV_MODEL_CACHE*, and compiled for the requested device.
        Subsequent calls find the cached IR on disk and skip the export step,
        so torch is not needed on the hot path after the first startup.

        The stored tuple is ``(compiled_model, processor, ov_device_str)`` where
        ``compiled_model`` is an ``openvino.CompiledModel``.  The rest of the
        inference code detects the OV path by checking whether the device string
        starts with ``"ov:"``.
        """
        import os
        import pathlib
        import openvino as ov
        from transformers import CLIPProcessor

        ov_device = ov_device_str[len("ov:"):]

        cache_dir = pathlib.Path(
            os.environ.get("OV_MODEL_CACHE", "/app/.cache/ov_ir")
        ) / spec.name.replace("/", "_")
        xml_path = cache_dir / "model.xml"

        if not xml_path.exists():
            # Export: load PyTorch model, trace + convert to OV IR, save to disk.
            import torch
            from transformers import CLIPVisionModelWithProjection

            torch_model = CLIPVisionModelWithProjection.from_pretrained(spec.hf_id)
            torch_model.eval()

            dummy_input = {"pixel_values": torch.zeros(1, 3, spec.image_size, spec.image_size)}
            ov_model = ov.convert_model(torch_model, example_input=dummy_input)

            cache_dir.mkdir(parents=True, exist_ok=True)
            ov.save_model(ov_model, str(xml_path))
        else:
            core_tmp = ov.Core()
            ov_model = core_tmp.read_model(str(xml_path))

        core = ov.Core()
        compiled = core.compile_model(ov_model, ov_device)

        processor = CLIPProcessor.from_pretrained(spec.hf_id)
        self._models[spec.name] = (compiled, processor, ov_device_str)
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

    def _resolve_image_bytes(self, image_url: Optional[str], image_base64: Optional[str]) -> bytes:
        if image_base64:
            return self._decode_base64(image_base64)
        if image_url:
            return self._fetch_image_bytes(image_url)
        raise ValueError("image_url or image_base64 is required")

    def _image_from_bytes(self, data: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise ValueError("Unable to decode image bytes") from exc

    def _load_image(self, image_url: Optional[str], image_base64: Optional[str]) -> Image.Image:
        data = self._resolve_image_bytes(image_url, image_base64)
        return self._image_from_bytes(data)

    @staticmethod
    def _normalize_embedding_np(feat: "np.ndarray", eps: float = 1e-12) -> "np.ndarray":
        """Normalize *feat* along the last axis, matching PyTorch's F.normalize semantics.

        Raises ``ValueError`` for non-finite inputs or outputs.
        Divides by ``max(||v||_2, eps)`` to avoid division by zero.
        """
        if not np.all(np.isfinite(feat)):
            raise ValueError("Embedding contains non-finite values before normalization")
        norm = np.linalg.norm(feat, axis=-1, keepdims=True)
        normalized = feat / np.maximum(norm, eps)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Normalization produced non-finite values")
        return normalized

    def _validate_embedding_result(
        self,
        spec: "ModelSpec",
        embedding: List[float],
        dims: int,
    ) -> None:
        """Assert invariants that every successful embedding result must satisfy.

        Raises ``ValueError`` with a descriptive message if any invariant is broken.
        """
        if any(isinstance(v, (list, np.ndarray)) for v in embedding):
            raise ValueError(
                f"Embedding must be a 1-D vector but contains nested values (model={spec.name})"
            )
        if len(embedding) != dims:
            raise ValueError(
                f"dims={dims} does not match len(embedding)={len(embedding)} (model={spec.name})"
            )
        if dims != spec.dims:
            raise ValueError(
                f"dims={dims} does not match spec.dims={spec.dims} for model={spec.name}"
            )
        if not all(math.isfinite(v) for v in embedding):
            raise ValueError(
                f"Embedding for model={spec.name} contains non-finite values (NaN or Inf)"
            )

    def embed(
        self,
        image_url: Optional[str],
        image_base64: Optional[str],
        model: Optional[str],
        normalize: bool,
        image_size: Optional[int]
    ) -> Tuple[List[float], int, str, str, int]:
        spec = self.resolve_model(model)
        if image_size is not None and image_size != spec.image_size:
            raise ValueError(
                f"image_size={image_size} is not supported for {spec.name}; "
                f"this model only accepts image_size={spec.image_size}"
            )
        target_size = spec.image_size
        if target_size <= 0:
            raise ValueError("image_size must be a positive integer")

        image_bytes = self._resolve_image_bytes(image_url, image_base64)

        # Cache check — skip inference entirely on a hit.
        if self._embedding_cache is not None:
            cache_key = EmbeddingLRUCache.make_key(
                image_bytes, spec.name, target_size, normalize
            )
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            cache_key = ""

        model_obj, processor, device = self._load_model(spec)
        image = self._image_from_bytes(image_bytes)

        if device.startswith("ov:"):
            # OpenVINO path: processor returns numpy tensors; compiled model
            # returns output[0] which is image_embeds from CLIPVisionModelWithProjection.
            inputs = processor(  # type: ignore[operator]
                images=image,
                return_tensors="np",
                size={"shortest_edge": target_size},
            )
            raw_result = model_obj(dict(inputs))  # type: ignore[operator]
            raw_output = raw_result[0].astype(np.float32)  # expected shape (1, dims)
            if raw_output.ndim != 2 or raw_output.shape[0] != 1 or raw_output.shape[1] != spec.dims:
                raise ValueError(
                    f"OpenVINO model returned unexpected output shape {raw_output.shape!r} "
                    f"(expected (1, {spec.dims})) for model={spec.name}"
                )
            feat = raw_output[0]  # shape (dims,)
            if normalize:
                feat = self._normalize_embedding_np(feat)
            embedding = feat.tolist()
        else:
            inputs = processor(  # type: ignore[operator]
                images=image,
                return_tensors="pt",
                size={"shortest_edge": target_size},
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            import torch

            with torch.no_grad():
                features = model_obj(**inputs).image_embeds  # type: ignore[operator]
                if normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)

            embedding = features[0].detach().cpu().numpy().astype(np.float32).tolist()

        dims = len(embedding)
        self._validate_embedding_result(spec, embedding, dims)

        if self.settings.embed_cleanup_every_n > 0:
            with self._embed_count_lock:
                self._embed_count += 1
                trigger = (self._embed_count % self.settings.embed_cleanup_every_n == 0)
            if trigger:
                from .memory import cleanup_gpu_memory
                cleanup_gpu_memory(device)

        result = (embedding, dims, "local", spec.name, target_size)
        if self._embedding_cache is not None and cache_key:
            self._embedding_cache.put(cache_key, result)
        return result

    def embed_batch(
        self,
        spec: ModelSpec,
        target_size: int,
        items: List[BatchItem],
    ) -> List[Union[Tuple[List[float], int, str, str, int], Exception]]:
        """Run a batch of images through the model in one forward pass.

        All items share the same resolved *spec* and *target_size*.
        Returns one result (or ``Exception``) per item, in the same order.
        Per-item image-load errors are returned as exceptions rather than
        aborting the whole batch.
        """
        if not items:
            return []

        # Pre-check cache: items already computed don't need model inference.
        outcomes: List[Any] = [None] * len(items)
        uncached_indices: List[int] = []
        uncached_payloads: List[Tuple[BatchItem, bytes]] = []
        uncached_cache_keys: List[str] = []
        if self._embedding_cache is not None:
            for i, item in enumerate(items):
                try:
                    image_bytes = self._resolve_image_bytes(item.image_url, item.image_base64)
                except Exception as exc:
                    outcomes[i] = exc
                    continue

                key = EmbeddingLRUCache.make_key(
                    image_bytes, spec.name, target_size, item.normalize
                )
                cached = self._embedding_cache.get(key)
                if cached is not None:
                    outcomes[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_payloads.append((item, image_bytes))
                    uncached_cache_keys.append(key)
        else:
            for i, item in enumerate(items):
                try:
                    image_bytes = self._resolve_image_bytes(item.image_url, item.image_base64)
                except Exception as exc:
                    outcomes[i] = exc
                    continue
                uncached_indices.append(i)
                uncached_payloads.append((item, image_bytes))

        # If everything was cached, skip model loading entirely.
        if not uncached_indices:
            return outcomes

        uncached_items = [payload[0] for payload in uncached_payloads]
        model_obj, processor, device = self._load_model(spec)

        # Decode images, capturing per-item errors so one bad image
        # doesn't abort the whole batch.
        pil_images: List[Optional[Image.Image]] = []
        load_errors: List[Optional[Exception]] = []
        for _item, image_bytes in uncached_payloads:
            try:
                pil_images.append(self._image_from_bytes(image_bytes))
                load_errors.append(None)
            except Exception as exc:
                pil_images.append(None)
                load_errors.append(exc)

        valid_sub_idx = [i for i, img in enumerate(pil_images) if img is not None]
        valid_images = [pil_images[i] for i in valid_sub_idx]

        # Partial outcomes for uncached items (index within uncached_items).
        uncached_outcomes: List[Any] = list(load_errors)

        if valid_images:
            if device.startswith("ov:"):
                # OpenVINO path: batch all valid images in one compiled model call.
                inputs = processor(  # type: ignore[operator]
                    images=valid_images,
                    return_tensors="np",
                    size={"shortest_edge": target_size},
                )
                raw_result = model_obj(dict(inputs))  # type: ignore[operator]
                n_valid = len(valid_images)
                raw_output = raw_result[0].astype(np.float32)  # expected shape (N, dims)
                if raw_output.ndim != 2 or raw_output.shape[0] != n_valid or raw_output.shape[1] != spec.dims:
                    raise ValueError(
                        f"OpenVINO model returned unexpected output shape {raw_output.shape!r} "
                        f"(expected ({n_valid}, {spec.dims})) for model={spec.name}"
                    )
                features_np = raw_output

                for batch_pos, sub_idx in enumerate(valid_sub_idx):
                    feat = features_np[batch_pos]
                    try:
                        if uncached_items[sub_idx].normalize:
                            feat = self._normalize_embedding_np(feat)
                        embedding_list = feat.tolist()
                        dims = len(embedding_list)
                        self._validate_embedding_result(spec, embedding_list, dims)
                        uncached_outcomes[sub_idx] = (embedding_list, dims, "local", spec.name, target_size)
                    except ValueError as exc:
                        uncached_outcomes[sub_idx] = exc
            else:
                import torch

                inputs = processor(  # type: ignore[operator]
                    images=valid_images,
                    return_tensors="pt",
                    size={"shortest_edge": target_size},
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    features = model_obj(**inputs).image_embeds  # type: ignore[operator]

                for batch_pos, sub_idx in enumerate(valid_sub_idx):
                    feat = features[batch_pos]
                    if uncached_items[sub_idx].normalize:
                        feat = torch.nn.functional.normalize(
                            feat.unsqueeze(0), p=2, dim=-1
                        ).squeeze(0)
                    embedding = feat.detach().cpu().numpy().astype(np.float32).tolist()
                    try:
                        self._validate_embedding_result(spec, embedding, len(embedding))
                        uncached_outcomes[sub_idx] = (embedding, len(embedding), "local", spec.name, target_size)
                    except ValueError as exc:
                        uncached_outcomes[sub_idx] = exc

        # Merge uncached results back into the full outcomes list.
        for sub_idx, orig_idx in enumerate(uncached_indices):
            outcomes[orig_idx] = uncached_outcomes[sub_idx]

        # Populate cache with new successful results.
        if self._embedding_cache is not None:
            for sub_idx, key in enumerate(uncached_cache_keys):
                outcome = uncached_outcomes[sub_idx]
                if not isinstance(outcome, Exception) and outcome is not None:
                    self._embedding_cache.put(key, outcome)

        # Cleanup tracking — count successful embeds.
        n_success = sum(1 for o in outcomes if not isinstance(o, Exception) and o is not None)
        if self.settings.embed_cleanup_every_n > 0 and n_success > 0:
            n = self.settings.embed_cleanup_every_n
            with self._embed_count_lock:
                before = self._embed_count
                self._embed_count += n_success
                after = self._embed_count
            if (after // n) > (before // n):
                from .memory import cleanup_gpu_memory
                cleanup_gpu_memory(device)

        return outcomes
