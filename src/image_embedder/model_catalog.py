# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Immutable identities for the supported publisher model snapshots."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    hf_id: str
    dims: int
    image_size: int
    # Unverified synthetic specs may be used by offline tests, never by loaders.
    revision: str = ""
    assets: tuple[tuple[str, str], ...] = ()

    def asset_digest(self, filename: str) -> str:
        for name, digest in self.assets:
            if name == filename:
                return digest
        raise ValueError(f"Unverified model asset: {self.name}/{filename}")

    @property
    def weights_file(self) -> str:
        for name, _digest in self.assets:
            if name in {"model.safetensors", "pytorch_model.bin"}:
                return name
        raise ValueError(f"No verified weights for {self.name}")


MODEL_CATALOG = MappingProxyType(
    {
        "ViT-L-14": ModelSpec(
            name="ViT-L-14",
            hf_id="openai/clip-vit-large-patch14",
            dims=768,
            image_size=224,
            revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
            assets=(
                (
                    "config.json",
                    "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
                ),
                (
                    "preprocessor_config.json",
                    "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
                ),
                (
                    "model.safetensors",
                    "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
                ),
            ),
        ),
        "ViT-B-16": ModelSpec(
            name="ViT-B-16",
            hf_id="openai/clip-vit-base-patch16",
            dims=512,
            image_size=224,
            revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
            assets=(
                (
                    "config.json",
                    "eaf1c9089a8553c913d27ea66407f8bfc2be9989c80c9f331ddb3d63d4c5e8ad",
                ),
                (
                    "preprocessor_config.json",
                    "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
                ),
                (
                    "pytorch_model.bin",
                    "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0",
                ),
            ),
        ),
    }
)
