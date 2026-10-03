# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Supported CPython/backend targets for dependency resolution and installation."""

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyProfile:
    name: str
    backend: str
    architecture: str
    inputs: tuple[str, ...]
    torch_version: str | None = None
    torch_index: str | None = None


def architecture(machine: str) -> str:
    aliases = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    try:
        return aliases[machine.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dependency architecture: {machine}") from error


def runtime_target() -> str:
    if (
        platform.system() != "Linux"
        or sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
    ):
        raise ValueError("Locked environments require Linux CPython 3.12")
    return architecture(platform.machine())


def _profiles() -> dict[str, DependencyProfile]:
    profiles = {
        "bootstrap": DependencyProfile(
            "bootstrap", "bootstrap", "any", ("requirements-bootstrap.txt",)
        ),
    }
    for arch in ("amd64", "arm64"):
        for backend in ("cpu", "audit"):
            name = f"{backend}-py312-linux-{arch}"
            profiles[name] = DependencyProfile(
                name,
                backend,
                arch,
                ("requirements.txt", "requirements-torch-cpu.txt")
                if backend == "cpu"
                else ("requirements-audit.txt",),
                "2.14.1+cpu" if backend == "cpu" else None,
                "https://download.pytorch.org/whl/cpu" if backend == "cpu" else None,
            )
    for backend, inputs, version, index in (
        (
            "qa",
            ("requirements.txt", "requirements-dev.txt", "requirements-torch-cpu.txt"),
            "2.14.1+cpu",
            "cpu",
        ),
        (
            "openvino",
            (
                "requirements.txt",
                "requirements-openvino.txt",
                "requirements-torch-cpu.txt",
            ),
            "2.14.1+cpu",
            "cpu",
        ),
        (
            "cuda",
            ("requirements.txt", "requirements-torch-cuda.txt"),
            "2.14.1+cu130",
            "cu130",
        ),
        (
            "cuda-legacy",
            ("requirements.txt", "requirements-torch-cuda-legacy.txt"),
            "2.14.1+cu126",
            "cu126",
        ),
    ):
        name = f"{backend}-py312-linux-amd64"
        profiles[name] = DependencyProfile(
            name,
            backend,
            "amd64",
            inputs,
            version,
            f"https://download.pytorch.org/whl/{index}",
        )
    return profiles


PROFILES = _profiles()


def selected_profile(backend: str, arch: str | None = None) -> DependencyProfile:
    target = arch or runtime_target()
    name = "bootstrap" if backend == "bootstrap" else f"{backend}-py312-linux-{target}"
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dependency profile: {name}") from error


def check_environment(profile: DependencyProfile) -> None:
    target = runtime_target()
    if profile.architecture not in ("any", target):
        raise ValueError(
            f"Profile {profile.name} does not match this {target} interpreter"
        )
