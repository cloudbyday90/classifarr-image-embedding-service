# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Finite embedding deadlines with compatibility for explicit legacy settings."""

import math


def positive_duration(value: object, name: str = "embedding_timeout_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{name} must be positive finite seconds")
    try:
        seconds = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be positive finite seconds") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be positive finite seconds")
    return seconds


def embedding_deadline(
    explicit: object, legacy: float, *, legacy_env_set: bool
) -> float:
    if explicit is not None:
        return positive_duration(explicit)
    # Preserve custom constructor/TOML values and even an explicit legacy default env value.
    if legacy_env_set or legacy != 15:
        return positive_duration(legacy)
    return 45.0
