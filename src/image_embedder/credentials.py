# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared header precedence and configured API-key comparison."""

import hmac


def extract_api_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:]
    return None


def matches_api_key(candidate: str | None, configured: str | None) -> bool:
    return bool(
        candidate
        and configured
        and hmac.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8"))
    )
