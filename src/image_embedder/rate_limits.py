# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stable quota identities without retaining caller-supplied credentials."""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import Settings
from .credentials import extract_api_key, matches_api_key


def client_address_key(request: Request) -> str:
    # The server owns proxy trust; never interpret forwarding headers here.
    return f"client:{get_remote_address(request)}"


def make_limiter(settings: Settings) -> Limiter:
    def verified_identity(request: Request) -> str:
        candidate = extract_api_key(
            request.headers.get("x-api-key"), request.headers.get("authorization")
        )
        if matches_api_key(candidate, settings.service_api_key):
            # There is one configured shared credential, hence one principal.
            # Neither its value nor arbitrary invalid values enter quota storage.
            return "credential:service"
        return client_address_key(request)

    return Limiter(key_func=verified_identity)
