# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded, non-executable messages for the disposable download worker."""

import json
import re
from dataclasses import dataclass

from .deadlines import positive_duration
from .input_limits import InputLimitExceeded
from .remote_fetch import RemoteFetchError

MAX_REQUEST_BYTES = 64 * 1024
MAX_URL_CHARACTERS = 8192
MAX_ERROR_BYTES = 1024
_SAFE_VALUES = frozenset(
    {
        "Invalid image URL",
        "Only http(s) image URLs are supported",
        "Image URL credentials are not supported",
        "Remote image URLs are disabled",
        "Remote image host is not allowlisted",
        "Remote image host resolves to a private address",
        "Unable to resolve remote image host",
        "Invalid remote image response length",
        "Invalid remote image redirect",
        "Remote image redirect limit exceeded",
        "Remote image HTTPS downgrade is not supported",
    }
)
_LIMIT = "Image payload exceeds maximum size"
_FAILURE = "Unable to fetch remote image"


@dataclass(frozen=True, slots=True)
class RemoteFetchOptions:
    allowed_remote_hosts: list[str]
    max_image_bytes: int
    request_timeout_seconds: float
    allow_remote_urls: bool = True


def encode_request(url: str, options: RemoteFetchOptions) -> bytes:
    if not isinstance(url, str) or not 0 < len(url) <= MAX_URL_CHARACTERS:
        raise ValueError("Invalid image URL")
    data = json.dumps(
        {
            "url": url,
            "hosts": options.allowed_remote_hosts,
            "max_bytes": options.max_image_bytes,
            "hop_timeout": options.request_timeout_seconds,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("Invalid image URL")
    return data


def decode_request(data: bytes) -> tuple[str, RemoteFetchOptions]:
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("Invalid image URL")
    value = json.loads(data)
    if not isinstance(value, dict) or set(value) != {
        "url",
        "hosts",
        "max_bytes",
        "hop_timeout",
    }:
        raise ValueError("Invalid image URL")
    url, hosts, maximum = value["url"], value["hosts"], value["max_bytes"]
    if (
        not isinstance(url, str)
        or not 0 < len(url) <= MAX_URL_CHARACTERS
        or not isinstance(hosts, list)
        or not all(isinstance(h, str) for h in hosts)
        or type(maximum) is not int
        or maximum <= 0
    ):
        raise ValueError("Invalid image URL")
    timeout = positive_duration(value["hop_timeout"], "request_timeout_seconds")
    return url, RemoteFetchOptions(hosts, maximum, timeout)


def encode_error(error: Exception) -> bytes:
    if isinstance(error, InputLimitExceeded):
        return b"I" + _LIMIT.encode("ascii")
    if isinstance(error, ValueError) and str(error) in _SAFE_VALUES:
        return b"V" + str(error).encode("ascii")
    detail = str(error) if isinstance(error, RemoteFetchError) else ""
    if not re.fullmatch(r"Remote image request failed \(HTTP [1-5][0-9]{2}\)", detail):
        detail = _FAILURE
    return b"F" + detail.encode("ascii")


def decode_response(data: bytes, maximum: int) -> bytes:
    if data.startswith(b"O") and len(data) <= maximum + 1:
        return data[1:]
    if not data or len(data) > MAX_ERROR_BYTES:
        raise RemoteFetchError(_FAILURE)
    try:
        detail = data[1:].decode("ascii")
    except UnicodeError:
        raise RemoteFetchError(_FAILURE) from None
    if data[:1] == b"I" and detail == _LIMIT:
        raise InputLimitExceeded(detail)
    if data[:1] == b"V" and detail in _SAFE_VALUES:
        raise ValueError(detail)
    if data[:1] == b"F" and re.fullmatch(
        r"Remote image request failed \(HTTP [1-5][0-9]{2}\)", detail
    ):
        raise RemoteFetchError(detail)
    raise RemoteFetchError(_FAILURE)
