# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Direct numeric-address transport with independently validated redirects."""

from collections.abc import Iterator
from contextlib import closing, contextmanager
from itertools import zip_longest
from urllib.parse import urljoin, urlsplit

import certifi
import urllib3
from urllib3.exceptions import ConnectTimeoutError, HTTPError, NewConnectionError
from urllib3.response import BaseHTTPResponse

from .config import Settings
from .input_limits import InputLimitExceeded
from .remote_url import RemoteDestination, check_url_text, resolve_remote_url

MAX_REDIRECTS = 3
MAX_CONNECT_ADDRESSES = 4
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class RemoteFetchError(RuntimeError):
    """Sanitized download failure retaining the API's server-error contract."""


def _pool(
    destination: RemoteDestination, address: str, timeout: float
) -> urllib3.HTTPConnectionPool:
    if destination.scheme == "https":
        return urllib3.HTTPSConnectionPool(
            address,
            port=destination.port,
            timeout=timeout,
            server_hostname=destination.host,
            assert_hostname=destination.host,
            cert_reqs="CERT_REQUIRED",
            ca_certs=certifi.where(),
        )
    return urllib3.HTTPConnectionPool(address, port=destination.port, timeout=timeout)


@contextmanager
def _response(
    destination: RemoteDestination, timeout: float
) -> Iterator[BaseHTTPResponse]:
    # Fallback only before a connection is established, within the approved set.
    first_ipv6 = ":" in destination.addresses[0]
    preferred = [a for a in destination.addresses if (":" in a) == first_ipv6]
    alternate = [a for a in destination.addresses if (":" in a) != first_ipv6]
    addresses = tuple(
        address
        for pair in zip_longest(preferred, alternate)
        for address in pair
        if address is not None
    )[:MAX_CONNECT_ADDRESSES]
    for index, address in enumerate(addresses):
        with closing(_pool(destination, address, timeout)) as pool:
            try:
                response = pool.urlopen(
                    "GET",
                    destination.target,
                    headers={
                        "Host": destination.authority,
                        "Accept-Encoding": "identity",
                    },
                    redirect=False,
                    retries=False,
                    preload_content=False,
                    assert_same_host=False,
                )
            except (ConnectTimeoutError, NewConnectionError):
                if index == len(addresses) - 1:
                    raise
                continue
            with closing(response):
                yield response
            return
    raise ValueError("Unable to resolve remote image host")


def _read_body(response: BaseHTTPResponse, max_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise ValueError("Invalid remote image response length") from exc
        if declared < 0:
            raise ValueError("Invalid remote image response length")
        if declared > max_bytes:
            raise InputLimitExceeded("Image payload exceeds maximum size")
    data = bytearray()
    for chunk in response.stream(amt=8192, decode_content=True):
        if len(data) + len(chunk) > max_bytes:
            raise InputLimitExceeded("Image payload exceeds maximum size")
        data.extend(chunk)
    return bytes(data)


def fetch_remote_image(image_url: str, settings: Settings) -> bytes:
    if not settings.allow_remote_urls:
        raise ValueError("Remote image URLs are disabled")
    current = image_url
    try:
        for hop in range(MAX_REDIRECTS + 1):
            destination = resolve_remote_url(current, settings.allowed_remote_hosts)
            with _response(destination, settings.request_timeout_seconds) as response:
                if response.status in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Invalid remote image redirect")
                    if hop == MAX_REDIRECTS:
                        raise ValueError("Remote image redirect limit exceeded")
                    check_url_text(location)
                    current = urljoin(destination.url, location)
                    if (
                        destination.scheme == "https"
                        and urlsplit(current).scheme == "http"
                    ):
                        raise ValueError(
                            "Remote image HTTPS downgrade is not supported"
                        )
                    continue
                if not 200 <= response.status < 300:
                    raise RemoteFetchError(
                        f"Remote image request failed (HTTP {response.status})"
                    )
                return _read_body(response, settings.max_image_bytes)
    except (HTTPError, OSError):
        # Route logging must not format transport exceptions containing URLs.
        raise RemoteFetchError("Unable to fetch remote image") from None
    raise ValueError("Remote image redirect limit exceeded")
