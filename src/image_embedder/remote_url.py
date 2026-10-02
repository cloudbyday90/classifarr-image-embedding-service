# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Canonical URL policy and public address resolution for one remote hop."""

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import requests

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def is_public_address(value: str) -> bool:
    if "%" in value:
        return False
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address):
        embedded = address.ipv4_mapped
        if address in _NAT64:
            embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        if embedded is not None and not is_public_address(str(embedded)):
            return False
    return address.is_global and not address.is_multicast and not address.is_reserved


def check_url_text(value: str) -> None:
    if (
        not value
        or value.startswith(" ")
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Invalid image URL")


@dataclass(frozen=True, slots=True)
class RemoteDestination:
    url: str
    scheme: str
    host: str
    port: int
    target: str
    addresses: tuple[str, ...]

    @property
    def authority(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        default_port = 443 if self.scheme == "https" else 80
        return host if self.port == default_port else f"{host}:{self.port}"


def resolve_remote_url(image_url: str, allowed_hosts: list[str]) -> RemoteDestination:
    check_url_text(image_url)
    try:
        original = urlsplit(image_url)
        if original.port == 0:
            raise ValueError("Port zero is not supported")
    except ValueError as exc:
        raise ValueError("Invalid image URL") from exc
    if original.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) image URLs are supported")
    if not original.hostname:
        raise ValueError("Invalid image URL")
    if original.username is not None or original.password is not None:
        raise ValueError("Image URL credentials are not supported")
    try:
        prepared = requests.Request("GET", image_url).prepare().url
        if prepared is None:
            raise ValueError("Missing prepared URL")
        parts = urlsplit(prepared)
        host = (parts.hostname or "").lower().rstrip(".")
        port = parts.port
    except (ValueError, requests.RequestException, UnicodeError) as exc:
        raise ValueError("Invalid image URL") from exc
    if not host or "%" in host or port == 0:
        raise ValueError("Invalid image URL")
    port = port if port is not None else (443 if parts.scheme == "https" else 80)
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
        if len(host) > 253 or any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in host.split(".")
        ):
            raise ValueError("Invalid image URL")
    else:
        host = str(literal)
    if allowed_hosts:
        allowed = {
            value.strip().lower().rstrip(".")
            for value in allowed_hosts
            if value.strip()
        }
        if host not in allowed:
            raise ValueError("Remote image host is not allowlisted")
    if host == "localhost":
        raise ValueError("Remote image host resolves to a private address")

    if literal is not None:
        addresses = (host,)
    else:
        try:
            infos = socket.getaddrinfo(
                host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
            )
        except socket.gaierror as exc:
            raise ValueError("Unable to resolve remote image host") from exc
        addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise ValueError("Unable to resolve remote image host")
    try:
        safe = all(is_public_address(address) for address in addresses)
    except ValueError as exc:
        raise ValueError("Unable to resolve remote image host") from exc
    if not safe:
        raise ValueError("Remote image host resolves to a private address")
    addresses = tuple(str(ipaddress.ip_address(address)) for address in addresses)
    target = parts.path or "/"
    if parts.query:
        target += f"?{parts.query}"
    authority = f"[{host}]" if ":" in host else host
    if port != (443 if parts.scheme == "https" else 80):
        authority += f":{port}"
    canonical = urlunsplit(
        (parts.scheme, authority, parts.path or "/", parts.query, "")
    )
    return RemoteDestination(canonical, parts.scheme, host, port, target, addresses)
