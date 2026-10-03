# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate explicit TCP proxy peers without inheriting ambient server trust."""

from ipaddress import ip_network


def validated_proxy_peers(value: object) -> list[str]:
    """An empty array disables forwarding; trust-all and implicit names fail closed."""
    message = (
        "forwarded_allow_ips must be an array of explicit IP addresses or networks"
    )
    if not isinstance(value, list):
        raise ValueError(message)
    peers = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip() or "%" in entry:
            raise ValueError(message)
        text = entry.strip()
        try:
            network = ip_network(text, strict=True)
        except ValueError as exc:
            raise ValueError(message) from exc
        if network.prefixlen == 0:
            raise ValueError("forwarded_allow_ips must not trust every address")
        peers.append(str(network) if "/" in text else str(network.network_address))
    return peers
