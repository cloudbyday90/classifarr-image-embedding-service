# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Explicit forwarding configuration cannot inherit ambient or wildcard trust."""

import pytest

from image_embedder import config, server
from image_embedder.config import Settings

ENV = "IMAGE_EMBEDDER_FORWARDED_ALLOW_IPS"


@pytest.fixture(autouse=True)
def isolated_forwarding(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    monkeypatch.setattr(config, "_TOML", {})


@pytest.mark.parametrize("peers", [[], ["127.0.0.1"], ["192.0.2.0/24", "::1"]])
def test_launcher_owns_forwarding_policy(monkeypatch, peers):
    settings = Settings(server_forwarded_allow_ips=peers)
    monkeypatch.setattr(server, "Settings", lambda: settings)
    calls = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    server.main()
    assert len(calls) == 1
    args, options = calls[0]
    assert args == ("image_embedder.main:app",)
    assert options["proxy_headers"] is bool(peers)
    assert options["forwarded_allow_ips"] == peers
    assert options["workers"] == settings.server_workers
    assert options["limit_concurrency"] == settings.server_concurrency


def test_default_and_environment_precedence(monkeypatch):
    assert Settings().server_forwarded_allow_ips == []
    monkeypatch.setattr(config, "_TOML", {"server": {"forwarded_allow_ips": ["::1"]}})
    assert Settings().server_forwarded_allow_ips == ["::1"]
    monkeypatch.setenv(ENV, " 192.0.2.0/24,2001:0db8::/32 ")
    assert Settings().server_forwarded_allow_ips == ["192.0.2.0/24", "2001:db8::/32"]
    monkeypatch.setenv(ENV, "")
    assert Settings().server_forwarded_allow_ips == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        "127.0.0.1",
        True,
        [True],
        [""],
        ["*"],
        ["0.0.0.0/0"],
        ["::/0"],
        ["localhost"],
        ["/tmp/proxy.sock"],
        ["192.0.2.1/24"],
        ["fe80::1%eth0"],
        ["192.0.2.1\nspoof"],
        ["192.0.2.1,192.0.2.2"],
    ],
)
def test_constructor_and_toml_reject_ambiguous_trust(monkeypatch, value):
    with pytest.raises(ValueError, match="forwarded_allow_ips"):
        Settings(server_forwarded_allow_ips=value)
    monkeypatch.setattr(config, "_TOML", {"server": {"forwarded_allow_ips": value}})
    # TOML cannot represent None; omitted values retain the safe empty default.
    if value is not None:
        with pytest.raises(ValueError, match="forwarded_allow_ips"):
            Settings()


@pytest.mark.parametrize(
    "value", ["*", "127.0.0.1,", ",127.0.0.1", "localhost", "::/0"]
)
def test_environment_rejects_ambiguous_trust(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match="forwarded_allow_ips"):
        Settings()
