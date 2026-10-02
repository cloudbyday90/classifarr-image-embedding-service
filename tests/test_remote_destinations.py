# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Destination policy, redirects, DNS changes and streaming resource ownership."""

import socket

import pytest
from remote_fakes import RemoteResponse, RemoteTransport, dns_answers
from urllib3.exceptions import NewConnectionError, ReadTimeoutError

from image_embedder import remote_fetch
from image_embedder.config import Settings
from image_embedder.input_limits import InputLimitExceeded
from image_embedder.remote_fetch import RemoteFetchError, fetch_remote_image
from image_embedder.remote_url import is_public_address, resolve_remote_url


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "169.254.169.254",
        "100.64.0.1",
        "192.0.0.8",
        "192.0.2.1",
        "0.0.0.0",
        "224.0.0.1",
        "::",
        "::1",
        "fd00::1",
        "fe80::1",
        "ff02::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "64:ff9b:1::1",
    ],
)
def test_nonpublic_and_embedded_private_addresses_are_refused(address):
    assert not is_public_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_addresses_remain_accepted(address):
    assert is_public_address(address)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1\\@1.2.3.4/x",
        "http://user:secret@images.example/x",
        "http://@images.example/x",
        "http://images.example:0/x",
        "http://images.example:65536/x",
        "http://images.example:bad/x",
        "http://[::1/x",
        "http://[2606:4700::1%eth0]/x",
        "http://%31%32%37.0.0.1/x",
        "http://images.example/\r\nx",
        " http://images.example/x",
        "http://images.example/\tx",
        "http:///x",
        "http://bad_host.example/x",
        "http://-bad.example/x",
        "http://images.example/\x7fx",
    ],
)
def test_ambiguous_authorities_controls_and_credentials_never_reach_dns(
    monkeypatch, url
):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("unsafe URL reached DNS")
    )
    with pytest.raises(ValueError):
        resolve_remote_url(url, [])


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://images.example/x", "//images.example/x"]
)
def test_only_http_schemes_are_accepted(url):
    with pytest.raises(ValueError, match="Only http"):
        resolve_remote_url(url, [])


@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001", "0177.0.0.1"])
def test_alternative_ipv4_literals_cannot_bypass_public_policy(host):
    # The system resolver interprets these as numeric addresses, without external DNS.
    with pytest.raises(ValueError, match="private address"):
        resolve_remote_url(f"http://{host}/x", [])


def test_canonical_host_idna_encoded_path_custom_port_and_fragment(monkeypatch):
    calls = dns_answers(monkeypatch, "8.8.8.8")
    destination = resolve_remote_url(
        "HTTPS://BÜCHER.example.:8443/a b?token=a%2Fb#ignored",
        ["XN--BCHER-KVA.EXAMPLE."],
    )
    assert destination.host == "xn--bcher-kva.example"
    assert destination.authority == "xn--bcher-kva.example:8443"
    assert destination.target == "/a%20b?token=a%2Fb"
    assert destination.url == "https://xn--bcher-kva.example:8443/a%20b?token=a%2Fb"
    assert calls == [(destination.host, 8443)]


@pytest.mark.parametrize("scheme,port", [("http", 80), ("https", 443)])
def test_scheme_default_port_and_deduplicated_addresses(monkeypatch, scheme, port):
    calls = dns_answers(monkeypatch, "8.8.8.8", "8.8.8.8")
    destination = resolve_remote_url(f"{scheme}://images.example/x", [])
    assert calls == [("images.example", port)]
    assert destination.addresses == ("8.8.8.8",)


@pytest.mark.parametrize("host", ["8.8.8.8", "[2606:4700:4700::1111]"])
def test_public_literal_does_not_use_dns(monkeypatch, host):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("literal resolved")
    )
    assert resolve_remote_url(f"http://{host}/x", []).addresses


@pytest.mark.parametrize("addresses", [(), ("8.8.8.8", "::1"), ("8.8.8.8", "10.0.0.1")])
def test_empty_or_mixed_dns_results_fail_before_transport(monkeypatch, addresses):
    dns_answers(monkeypatch, *addresses)
    transport = RemoteTransport(monkeypatch)
    with pytest.raises(ValueError):
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
    assert not transport.calls


def test_disabled_fetch_never_parses_resolves_or_connects(monkeypatch):
    monkeypatch.setattr(
        remote_fetch,
        "resolve_remote_url",
        lambda *_a: pytest.fail("disabled fetch resolved"),
    )
    with pytest.raises(ValueError, match="disabled"):
        fetch_remote_image("http://images.example/x", Settings())


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/private",
        "//169.254.169.254/latest",
        "http://[::1]/x",
        "http://100.64.0.1/x",
        "http://127.0.0.1\\@8.8.8.8/x",
        "http://user:secret@8.8.8.8/x",
        "file:///etc/passwd",
    ],
)
def test_redirect_security_triggers_do_not_open_second_pool(monkeypatch, location):
    dns_answers(monkeypatch, "8.8.8.8")
    redirect = RemoteResponse(302, {"location": location}, chunks=(b"x" * 10000,))
    transport = RemoteTransport(monkeypatch, redirect)
    with pytest.raises(ValueError):
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
    assert len(transport.calls) == 1
    assert redirect.closed and not redirect.streamed
    assert transport.closed == ["8.8.8.8"]


def test_each_redirect_host_must_be_allowlisted(monkeypatch):
    dns_answers(monkeypatch, "8.8.8.8")
    redirect = RemoteResponse(302, {"location": "http://unlisted.example/x"})
    transport = RemoteTransport(monkeypatch, redirect)
    with pytest.raises(ValueError, match="allowlisted"):
        fetch_remote_image(
            "http://images.example/x",
            Settings(allow_remote_urls=True, allowed_remote_hosts=["images.example"]),
        )
    assert len(transport.calls) == 1 and redirect.closed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_relative_and_cross_host_public_redirects_keep_host_and_bounds(
    monkeypatch, status
):
    calls = dns_answers(monkeypatch, "8.8.8.8")
    responses = [
        RemoteResponse(status, {"location": "/next?x=1"}),
        RemoteResponse(status, {"location": "//cdn.example/final"}),
        RemoteResponse(),
    ]
    transport = RemoteTransport(monkeypatch, *responses)
    settings = Settings(
        allow_remote_urls=True,
        max_image_bytes=3,
        allowed_remote_hosts=["images.example", "cdn.example"],
    )
    assert fetch_remote_image("https://images.example/start", settings) == b"abc"
    assert calls == [
        ("images.example", 443),
        ("images.example", 443),
        ("cdn.example", 443),
    ]
    assert [call[4] for call in transport.calls] == ["/start", "/next?x=1", "/final"]
    for call in transport.calls:
        assert call[1] == "8.8.8.8"
        assert call[5] == {
            "headers": {"Host": call[0].host, "Accept-Encoding": "identity"},
            "redirect": False,
            "retries": False,
            "preload_content": False,
            "assert_same_host": False,
        }
    assert all(response.closed for response in responses)
    assert not responses[0].streamed and not responses[1].streamed


def test_same_host_redirect_rechecks_changed_dns(monkeypatch):
    calls = []

    def resolve(host, port, **_kwargs):
        calls.append(host)
        address = "8.8.8.8" if len(calls) == 1 else "127.0.0.1"
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    redirect = RemoteResponse(302, {"location": "/next"})
    transport = RemoteTransport(monkeypatch, redirect)
    with pytest.raises(ValueError, match="private address"):
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
    assert calls == ["images.example", "images.example"]
    assert len(transport.calls) == 1 and redirect.closed


@pytest.mark.parametrize(
    "location,error",
    [
        (None, "Invalid remote"),
        ("\t/next", "Invalid image"),
        ("http://images.example/x", "downgrade"),
    ],
)
def test_invalid_and_downgrade_redirects_close_without_drain(
    monkeypatch, location, error
):
    dns_answers(monkeypatch, "8.8.8.8")
    response = RemoteResponse(302, {"location": location} if location else {})
    transport = RemoteTransport(monkeypatch, response)
    with pytest.raises(ValueError, match=error):
        fetch_remote_image("https://images.example/x", Settings(allow_remote_urls=True))
    assert response.closed and not response.streamed and len(transport.closed) == 1


def test_redirect_loop_is_bounded(monkeypatch):
    dns_answers(monkeypatch, "8.8.8.8")
    responses = [RemoteResponse(302, {"location": "/same"}) for _ in range(4)]
    transport = RemoteTransport(monkeypatch, *responses)
    with pytest.raises(ValueError, match="redirect limit"):
        fetch_remote_image(
            "http://images.example/same", Settings(allow_remote_urls=True)
        )
    assert len(transport.calls) == 4 and all(response.closed for response in responses)


@pytest.mark.parametrize(
    "headers,chunks,error",
    [
        ({"content-length": "4"}, (), InputLimitExceeded),
        ({}, (b"ab", b"cd"), InputLimitExceeded),
        ({"content-length": "-1"}, (), ValueError),
        ({"content-length": "invalid"}, (), ValueError),
    ],
)
def test_streaming_and_declared_refusals_close_response_and_pool(
    monkeypatch, headers, chunks, error
):
    dns_answers(monkeypatch, "8.8.8.8")
    response = RemoteResponse(headers=headers, chunks=chunks)
    transport = RemoteTransport(monkeypatch, response)
    with pytest.raises(error):
        fetch_remote_image(
            "http://images.example/x",
            Settings(allow_remote_urls=True, max_image_bytes=3),
        )
    assert response.closed and transport.closed == ["8.8.8.8"]


@pytest.mark.parametrize("status", [304, 404, 500])
def test_http_failures_are_sanitized_and_closed(monkeypatch, status):
    dns_answers(monkeypatch, "8.8.8.8")
    response = RemoteResponse(status)
    transport = RemoteTransport(monkeypatch, response)
    with pytest.raises(RemoteFetchError, match=f"HTTP {status}") as error:
        fetch_remote_image(
            "http://images.example/x?secret=token", Settings(allow_remote_urls=True)
        )
    assert "secret" not in str(error.value) and "images.example" not in str(error.value)
    assert response.closed and not response.streamed and transport.closed


def test_connect_fallback_uses_only_the_validated_set(monkeypatch):
    dns_answers(monkeypatch, "2606:4700:4700::1111", "8.8.8.8")
    response = RemoteResponse()
    transport = RemoteTransport(
        monkeypatch, NewConnectionError(None, "unreachable"), response
    )
    assert (
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
        == b"abc"
    )
    assert [call[1] for call in transport.calls] == ["2606:4700:4700::1111", "8.8.8.8"]
    assert len(transport.closed) == 2 and response.closed


@pytest.mark.parametrize("ipv6_first", [True, False])
def test_connect_fallback_reaches_both_families_within_budget(monkeypatch, ipv6_first):
    ipv6 = [f"2606:4700:4700::{i}" for i in range(1, 5)]
    ipv4 = [f"8.8.8.{i}" for i in range(1, 5)]
    primary, secondary = (ipv6, ipv4) if ipv6_first else (ipv4, ipv6)
    dns_answers(monkeypatch, *primary, secondary[0])
    response = RemoteResponse()
    transport = RemoteTransport(
        monkeypatch, NewConnectionError(None, "unreachable"), response
    )
    assert (
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
        == b"abc"
    )
    assert [call[1] for call in transport.calls] == [primary[0], secondary[0]]
    assert len(transport.closed) == 2 and response.closed


def test_connect_fallback_has_a_finite_address_budget(monkeypatch):
    dns_answers(monkeypatch, *[f"8.8.8.{i}" for i in range(1, 7)])
    transport = RemoteTransport(
        monkeypatch, *[NewConnectionError(None, "unreachable")] * 4
    )
    with pytest.raises(RemoteFetchError, match="Unable to fetch"):
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
    assert len(transport.calls) == len(transport.closed) == 4


def test_stream_failure_is_sanitized_and_never_retries_another_address(monkeypatch):
    dns_answers(monkeypatch, "8.8.8.8", "1.1.1.1")
    response = RemoteResponse(error=ReadTimeoutError(None, "secret-url", "timeout"))
    transport = RemoteTransport(monkeypatch, response)
    with pytest.raises(RemoteFetchError, match="Unable to fetch") as error:
        fetch_remote_image("http://images.example/x", Settings(allow_remote_urls=True))
    assert "secret" not in str(error.value)
    assert response.closed and len(transport.calls) == len(transport.closed) == 1
