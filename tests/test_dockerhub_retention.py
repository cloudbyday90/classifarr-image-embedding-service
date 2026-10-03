# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import json
from unittest.mock import MagicMock
from urllib.error import HTTPError

import cleanup_dockerhub
import pytest
from dockerhub_client import DockerHubClient, NoRedirects
from dockerhub_retention import LIST_PATH, deletion_plan, inventory, page_path


def tags():
    return [
        {"name": f"v1.0.{index}", "last_updated": f"2026-09-{index + 1:02d}T00:00:00Z"}
        for index in range(8)
    ]


def test_retention_keeps_five_newest_latest_and_current_even_with_clock_skew():
    data = tags() + [{"name": "latest", "last_updated": "2026-09-01T00:00:00Z"}]
    assert deletion_plan(data, "v1.0.0") == ["v1.0.1", "v1.0.2"]
    assert deletion_plan(data, "v1.0.7") == ["v1.0.0", "v1.0.1", "v1.0.2"]


@pytest.mark.parametrize(
    "change", ["duplicate", "name", "timestamp", "timezone", "current", "record"]
)
def test_rejects_unsafe_retention_inputs(change):
    data = tags()
    current = "v1.0.7"
    if change == "duplicate":
        data.append(data[0])
    elif change == "name":
        data[0]["name"] = "../outside"
    elif change == "timestamp":
        data[0]["last_updated"] = None
    elif change == "timezone":
        data[0]["last_updated"] = "2026-09-01T00:00:00"
    elif change == "current":
        current = "v9.9.9"
    else:
        data[0] = None
    with pytest.raises(ValueError):
        deletion_plan(data, current)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.test" + LIST_PATH + "?page=2",
        "http://hub.docker.com" + LIST_PATH,
        "https://user:secret@hub.docker.com" + LIST_PATH,
        "https://hub.docker.com:443" + LIST_PATH,
        "https://hub.docker.com/v2/namespaces/other/repositories/other/tags",
        "https://hub.docker.com" + LIST_PATH + "?page=2&page=3",
        "https://hub.docker.com" + LIST_PATH + "?token=secret",
        "https://hub.docker.com" + LIST_PATH + "?page=0",
        "https://hub.docker.com" + LIST_PATH + "?page=101",
        "https://hub.docker.com" + LIST_PATH + "#fragment",
    ],
)
def test_pagination_never_moves_credentials_to_other_destinations(url):
    with pytest.raises(ValueError):
        page_path(url)


def test_collects_complete_paginated_inventory_before_retention():
    client = MagicMock()
    client.request.side_effect = [
        {
            "count": 8,
            "results": tags()[:4],
            "next": "https://hub.docker.com" + LIST_PATH + "?page=2&page_size=100",
        },
        {"count": 8, "results": tags()[4:], "next": None},
    ]
    assert inventory(client) == tags()
    assert [call.args[0] for call in client.request.call_args_list] == ["GET", "GET"]


@pytest.mark.parametrize(
    "change",
    ["count", "incomplete", "loop", "next", "empty_next", "records", "changed_count"],
)
def test_inventory_failure_prevents_any_deletion(change):
    client = MagicMock()
    first = {"count": 8, "results": tags(), "next": None}
    if change == "count":
        first["count"] = True
    elif change == "incomplete":
        first["results"] = tags()[:4]
    elif change == "loop":
        first["next"] = "https://hub.docker.com" + LIST_PATH + "?page_size=100"
    elif change == "next":
        first["next"] = "https://evil.test/inventory"
    elif change == "empty_next":
        first["next"] = ""
    elif change == "records":
        first["results"] = {}
    else:
        first["results"] = tags()[:4]
        first["next"] = "https://hub.docker.com" + LIST_PATH + "?page=2"
    client.request.side_effect = [
        first,
        {"count": 9, "results": tags()[4:], "next": None},
    ]
    with pytest.raises(ValueError):
        inventory(client)
    assert all(call.args[0] == "GET" for call in client.request.call_args_list)


def test_authentication_serializes_secret_as_data_and_never_exports_token():
    client = DockerHubClient()
    reply = MagicMock()
    reply.status = 200
    reply.read.return_value = b'{"access_token":"fixture-bearer"}'
    reply.__enter__.return_value = reply
    client._opener = MagicMock()
    client._opener.open.return_value = reply
    secret = "quote'$(command)\nfixture"
    client.authenticate("fixture", secret)
    request = client._opener.open.call_args.args[0]
    assert request.full_url == "https://hub.docker.com/v2/auth/token"
    assert json.loads(request.data) == {"identifier": "fixture", "secret": secret}
    assert request.get_method() == "POST"
    assert "Authorization" not in request.headers


@pytest.mark.parametrize(
    "token", [None, "", "token\ninjection", "nonascii-é", "x" * 8193]
)
def test_refuses_invalid_bearer_tokens(token, monkeypatch):
    client = DockerHubClient()
    monkeypatch.setattr(client, "request", lambda *args: {"access_token": token})
    with pytest.raises(ValueError):
        client.authenticate("fixture", "fixture")


def test_http_error_never_echoes_response_body_or_credential():
    client = DockerHubClient()
    client._opener = MagicMock()
    client._opener.open.side_effect = HTTPError(
        "https://hub.docker.com/v2/auth/token",
        401,
        "fixture-secret",
        {},
        io.BytesIO(b"fixture-secret"),
    )
    with pytest.raises(ValueError) as error:
        client.authenticate("fixture", "fixture-secret")
    assert "fixture-secret" not in str(error.value)


def test_bearer_request_refuses_redirect():
    with pytest.raises(ValueError, match="redirects"):
        NoRedirects().redirect_request(None, None, 302, "", {}, "https://evil.test")


def test_delete_uses_in_memory_bearer_and_requires_confirmed_success():
    client = DockerHubClient()
    client._token = "fixture-bearer"
    reply = MagicMock()
    reply.__enter__.return_value = reply
    reply.status = 204
    client._opener = MagicMock()
    client._opener.open.return_value = reply
    assert (
        client.request("DELETE", "/v2/repositories/fixture/repository/tags/v1/") is None
    )
    request = client._opener.open.call_args.args[0]
    assert request.get_method() == "DELETE"
    assert request.headers["Authorization"] == "Bearer fixture-bearer"
    reply.read.assert_not_called()
    reply.status = 500
    with pytest.raises(ValueError):
        client.request("DELETE", "/v2/repositories/fixture/repository/tags/v1/")


@pytest.mark.parametrize("content", [b"not JSON", b"[]", b"x" * (1024 * 1024 + 1)])
def test_api_rejects_malformed_non_object_or_excess_responses(content):
    client = DockerHubClient()
    reply = MagicMock()
    reply.__enter__.return_value = reply
    reply.status = 200
    reply.read.return_value = content
    client._opener = MagicMock()
    client._opener.open.return_value = reply
    with pytest.raises(ValueError):
        client.request("GET", LIST_PATH)


def release_environment(monkeypatch):
    for key, value in {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_TYPE": "tag",
        "GITHUB_REF_NAME": "v1.0.7",
        "DOCKERHUB_USERNAME": "fixture",
        "DOCKERHUB_TOKEN": "fixture-secret",
    }.items():
        monkeypatch.setenv(key, value)


def test_cleanup_runs_validated_plan_and_stops_after_first_delete_failure(
    monkeypatch, capsys
):
    release_environment(monkeypatch)
    client = MagicMock()
    client.request.side_effect = [
        {"count": 8, "results": tags(), "next": None},
        None,
        ValueError("request stopped"),
    ]
    monkeypatch.setattr(cleanup_dockerhub, "DockerHubClient", lambda: client)
    with pytest.raises(ValueError):
        cleanup_dockerhub.main()
    calls = client.request.call_args_list
    assert [call.args[0] for call in calls] == ["GET", "DELETE", "DELETE"]
    assert calls[1].args[1].endswith("/v1.0.0/")
    assert calls[2].args[1].endswith("/v1.0.1/")
    assert "fixture-secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_REF_TYPE", "branch"),
        ("GITHUB_REF_NAME", "unreviewed"),
    ],
)
def test_cleanup_event_guard_runs_before_authentication(monkeypatch, key, value):
    release_environment(monkeypatch)
    monkeypatch.setenv(key, value)
    client = MagicMock()
    monkeypatch.setattr(cleanup_dockerhub, "DockerHubClient", lambda: client)
    with pytest.raises(ValueError):
        cleanup_dockerhub.main()
    client.authenticate.assert_not_called()
