# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small Docker Hub client: in-memory credentials, no redirects, bounded JSON."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ORIGIN = "https://hub.docker.com"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Docker Hub redirects are refused")


class DockerHubClient:
    def __init__(self):
        self._opener = build_opener(ProxyHandler({}), NoRedirects())
        self._token = None

    def request(self, method: str, path: str, body: dict | None = None):
        if not path.startswith("/v2/") or any(ord(c) < 32 or c in "\\#" for c in path):
            raise ValueError("Invalid Docker Hub request path")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = json.dumps(body).encode() if body is not None else None
        request = Request(ORIGIN + path, method=method, headers=headers, data=data)
        try:
            with self._opener.open(request, timeout=30) as response:
                if response.status != (204 if method == "DELETE" else 200):
                    raise ValueError("Unexpected Docker Hub HTTP status")
                if method == "DELETE":
                    return None
                content = response.read(1024 * 1024 + 1)
                if len(content) > 1024 * 1024:
                    raise ValueError("Docker Hub response exceeds one MiB")
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("Docker Hub returned a non-object response")
                return result
        except (HTTPError, URLError, OSError, ValueError):
            # Never echo server bodies, credentials, tokens or exception request reprs.
            raise ValueError("Docker Hub request failed; cleanup stopped") from None

    def authenticate(self, identifier: str, secret: str) -> None:
        if not identifier or not secret:
            raise ValueError("Docker Hub cleanup credentials are missing")
        result = self.request(
            "POST", "/v2/auth/token", {"identifier": identifier, "secret": secret}
        )
        token = result.get("access_token")
        if (
            not isinstance(token, str)
            or not 0 < len(token) <= 8192
            or not token.isascii()
            or any(c.isspace() or ord(c) < 32 for c in token)
        ):
            raise ValueError("Docker Hub returned an invalid access token")
        self._token = token
