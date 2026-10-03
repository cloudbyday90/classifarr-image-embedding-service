# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tag-only Docker Hub cleanup; no secrets in CLI arguments, logs or outputs."""

import os

from dockerhub_client import DockerHubClient
from dockerhub_retention import DELETE_PATH, deletion_plan, inventory


def main() -> None:
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_REF_TYPE") != "tag"
        or not tag.startswith("v")
    ):
        raise ValueError("Docker Hub cleanup requires a version tag push")
    client = DockerHubClient()
    client.authenticate(
        os.environ.get("DOCKERHUB_USERNAME", ""), os.environ.get("DOCKERHUB_TOKEN", "")
    )
    plan = deletion_plan(inventory(client), tag)
    print(f"Validated complete Docker Hub inventory; deleting {len(plan)} old tags")
    for name in plan:
        client.request("DELETE", f"{DELETE_PATH}/{name}/")
        print(f"Deleted tag: {name}")


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        raise SystemExit(
            "Docker Hub cleanup failed; no further tags will be deleted"
        ) from None
