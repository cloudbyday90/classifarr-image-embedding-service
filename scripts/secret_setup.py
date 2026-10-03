# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set up separate application defaults and a private shared service credential."""

import secrets
from dataclasses import dataclass, field
from pathlib import Path

from secret_publication import (
    SetupError,
    publish_defaults,
    publish_private,
    setup_lock,
    validate_root,
    validate_target,
)
from setup_templates import CONFIG_TEMPLATE, ENV_TEMPLATE


@dataclass(frozen=True)
class SetupResult:
    api_key: str = field(repr=False)
    config_created: bool


def generate_setup(root: Path, *, force: bool = False) -> SetupResult:
    validate_root(root)
    with setup_lock(root):
        env_file, config_file = root / ".env", root / "config.toml"
        if validate_target(env_file) and not force:
            raise SetupError(
                ".env already exists. Use --force only to rotate the service key."
            )
        config_created = not validate_target(config_file)
        if config_created:
            try:
                publish_defaults(config_file, CONFIG_TEMPLATE.encode("utf-8"))
            except FileExistsError:
                # A valid config published by another writer takes precedence.
                validate_target(config_file)
                config_created = False
        api_key = secrets.token_urlsafe(32)
        publish_private(
            env_file,
            ENV_TEMPLATE.format(api_key=api_key).encode("utf-8"),
            replace=force,
        )
        return SetupResult(api_key, config_created)
