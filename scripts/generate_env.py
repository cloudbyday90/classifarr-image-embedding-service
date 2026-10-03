#!/usr/bin/env python3
# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create private .env data and missing defaults; explicit display and rotation."""

import argparse
import sys
from pathlib import Path

from secret_publication import SetupError
from secret_setup import generate_setup

ROOT = Path(__file__).absolute().parent.parent


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(
        description="Set up a private .env and missing config.toml defaults."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rotate .env; preserve existing config.toml.",
    )
    parser.add_argument(
        "--show-key",
        action="store_true",
        help="Explicitly display the newly generated key; may expose it to console capture.",
    )
    args = parser.parse_args(argv)
    try:
        result = generate_setup(root, force=args.force)
    except SetupError as error:
        print(f"Setup refused: {error}", file=sys.stderr)
        return 1
    except FileExistsError:
        print(
            "Setup refused: another writer created a setup file. Existing data was preserved.",
            file=sys.stderr,
        )
        return 1
    except OSError:
        print(
            "Setup failed: check directory ownership, filesystem ACL/hard-link support and write permissions. Existing secrets were not intentionally modified.",
            file=sys.stderr,
        )
        return 1

    print(f"Generated private {root / '.env'}")
    if result.config_created:
        print(f"Generated {root / 'config.toml'} (default settings; edit to customise)")
    else:
        print("Existing config.toml preserved.")
    if args.show_key:
        print(f"SERVICE_API_KEY={result.api_key}")
    else:
        print(
            "Key saved privately; open .env in a private editor to copy SERVICE_API_KEY."
        )
    print(
        "In Classifarr: Settings -> API Keys, use the embed_service tier, or set IMAGE_EMBEDDER_API_KEY."
    )
    print(
        "After rotation, update Classifarr and recreate the service container to load the new key."
    )
    print("Run: docker compose up -d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
