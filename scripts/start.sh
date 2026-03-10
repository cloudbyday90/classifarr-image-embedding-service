#!/usr/bin/env bash
# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# First-time setup guard: generates .env if missing, then starts the stack.
# On subsequent runs (when .env already exists) it goes straight to docker compose up -d.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$ROOT/.env" ]; then
    echo ""
    echo "  No .env found — running first-time setup..."
    echo ""
    python3 "$ROOT/scripts/generate_env.py"
    echo ""
    echo "  *** Copy the SERVICE_API_KEY above into Classifarr before continuing. ***"
    echo "  In Classifarr: Settings -> API Keys, create a key with the 'embed_service' tier."
    echo "  Or set IMAGE_EMBEDDER_API_KEY in Classifarr's environment."
    echo ""
    read -rp "  Press Enter when ready to start the stack..."
else
    echo "  .env found — skipping key generation."
fi

cd "$ROOT"
docker compose up -d
