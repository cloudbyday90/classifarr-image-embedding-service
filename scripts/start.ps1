#!/usr/bin/env pwsh
# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# First-time setup guard: generates .env if missing, then starts the stack.
# On subsequent runs (when .env already exists) it goes straight to docker compose up -d.

$Root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path "$Root\.env")) {
    Write-Host ""
    Write-Host "  No .env found — running first-time setup..." -ForegroundColor Yellow
    Write-Host ""
    python "$Root\scripts\generate_env.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host ""
    Write-Host "  *** Copy the SERVICE_API_KEY above into Classifarr before continuing. ***" -ForegroundColor Cyan
    Write-Host "  In Classifarr: Settings -> API Keys, create a key with the 'embed_service' tier." -ForegroundColor Cyan
    Write-Host "  Or set IMAGE_EMBEDDER_API_KEY in Classifarr's environment." -ForegroundColor Cyan
    Write-Host ""
    Read-Host "  Press Enter when ready to start the stack"
} else {
    Write-Host "  .env found — skipping key generation." -ForegroundColor Green
}

Set-Location $Root
docker compose up -d
