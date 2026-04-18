#!/usr/bin/env pwsh
# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

$Root = Split-Path -Parent $PSScriptRoot
$VenvScripts = Join-Path $Root ".venv\Scripts"
$PytestExe = Join-Path $VenvScripts "pytest.exe"

if (-not (Test-Path $PytestExe)) {
    Write-Error "pytest is not installed in $VenvScripts. Run '.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt' first."
    exit 1
}

if (-not ($env:PATH -split ';' | Where-Object { $_ -eq $VenvScripts })) {
    $env:PATH = "$VenvScripts;$env:PATH"
}

Set-Location $Root
Write-Host "Dev shell ready. 'pytest' now resolves to $PytestExe" -ForegroundColor Green
