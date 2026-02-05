# Agent Instructions

## Project Overview

This repository contains the standalone image embedding service for Classifarr. It exposes a small HTTP API to generate image embeddings and to list supported image models.

## Repo Layout

- `src/image_embedder` - FastAPI service code
- `tests` - Pytest suite
- `Dockerfile` / `docker-compose.yml` - Container build and local run
- `.github/workflows` - CI/CD

## Working Rules

- Keep API responses stable: `/health`, `/models`, `/embed-image`
- If you change request/response schemas, update tests and README examples
- Keep changes scoped and document behavior changes in `README.md`

## Execution Pattern

Use deterministic code for data processing. Prefer adding tests for behavior changes.

## Summary

Be pragmatic. Be reliable. Keep the service minimal and predictable.
