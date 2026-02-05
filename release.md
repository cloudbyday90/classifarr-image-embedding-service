# Release Checklist (Template)

## Release Metadata

- Version: v0.0.0
- Date: YYYY-MM-DD
- Owner: 
- Scope/Highlights: 

## Preconditions

- Implementation plan complete and signed off.
- Local tests pass.
- No failing CI or pending review tasks.
- Docker build validates successfully.

## Local Testing (Required Before Release)

1. Tests (local Python):
   - `pytest`
2. Tests (Docker fallback):
   - `docker run --rm -v "$PWD:/app" -w /app python:3.12-slim bash -lc "python -m pip install -r requirements.txt -r requirements-dev.txt && pytest -q"`
3. Docker build:
   - `docker build -t classifarr-image-embedder:local .`

## Prerequisites

- [GitHub CLI (gh)](https://cli.github.com/) installed and authenticated (`gh auth login`).
- Docker build pipeline configured to trigger on Release creation.
- CI green (tests + docker build).

## Release Steps

1. **Update Version**: Update version references:
   - `src/image_embedder/main.py` (FastAPI `version=...`)
   - `src/image_embedder/__init__.py` (optional `__version__`)
   - `README.md` (if version displayed)
2. **Release Notes (High-Level)**: Update `RELEASE_NOTES.md` with user-facing highlights (emojis/graphs ok).
3. **Changelog (Technical)**: Update `CHANGELOG.md` with technical changes, breaking changes, and config tweaks.
4. **Commit**: Commit changes with message `release: vX.Y.Z`.
5. **Create GitHub Release & Tag**: Use GitHub CLI to create the **GitHub Release** and tag simultaneously. This should trigger the Docker build pipeline.

```bash
# Syntax: gh release create <tag> --title "<title>" --notes-file <file> --target <branch>

# Example:
gh release create v0.1.0 --title "v0.1.0" --notes-file RELEASE_NOTES.md --target master
```

> **Note**: Ensure `RELEASE_NOTES.md` contains ONLY the notes for the current release if using it as the source file, or copy the specific section to a temp file. Alternatively, use `--generate-notes` for auto-generated notes.

## Post-Release Verification

- Confirm container build completed and pushed.
- Pull and smoke test: `docker run --rm -p 8000:8000 ghcr.io/cloudbyday90/classifarr-image-embedder:<tag>`
- Validate `/health` and `/models` endpoints.
- Confirm release tag matches the published container tag (e.g., `v0.1.0` -> `:0.1.0`).
- Monitor logs and issue tracker for regressions.
