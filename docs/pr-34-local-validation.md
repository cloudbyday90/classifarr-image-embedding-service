# PR 34 local HTTPX floor adoption

Assessment date: 2026-10-03. Baseline: `d72eec64388bd4b5712a31fb1105378c895cc0da`.

## Selection and design

GitHub MCP enumerated twelve currently open PRs. Exclude already adopted checkout/Gitleaks/OSV/pytest-cov updates, superseded Transformers/Torch floors and unsupported Ubuntu 25.10. The eligible pool was **52, 48, 46, 34, 27**; one uniform random draw selected [PR 34](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/34), head `b4d4ccc31d1a24bdfc3a628ba56b91ba2e2546b2`.

Implement only its `httpx>=0.28.1` development floor, preserving newer unrelated floors. The complete QA profile already locks HTTPX 0.28.1. Renew its input attestation after proving the requirement is the sole dependency-input change, retaining every existing wheel identity and hash. This is local implementation and testing, not an upstream merge or fresh dependency resolution.

## Official research and tradeoffs

The official [HTTPX changelog](https://github.com/encode/httpx/blob/master/CHANGELOG.md) and [0.28.1 release](https://github.com/encode/httpx/releases/tag/0.28.1), discovered through web MCP on October 3, 2026, document the SSL fix for disabled verification combined with client certificates. This floor rejects the older patch while retaining the tested API and complete wheel contract. It does not change service HTTPS verification policy; HTTPX is QA tooling here. Freezing all transitive wheels is stronger than a floor alone but requires deliberate reviewed refreshes. A broader HTTPX migration would require separate compatibility evidence.

Recommendation: adopt this minimum and retain reviewed target-specific hash locks, isolated installation and exact inventories. Keep authored JavaScript ESM; this change and the setup implementation use Python.

## Validation and outcome

The one-line floor is adopted. All nine lock contracts validate; only QA's `requirements-dev.txt` input hash changes. Every QA wheel/version/URL/hash and the complete lock text are byte-for-byte unchanged, and non-QA inputs/locks are preserved. Actual existing QA image `sha256:c79f2bf0bcb878c9e84ab7da5652a3e0aeb84ae44f781b171a45f7ab3dbc9edf` passes pip check and exact verification for **67 packages**, including HTTPX 0.28.1. No fresh resolution or backend build is claimed.

Full locked QA passes **758 tests**, with two optional production and five native Windows skips; coverage remains **94.31% / 88.71%**. The accompanying [secret setup](secret-setup.md) record explains 48 new cross-platform cases, native access checks and contextual CodeQL alerts. The [validation archive](validation/secret-setup-2026-10-03.json) records the random pool, immutable PR identity, source/log hashes and outcomes. The original PR is left unmerged; no release, tag, version bump or separate branch is created.
