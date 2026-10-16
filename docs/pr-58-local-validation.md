# PR 58 local Setuptools floor adoption

Assessment: 2026-10-16. Baseline: `c38092c926c894c3b7da819b7f83dbf75000df9b`.

## Selection and research

GitHub MCP returned seven open proposals after the earlier local Transformers
adoption. An unbiased `secrets.choice([55,56,58,59,60,61,62])` draw selected
[PR #58](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/58),
head `76710614970d513c84ba191057ec441d2eb231ee`. Its fetched patch changes only
`setuptools>=83.0.0` to `>=84.0.0` in shared requirements.

Official publisher URLs were discovered and fetched through web MCP.
[Setuptools history](https://setuptools.pypa.io/en/latest/history.html?highlight=%22--build-option%22)
and [the 84.0.0 publisher listing](https://pypi.org/project/setuptools/84.0.0/)
identify the August 8 release and wheel. All six affected Linux graphs already
contain that exact wheel. Its publisher SHA-256 is
`51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670`.
Do not describe a raised floor as a newly installed security patch.

## Decision and tradeoffs

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Raise the floor and renew affected input attestations | Reproduces PR intent; aligns declarations with reviewed graphs; no wheel churn | Requires proving complete native inventories remain valid | Adopt |
| Resolve unrelated packages again | Broad refresh | Adds unrelated compatibility and artifact changes | Defer |
| Change the floor without renewing attestations | Small edit | Breaks the explicit dependency-input contract | Reject |

The six affected manifests renew only input hashes. Their full artifact records
and all 13 wheel-lock text files remain unchanged. The seven unaffected profiles
remain unchanged. The original PR is not merged.

## Outcome

Exact native inventories and strict `pip check` pass for CPU amd64/arm64
(56 packages each), QA (70), OpenVINO (58), modern CUDA (75) and legacy CUDA (75).
ARM64 runs target CPython under QEMU, not physical ARM performance testing.
Fresh strict OSV reports pass the five actual amd64 installed graphs and the
isolated 29-package audit-tool graph with complete inventories, zero skipped
packages and zero known vulnerabilities. CPU ARM64's verified name/version
inventory matches audited CPU amd64; no native ARM audit-tool invocation is
claimed. All 13 input/lock profiles validate.

Complete locked QA passes **1,162 tests**, seven existing skips and unchanged
coverage floors. Current/baseline source produce byte-identical OpenAPI JSON.
The [cooperative inference outcome](cooperative-inference-cancellation.md)
separately records real CPU/OpenVINO/RTX model parity, chunking, cleanup and
performance tradeoffs; the floor-only adoption introduces no new native wheel.
Native CodeQL retains only the two existing setup alerts, and scoped typing,
lint, skill, copyright and secret checks pass.

GitHub MCP reconfirmed the original PR open/unmerged at the same reviewed head
before delivery. Keep the aligned 84.0.0 floor and continue separate reviewed
dependency refreshes and strict audits. The
[validation archive](validation/cooperative-inference-2026-10-16.json) preserves
the random selection, all 13 unchanged artifact/text proofs, six native images
and exact reports. Delivery stays directly on master with Unreleased changes;
no new branch, original PR merge, release, tag or version bump is created.
