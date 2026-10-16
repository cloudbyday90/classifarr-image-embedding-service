# PR 57 local Transformers migration

Assessment: 2026-10-15. Baseline: `e5b55d03ac61ce36c16bf611bc5017c8c01c4955`.

## Selection and design

GitHub MCP returned eight open PRs, #55–#62, through the repository-provided
pull collection. Their actual patches were reviewed. An unbiased local
`secrets.choice` draw over those eight selected
[PR #57](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/57),
head `a0ce0d2644c5d5b3eabd316a364bc9fa0e1a0304`. It raises the Transformers
minimum from 5.17.0 to 5.19.0; existing Linux locks contain 5.18.0, so this is
a real wheel migration. Reproduce the intent locally without merging the PR.

Official publisher URLs were discovered and fetched using web services.
[The 5.19.0 wheel listing](https://pypi.org/project/transformers/5.19.0/)
provides the universal Python wheel, its SHA-256 and publisher provenance.
[The versioned installation guide](https://huggingface.co/docs/transformers/v5.19.0/installation)
describes supported Python/PyTorch versions and pinned offline-cache behavior.
The downloaded wheel is hash-verified before inspecting its metadata or
installing it. Runtime requirements permit the reviewed Hub 1.33.0 and
Tokenizers 0.23.2; optional extras are outside this migration.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Change only the floor | Small diff | Existing 5.18.0 locks would violate the new declaration | Reject |
| Refresh every dependency to current latest | Broad maintenance | Confounds calibration and adds unrelated native migrations | Defer |
| Constrain the complete graph and replace only the reviewed 5.19.0 wheel | Implements selected PR with a reviewable delta | Requires affected native inventory, model and IR validation | Adopt |

PyPI also lists 5.20.0, published October 15. This scoped adoption deliberately
tests 5.19.0 as requested by the selected PR; it does not claim to install the
latest release. Preserve publisher model revisions, restricted loading, explicit
PIL preprocessing, native backend selection and strict advisory checks.

## Validation and outcome

The publisher wheel SHA-256 is
`afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518`.
Six complete Linux graphs migrate exactly one artifact, Transformers
**5.18.0 → 5.19.0**: CPU amd64/arm64 (56 packages each), OpenVINO (58), modern and
legacy CUDA (75 each) and QA (70). Every other wheel identity is unchanged.
Each baseline inventory is verified, native pip evaluates all inputs with
reviewed-version constraints and the verified local wheel, then hash-required
installation, strict `pip check` and exact complete inventory checks pass.
These are constrained installed-graph migrations, not clean-slate index
resolutions. ARM64 executes the target CPython under QEMU; it is not a physical
ARM performance result. All 13 input/lock profiles validate. The seven unaffected
profiles, including Windows contracts, remain unchanged.

Fresh strict OSV audits pass for the actual QA/CPU/OpenVINO/modern CUDA/legacy
CUDA installed graphs with complete 70/56/58/75/75 inventories, no skipped
packages and no known vulnerabilities. The isolated 29-package audit-tool graph
also passes. CPU ARM64 has the same name/version inventory as audited CPU amd64,
with its own native artifact/inventory verification; no native ARM audit-tool
execution or accelerator throughput is claimed.

Both pinned models pass production CPU and OpenVINO single/batch/direct parity
at 1e-4, plus authenticated API contracts. OpenVINO cold export and first/reload
parity pass at 1e-6 under distinct 5.19.0 IR contracts. The old 5.18.0 IR files
remain byte-identical; no publisher cache is modified. A separate fixed-input,
single-thread CPU comparison checks 16 raw/normalized single/batch vectors
across both library versions at 1e-4; all 16 maximum absolute differences are
zero under those fixed inputs and controls. The old graph is correctly rejected by
the new exact-inventory contract. Private vectors are not committed.

Full QA passes **1,132 tests**, with seven existing skips; coverage remains
**95.46% / 90.56%** above unchanged floors. Current and baseline source produce
byte-identical OpenAPI documents under the same final graph and settings
(`warmup_on_startup=False`, sorted JSON serialization). Native CodeQL Python
analysis retains only the two existing contextual setup alerts. No exclusions,
advisory suppressions or weakened test thresholds are introduced.

The [timing outcome](execution-timing-calibration.md) retains the failed
four-caller maximum-batch experiment rather than claiming universal capacity.
The [native archive](validation/execution-timing-2026-10-15.json) records images,
wheel deltas, audits, model/IR evidence, source hashes and limitations. PR #57 is
adopted locally; its original commits are not merged. Keep this scoped upgrade,
then review later versions independently against the same native/model gates.
No release, tag, branch or version bump is created.
