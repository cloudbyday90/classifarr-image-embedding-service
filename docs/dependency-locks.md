# Hash-locked Python backend environments

Assessment date: 2026-10-02. Baseline: `eb61eb63167438cb1c14f18a600d56574552c937`. Python and direct delivery on master remain user decisions.

Current follow-up, 2026-10-03: [native Windows contract validation](windows-native-validation.md) adds four separate platform/bootstrap profiles. The [HTTPX2 QA migration](httpx2-test-clients.md) adds three reviewed wheels to QA, now 70, while preserving every original QA artifact and all twelve other profile graphs. The current thirteen profiles contain 517 artifact records. [PR 53](pr-53-local-validation.md) renews the Transformers floor without refreshing production wheels. The [dependency-migration skill](project-dependency-migration-skill.md) guides these scoped changes; the original delivery measurements below retain their own dates and evidence.

## Evidence and official research

Existing backend files pin Torch flavor, but shared runtime/dev dependencies and installer tooling still resolve from ranges. The capacity record already provides reviewed installed versions; repeated resolution can change the numerical/artifact environment without a deliberate update.

Official links were discovered through web search and GitHub MCP. [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/?highlight=no-deps) requires every dependency to be pinned and hashed in hash-checking mode; binary-only selection avoids executing source builds. [pip repeatable installs](https://pip.pypa.io/en/latest/topics/repeatable-installs/) explains wheelhouse portability limits and hash checking; that page is development documentation, while the implementation uses the tested stable pip 26.2.1 interface. [Astral's index guidance](https://docs.astral.sh/uv/concepts/indexes/) demonstrates explicit package-to-index selection against dependency confusion. Its [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) documents accelerator-local versions and distinct vendor indexes.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Exact versions without artifact hashes | Simple version review | Allows changed/substituted distributions; incomplete transitive selection | Reject |
| A new universal project manager and lock format | Unified resolver/index features | New tooling and migration; deployment still needs target wheel/hardware validation | Defer |
| Target-specific pip reports rendered as complete direct-wheel/hash locks | Uses current installer; exact publisher artifacts; native marker resolution; no mixed indexes during installation | More lock files; approved artifacts must remain available; generation still trusts publisher metadata | Adopt |
| Frozen OS snapshot and mirrored wheelhouse | Stronger availability and byte-level reproduction | Repository/storage and timely security-refresh responsibility | Keep as an optional deployment policy |

## Design

Keep input ranges as update intent, alongside small profile, report-validation, generation and installation modules. Resolve inside CPython 3.12 Linux for the actual target architecture, using the approved Torch vendor index only for selecting its exact wheel and PyPI for the remaining graph. Preserve the current reviewed native versions as generation constraints where possible. Render all selected transitive wheels, including pip, with SHA-256 hashes and an origin/environment/input manifest.

Support CPU amd64/arm64, CUDA 13 amd64, legacy CUDA 12.6 amd64, OpenVINO amd64, Linux amd64 QA and architecture-specific audit-tool environments. A hashed bootstrap pins the installer used for resolution/installation. Reject unsupported environments, stale input contracts, malformed or duplicate artifacts, credentialed/unapproved URLs and altered lock text. Installation uses isolated pip, no index resolution, required hashes and binary wheels; dependency and exact-inventory checks must pass afterward. A reviewed Git writer can still replace locks and their attestations; hashes protect the approved artifact contract rather than authenticating the repository itself.

CPU/OpenVINO retain their current native versions and existing model/cache contracts. ARM CPU and CUDA shared packages align with that reviewed CPU baseline, including Transformers 5.18.0; backend Torch flavors and accelerator-specific dependencies retain their reviewed identities. The [official Torch artifact index](https://download.pytorch.org/whl/torch/) links its exact `download-r2.pytorch.org` distribution host, accepted only for Torch under the selected backend path. Non-Torch packages must use PyPI's artifact host.

Bootstrap always reinstalls the approved pip wheel, even when a seed installer has the same version. Its verification checks the reviewed version; full profiles also require `pip check` and an exact distribution inventory. Isolated audit targets must be empty and include their complete locked tooling graph without modifying the application's environment. Images normalize copied requirement files/manifests to root-owned 0644 and their directories to 0755 so native Linux generation's private temporary-file modes cannot prevent non-root verification.

CUDA ARM remains unsupported until the prior wheel metadata problem is repaired upstream. Windows/macOS range-based contributor setup remains separate from these Linux deployment contracts. No API, model precision, worker/admission or release behavior changes.

## Validation and outcome

Implemented nine complete profiles with 446 wheel records across their lock/manifest pairs. All committed contracts pass offline checks. Fresh Docker builds succeed for CPU amd64/arm64, CUDA 13 amd64, legacy CUDA 12.6 amd64 and OpenVINO amd64; a locked QA image also installs successfully. Shipped non-root smokes verify exact inventory, native projection, writable cache, authentication/readiness/startup and graceful shutdown. OpenVINO conversion/reload passes. CUDA 13 executes on the RTX 5070 Ti with driver 616.92; legacy CUDA uses its documented CPU fallback in this validation. ARM64 is emulated, without a claim about physical ARM throughput.

Full offline QA passes **625 tests**, with **two optional production-weight cases skipped**; the **64 new boundary cases** reject unapproved/credentialed/substituted artifacts, unsupported environments, stale/tampered contracts, inventory drift and unreviewed bootstrap versions. Coverage remains **94.31% lines / 88.71% branches**, above unchanged **89.42% / 79.37%** floors. Real pip negative controls reject a wrong SHA-256 and an incomplete dependency graph, while a fresh hashed bootstrap succeeds and a populated target remains untouched.

Strict installed-package OSV audits pass without known vulnerabilities or skipped packages: **56 CPU amd64**, **56 CPU arm64**, **75 CUDA**, **75 legacy CUDA**, **58 OpenVINO** and **67 QA**. Both isolated audit-tool profiles contain **29** audited packages. CPU/OpenVINO retain every reviewed baseline package version. Actual offline production checks pass both model aliases, single/batch/direct parity and protected API contracts; OpenVINO reload parity also passes. Representative 224-pixel batches **1/8/32**, with both models resident, pass within CPU **4 GiB** and OpenVINO **8 GiB** containment, with no recorded memory-limit events or OOM kills. These warm/representative observations supplement the earlier cold/export sizing record rather than replacing deployment calibration.

The [validation archive](validation/dependency-locks-2026-10-02.json) records profile identities, package inventories, source/lock/log hashes, final image identities, native outcomes and representative capacity journals. Raw logs, wheels and model assets remain outside Git. Ruff, scoped Pyright, workflow lint, copyright, coverage ratchet, secret and whitespace checks pass. The first ARM build's Ubuntu DNS failure was retried successfully without skipping checks. Hardware outside the tested NVIDIA device, physical ARM/Intel GPU execution and hosted GitHub jobs remain separate validation scopes. Hash locks cover approved Python artifacts, while apt/driver refresh follows the independent [update policy](runtime-update-policy.md); this is not a complete repository or OS security audit.

## Subsequent CI iteration

The [CI execution](ci-execution-contracts.md), [native artifact](ci-native-artifacts.md) and [release cleanup](release-cleanup.md) records now implement the remaining reference, tool-download and credential review. This document retains its original iteration outcomes; current priorities are in the [recommendation stack](recommendation-stack.md).
