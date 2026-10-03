# Reviewed dependency and base refresh policy

Assessment date: 2026-10-02. Baseline: `eb61eb63167438cb1c14f18a600d56574552c937`.

## Evidence and official research

Pinned bytes give reviewed reproduction but do not automatically incorporate security fixes. [Docker build guidance](https://docs.docker.com/build/building/best-practices/), discovered through official-domain web search, distinguishes digest-pinned bases, fresh pulls, cache bypass and reviewed Dependabot updates. Existing OCI bases are pinned and Docker/Python update proposals arrive weekly. Ubuntu 25.10 is excluded from the eligible PR pool because the [official security announcement](https://lists.ubuntu.com/archives/ubuntu-security-announce/2026-July/010877.html) confirms support ended July 9, 2026.

## Alternatives and recommendation

| Policy | Pros | Cons | Decision |
|---|---|---|---|
| Automatically resolve latest packages on every build | Immediate uptake | Unreviewed artifact/native drift; weak rollback evidence | Reject |
| Freeze locks and bases indefinitely | Predictable bytes | Retains newly disclosed vulnerabilities | Reject |
| Reviewed weekly refresh plus urgent advisory response | Explicit provenance/diffs; numerical and operational gates; rollback record | Maintainer work and backend/architecture validation cost | Adopt |

## Design

Keep scheduled proposals and add periodic validation/auditing of the committed environments. Weekly CI uses `--pull --no-cache` against reviewed base digests so apt steps run again; it validates without publishing. Dependabot proposes changes to input intent and bases; a maintainer generates and reviews the complete affected lock/manifest pairs rather than relying on the bot to update derived wheel URLs or hashes. Input changes without regeneration fail the contract gate. Regular updates may reuse reviewed installed-version constraints; upgrades deliberately remove or adjust the relevant constraint. Do not silently upgrade native packages while adopting an unrelated maintenance PR.

Before accepting a refresh, run full offline QA/coverage, strict installed-environment OSV without skipped packages, dependency/inventory validation and each affected native backend smoke. Changes to Torch, Transformers or OpenVINO require production-model/artifact and representative capacity checks. CUDA/device-support changes require applicable hardware gates. Preserve old evidence and record new package/base identities in separate outcomes. Failed hashes, dependency checks or audits are failures to investigate, never reasons to suppress checks.

Review security advisories promptly rather than waiting for the weekly cycle. Keep immutable prior commits/images available for rollback, without reverting to a known vulnerable environment as a standing deployment. Align OpenVINO bindings/base, CUDA flavor/driver and supported OS lifetime. Current apt repositories continue providing OS fixes during clean builds; Python locks do not freeze apt or make entire images byte-identical. A deployment needing frozen OS bytes should adopt a reviewed snapshot/mirror with its own refresh policy.

Publishing remains tag-only and unchanged. This iteration creates no tag, release, version bump or upstream PR merge.

## Validation and outcome

Implemented weekly default-branch validation/manual dispatch, fresh scheduled backend builds, complete lock checks, isolated locked audit tooling and strict audits of runtime/tool inventories. Lock/input changes trigger the separate production-model workflow. Existing tag-only release conditions and publication dependencies remain unchanged.

Local locked QA passes **625 tests** with two optional production cases skipped and unchanged coverage floors. All five supported backend images pass fresh installation/inventory and native service smokes; all six runtime/QA environments and both audit-tool architectures pass strict OSV without known vulnerabilities or skipped packages. Actual CPU/OpenVINO production parity and representative maximum batches also pass. Details and immutable identities are in the separate [dependency outcome](dependency-locks.md) and its [validation archive](validation/dependency-locks-2026-10-02.json).

Workflow syntax and local commands are validated; scheduled GitHub execution is not observed locally. ARM checks use emulation, modern CUDA uses the available RTX 5070 Ti, and legacy NVIDIA/Intel GPU capacity requires appropriate hardware. Apt remains a separately refreshed source of OS packages; the Python locks do not imply identical full-image bytes or complete security assurance. The next item is remaining CI/reusable reference pinning, effective permissions and native tool-download verification, as prioritized in the [recommendation stack](recommendation-stack.md).
