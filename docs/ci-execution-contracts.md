# CI execution contracts

Assessment date: 2026-10-03. Baseline: `2e3813aa1b2da6caf2729ce9eea8d95946ba011f`. This iteration works directly on master and creates no release.

## Design and official evidence

The remaining CI references use mutable tags. A pinned OSV caller still selects a mutable nested scanner image. Gitleaks' upstream installer downloads an unchecked native archive, and Docker release setup selects floating Buildx, BuildKit and QEMU artifacts. Copyright checkout has no contents permission. Docker Hub cleanup interpolates secrets into shell code and accepts incomplete HTTP responses.

[GitHub's secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use?learn=getting_started&learnProduct=actions) supports full commit pins verified in the original repository, minimum job permissions and passing dynamic values through environment variables. [Reusable-workflow guidance](https://docs.github.com/en/enterprise-cloud%40latest/actions/reference/workflows-and-actions/reusing-workflow-configurations) explains that callees cannot increase caller permissions. A caller pin alone does not freeze its nested execution graph.

GitHub MCP discovered official releases, dereferenced annotated tags and supplied actual asset URLs/digests. Docker registry inspection supplies content digests for the publisher images. Small Python services will verify native artifacts before extraction or execution, with bounded downloads, fixed platform contracts and atomic publication. Authored JavaScript must use ESM; this implementation retains Python and introduces no CommonJS.

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Pin only top-level actions | Small change; Dependabot can propose updates | Native and nested dependencies can still drift | Insufficient alone |
| Pin actions and own bounded native/image contracts | Reviewed bytes; smaller credential surface; predictable failure | Maintainer must review refreshed hashes and owned OSV workflow | Adopt |
| Require publisher provenance for every artifact | Can authenticate build identity beyond byte integrity | Publisher coverage and verifier trust vary | Add when identity policy and coverage are verified |

[GitHub attestation guidance](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations) distinguishes provenance verification from fetching metadata. Checksums identify reviewed bytes; they do not independently establish an uncompromised publisher or builder. Advisory databases remain intentionally current, rather than frozen with scanner binaries.

## Implemented behavior

Keep API/model contracts unchanged. Gitleaks scans complete fetched history with full redaction, publishes a sanitized SARIF artifact and a static job summary, and fails on leaks or operational errors. Remove automatic PR comments and their write grant. OSV preserves differential PR gating, complete-report checks and tag-only release scans through an owned reusable workflow with digest-pinned scanner/reporter bytes. Trivy retains its existing severity, unfixed-vulnerability and reporting policy while using a verified binary and pinned wrapper.

All checkout steps disable persisted credentials. Release publication and package cleanup remain gated by tag pushes and successful validation; no release job is exercised against external registries during local testing. Docker Hub cleanup keeps credentials in memory, bounds and validates pagination, and rejects malformed responses before deletion.

The owned OSV workflow scans separate exact base/head checkouts rather than switching an authenticated working tree. It uses the reviewed 2.6.0 image for both scanner and reporter and current output-file flags. Dependabot retains the reviewed 2.3.8 image and contents-read scope, with the direct native entrypoint and explicit failure when no lockfile exists. The older release caller is replaced by the same owned digest-pinned 2.6.0 full-scan contract.

## Effective permissions

| Jobs | Granted permissions | Purpose and limits |
|---|---|---|
| QA/backend, copyright, model/capacity and Gitleaks | contents read | Checkout; no PR/repository write. Artifacts use the runner service without a new repository write grant |
| Dependabot OSV | contents read | Direct vulnerability gate; no SARIF upload |
| CodeQL, Trivy and ordinary/release OSV | contents/actions read, security-events write | Checkout and SARIF integration; no repository/package/PR writes |
| Docker publication | contents read, packages write | Successful validation plus version-tag push; Docker Hub credentials are step inputs |
| GHCR/Docker Hub cleanup | contents read, packages write | Successful publication plus version-tag push; Hub credentials only in its cleanup step |

Every workflow denies permissions by default. GitHub may further downgrade tokens for forks/Dependabot; effective hosted behavior must be checked on a real run. No permission-elevation trigger is introduced. Copyright checkout now has its required explicit contents-read grant.

Native tool/builder contracts and their tradeoffs are in [native artifacts](ci-native-artifacts.md). Credentials, API research, conservative retention and outcomes are in [release cleanup](release-cleanup.md).

## Outcome

The full locked QA suite passes **714 tests**, with two optional production-weight cases skipped. **89** new boundary/policy cases pass, and coverage remains **94.31% lines / 88.71% branches**, above unchanged floors. Six existing native environments pass exact inventories against renewed input attestations; their wheel graphs are unchanged. Random [PR 54](pr-54-local-validation.md) is adopted locally.

All native downloads match reviewed hashes; actual Gitleaks, Trivy, OSV scanner/reporter, CodeQL, Buildx, QEMU and BuildKit component executions complete. CodeQL Actions has zero findings; Python reports two existing secret-setup alerts, described with their context in the native record. Workflow lint, Ruff, scoped Pyright, copyright, coverage, whitespace and secret checks pass. Hosted integration, privileged runner setup and registry publication/deletion are separate gates. See the [validation archive](validation/ci-contracts-2026-10-03.json).

Recommend retaining the modular Python service, complete backend wheel/hash locks, verified model/IR contracts, digest-pinned builder/scanner images and minimum job credentials. Commit pins and native contracts improve reviewability at the cost of intentional refresh and owned OSV maintenance. The next small item is owner-only secret-file publication with explicit key display; then bound aggregate upload/DNS/download lifetime.
