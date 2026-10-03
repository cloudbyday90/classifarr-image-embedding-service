# Verified native CI artifacts

Assessment date: 2026-10-03. This document covers native tools and builder images, separately from [workflow permissions and ownership](ci-execution-contracts.md).

## Design and publisher evidence

GitHub MCP release metadata supplied the actual asset URLs and SHA-256 digests committed in [.github/ci-tools.json](../.github/ci-tools.json). Full commit pins freeze action source; separate artifact contracts freeze downloaded native bytes.

| Component | Reviewed version | Installation contract | Local outcome |
|---|---|---|---|
| [Gitleaks](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1) | 8.30.1 | Hash archive before selecting its single regular executable | Clean/leak/error controls pass; token fully redacted; 78-commit history clean |
| [Trivy](https://github.com/aquasecurity/trivy/releases/tag/v0.69.3) | 0.69.3 | Preserve the existing wrapper's native version; hash archive; disable wrapper setup/cache | Filesystem and configuration scans pass under the existing HIGH/CRITICAL policy |
| [Buildx](https://github.com/docker/buildx/releases/tag/v0.37.2) | 0.37.2 | Hash standalone binary and atomically install the Docker CLI plugin | Native version execution passes |
| [CodeQL bundle](https://github.com/github/codeql-action/releases/tag/codeql-bundle-v2.27.1) | 2.27.1 | Verify complete Linux archive before the pinned action extracts it; override ambient toolcache selection | Actual CLI extraction and Python/Actions security-extended analysis complete |
| [QEMU/binfmt](https://github.com/tonistiigi/binfmt/releases/tag/deploy/v10.2.3-68) | QEMU 10.2.3 | Digest-pinned image; configure only ARM64 installation on disposable release runners | Digest pull and native version execution pass; host registrations were not changed |
| [BuildKit](https://github.com/moby/buildkit/releases/tag/v0.33.1) | 0.33.1 | Digest-pinned builder image; explicit daemon flags replace implicit insecure entitlements | Digest pull and native daemon version execution pass |
| [OSV scanner/reporter](https://github.com/google/osv-scanner-action/releases/tag/v2.6.0) | 2.6.0 | Digest-pinned image used directly in the owned reusable workflow | Real clean/vulnerable/no-lockfile and differential/full-report controls pass |

The release action's [flag handling](https://github.com/docker/setup-buildx-action/blob/f87e5991a6d7451dcb8d9637bfbc97413f497069/src/context.ts) adds insecure entitlements when the input is empty. Set the valid neutral daemon flag `--debug=false` explicitly; an empty string does not remove those defaults.

## Boundaries and tradeoffs

Three small Python modules separate the reviewed manifest, HTTPS downloader and installation/extraction. Initial URLs must name GitHub release assets; redirects remain on approved HTTPS asset hosts. Disable ambient proxies. Use bounded streaming, a ten-minute elapsed check with 30-second blocking reads, fixed byte ceilings and private staging. DNS and blocked operations can extend that elapsed check; workflow job timeouts remain the outer bound. Publish only after digest verification, without executing during installation. Re-download on every installation rather than trusting a restored executable cache.

Gitleaks/Trivy extraction copies only the exact bounded regular executable; it never unpacks arbitrary archive paths or links. CodeQL remains an archive for its reviewed action to extract. Tool installation supports Linux amd64 CI runners; service ARM64 support remains unchanged.

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| Trust wrapper downloads and restored binaries | Smaller local implementation; upstream owns setup | Artifact/cache identity can drift beyond the action pin | Replace on the identified native paths |
| Commit reviewed SHA-256 plus official asset URL | Detects changed bytes before execution; no verifier dependency | Does not independently authenticate an uncompromised publisher; refresh requires review | Adopt |
| Enforce signatures/provenance for all tools | Can authenticate builder identity | Coverage, trusted identities and verifier bootstrap differ across publishers | Follow-up once a supported identity policy is defined |

Keep vulnerability and misconfiguration databases current. Trivy's existing 0.69.3 native contract is retained rather than silently adopting the separately available 0.75.0 release; refresh the scanner deliberately with equivalent controls. GitHub runner images, setup-python's interpreter installation, apt repositories and host drivers remain external platform trust/update contracts. These changes freeze repository-controlled references and the identified tool/builder paths; they do not make the entire hosted platform immutable.

## Validation and outcome

All four native downloads match their reviewed publisher digests and install non-root. Boundary tests reject wrong hashes, empty/oversized/redirected responses, unsupported platforms, malformed manifests, archive links/traversal/duplicates and incomplete publication. Failed installation preserves an existing binary and removes private staging.

Actual Gitleaks controls return 0 for clean history, 2 for a synthetic secret with one fully redacted finding, and 1 for invalid configuration. Complete fetched repository history returns 0. Trivy reports zero selected HIGH/CRITICAL filesystem findings and zero selected configuration findings; its existing unfixed-vulnerability filter is preserved.

CodeQL Actions analysis returns zero findings. Python security-extended analysis reports two alerts in the unchanged `scripts/generate_env.py`: console output of the generated API key and plaintext storage in the intended Compose `.env`. No new CI helper is flagged. Tighten default secret display and owner-only publication as the next small fix; plaintext storage itself needs contextual treatment because Compose consumes that file. Do not suppress the alerts simply to claim a clean scan.

OSV's actual scanner produces a clean report (exit 0), a completed vulnerable fixture report (exit 1), and no report when no lockfile exists (exit 128). The new report guard rejects the last case. Offline reporter gates pass clean/clean and unchanged-vulnerability comparisons, and fail newly introduced and full-release vulnerabilities. These intentionally vulnerable fixtures are outside the repository and never installed in the service. The Dependabot path retains its 2.3.8 image with a direct native entrypoint; clean/vulnerable/no-lockfile controls likewise return 0/1/128.

The [validation archive](validation/ci-contracts-2026-10-03.json) records identities, source hashes and outcomes. Hosted cache/upload/SARIF integration, token downgrades for forks/Dependabot, release publication and privileged QEMU/BuildKit setup remain hosted integration gates. No registry deletion or release publication was performed locally.
