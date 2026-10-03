# PR 27: local Trivy action adoption

Assessment date: 2026-10-03. [PR 27](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/27) is selected from the eligible open pool `[48, 46, 27]` with uniform draw `0.9980073941223363`. Other open PRs already adopted, superseded or targeting an unsupported OS were excluded before the draw. The PR remains unmerged; its change is implemented directly on local `master`.

## Design and provenance

The proposal updates three Trivy action uses to v0.36.0. GitHub MCP resolved the tag to commit `ed142fd0673e97e23eac54620cfb913e5ce36c25` and fetched its action metadata and entrypoint by that commit. [The immutable upstream release](https://github.com/aquasecurity/trivy-action/releases/tag/v0.36.0) was published April 22, 2026. [Reviewed action source](https://github.com/aquasecurity/trivy-action/blob/ed142fd0673e97e23eac54620cfb913e5ce36c25/action.yaml) is composite Bash; [the entrypoint](https://github.com/aquasecurity/trivy-action/blob/ed142fd0673e97e23eac54620cfb913e5ce36c25/entrypoint.sh) invokes an argument array and reads shell-escaped input exports. No authored JavaScript or CommonJS is added.

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Pin reviewed v0.36.0, keep native binary verification | Adopts the selected PR with immutable source and existing credential controls | Upstream default scanner differs; nested action paths require continued disabling | Adopt |
| Follow the mutable action tag and defaults | Less local maintenance | Unreviewed source/native version/cache paths can change | Reject |
| Remove the wrapper | Fewer dependencies | Changes existing workflow input/report integration beyond this PR | Evaluate separately if maintenance becomes costly |

Preserve `skip-setup-trivy: true` and `cache: false` for all three invocations. Each also sets `TRIVY_CMD` to the absolute verified-tools binary path, preventing PATH or ambient command substitution. The separately SHA-256 verified Trivy 0.69.3 binary remains the scanner, regardless of the action's v0.70.0 default. Its nested setup/cache actions remain unreachable. Checkout credentials are not persisted; no PR write, SBOM submission or additional token is introduced. Preserve configured SARIF reports and failure gates.

## Outcome

All three workflow invocations now pin the resolved v0.36.0 commit. Native local execution of the exact fetched input-export, entrypoint and cleanup Bash steps passes **12 scan cases** with verified Trivy **0.69.3**, network disabled and a previously populated native database/check cache. Deliberate fixtures produce one secret result, 25 selected vulnerability results and one configuration result; each failure gate returns 1. Clean controls and the checkout-equivalent repository return 0; repository filesystem vulnerability/secret and configuration SARIF reports contain zero selected results. Deliberate fixtures remain outside source control.

Actual `printf %q` exports preserve shell metacharacters literally, and cleanup removes exports after successful and failed scans. No setup/cache action, SBOM upload or token is exercised. Native CodeQL Actions analyzes all nine workflows and reports zero alerts; immutable-ref, least-permission and native-binary policy tests and actionlint pass. Full locked QA passes with unchanged dependency contracts and exact 67-package inventory. The [archive](validation/response-send-lifetimes-2026-10-03.json) retains source/blob identities, scan-case results and evidence hashes. Hosted SARIF publication and GitHub composite runner orchestration remain separate from local shell execution. The original PR remains unmerged; no release, version bump or tag is created.
