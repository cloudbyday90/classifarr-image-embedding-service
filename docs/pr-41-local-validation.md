# Open PR 41: local Gitleaks Node 24 adoption

Assessment date: 2026-10-02. Baseline: `484c7ed3515e97fe74899d932cb89b4346b21174`. This record covers a local implementation of the selected PR; validation results are recorded below. No upstream PR merge or release is requested.

## Selection and official evidence

GitHub MCP returned 16 open PRs. Four dependency proposals already superseded by locally adopted changes were excluded; a uniform random choice among the 12 remaining candidates selected [PR 41](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/41), upgrading Gitleaks action 2.3.9 to 3.0.0. Its inspected head was `66ba03671b3deea9e808de88ff6ce7fb038c4681`; the PR changes one workflow action reference.

GitHub MCP fetched the official action metadata, README, package metadata and scanner source. Git remote tag inspection and a local detached checkout verified v3.0.0 at `e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e`. The [action metadata](https://github.com/gitleaks/gitleaks-action/blob/e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e/action.yml) declares Node 24. The [README](https://github.com/gitleaks/gitleaks-action/blob/e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e/README.md) describes the runtime upgrade and the minimum runner version, v2.327.1. The upstream bundle retains CommonJS packaging. The user explicitly approved PR 41 and clarified that ESM is required for code we author, so this external packaging is accepted. Application additions remain Python; the temporary local validation helper uses ESM.

The [scanner wrapper](https://github.com/gitleaks/gitleaks-action/blob/e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e/src/gitleaks.js) invokes the native scanner with redaction, SARIF output and leak exit code 2. Its [entrypoint](https://github.com/gitleaks/gitleaks-action/blob/e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e/src/index.js) defaults to native Gitleaks 8.24.3 and maps detected leaks to action exit 1. GitHub MCP release metadata supplied the actual [native release](https://github.com/gitleaks/gitleaks/releases/tag/v8.24.3) asset and checksum URLs; they were not inferred from a guessed download link. The downloaded Linux x64 archive is checked against its published SHA-256, `9991e0b2903da4c8f6122b5c3186448b927a5da4deef1fe45271c3793f4ee29c`.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Retain 2.3.9 | No runtime transition | Leaves the selected maintenance update unapplied | Reject |
| Adopt the v3 tag | Matches the PR directly | Tag can move | Reject |
| Adopt verified v3.0.0 by full commit | Node 24 compatibility; reviewed immutable wrapper | Hosted runners must support Node 24; native binary download/cache is a separate dependency | Adopt |
| Replace the wrapper with a new authored action | Could redesign permissions and native artifact verification | Broader maintenance and reporting contract work | Defer |

Only the Gitleaks `uses` reference changes. Existing triggers, full-history checkout and job permissions stay intact. The original PR is left open/unmerged. A future CI dependency review should consider remaining refs, cache/reporting dependencies and the wrapper's native binary download verification separately.

## Local validation and outcome

Validation executes the exact checked-out `dist/index.js` under Node 24 in a non-root Linux container. A local HTTP fixture supplies the owner-type response. Ephemeral repositories provide clean history, a synthetic token and an invalid configuration. Comments, artifacts and summaries are disabled for the local runs; the fixture records API requests and refuses any unexpected method/path. No real GitHub credential is supplied, and no external comment or artifact is written.

The actual Node 24.19.0 bundle passed all three controls: clean action/native exits **0/0**, detected-leak exits **1/2** with one redacted SARIF finding, and invalid-configuration exits **1/1** with the wrapper's unexpected-error message. The clean SARIF contained no findings; the synthetic token appeared in neither scanner output nor SARIF. The only fixture API calls were three `GET /users/fixture` reads. All downloaded native archives matched the published checksum. The local fixture removes each completed download before the next action invocation because the upstream installer uses a fixed temporary filename and the hosted cache service is absent here.

GitHub MCP reconfirmed PR 41 is open and unmerged. Hosted cache availability, actual runner permissions and GitHub reporting remain CI concerns. The published binary checksum protects this local validation download; the upstream wrapper itself does not implement that checksum check.

The full application suite, installed-package audits and workflow lint also cover the accompanying model-contract iteration. See [model artifact contracts](model-artifact-contracts.md) and the [recommendation stack](recommendation-stack.md) for their separate design and outcomes. No release, tag or version bump is created.

## Subsequent CI iteration

The [CI execution](ci-execution-contracts.md), [native artifact](ci-native-artifacts.md) and [release cleanup](release-cleanup.md) records now implement the remaining reference, tool-download and credential review. This document retains its original iteration outcomes; current priorities are in the [recommendation stack](recommendation-stack.md).
