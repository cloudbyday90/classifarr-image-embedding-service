# PR 53: local Transformers minimum adoption

On 2026-10-03 GitHub MCP listed nine open PRs: 53, 44, 42, 41, 40, 35, 34, 33 and 27. Fresh [PR 53](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/53) raises `transformers>=5.10.1` to `transformers>=5.17.0`. Its head is `d9100e430be2fa312ce5a8ff837698873d97b874`, base `3b9573ba9ef156fa45c898e611ecc04a4d64bf09`, open/unmerged.

Eligible pool: `[53]`. Uniform random draw: `0.5933955162861888`; selected 53. Earlier assessments excluded it as superseded by the 5.18.0 runtime graph; fresh review shows its declared floor is still unapplied and can be adopted without downgrading that graph. PR 35 would lower the current Torch requirement and force a CPU-only local version; 44 targets an expired interim Ubuntu base; 42/34/33/27 are already adopted; 41/40 target replaced scanner wrappers.

The [official Transformers release collection](https://github.com/huggingface/transformers/releases), discovered through search, confirms the upstream release. The existing reviewed runtime uses 5.18.0 with verified catalog artifacts and passes the proposed minimum. Adopt the PR's higher floor locally while retaining those versions, model identities, preprocessing and backend contracts.

Pros: removes obsolete declared compatibility without refreshing working runtimes. Cons: a minimum is not a complete lock and does not independently prove every version above it. Recommendation: adopt the floor, renew affected attestations and compare every production artifact/hash. The separate HTTPX2 QA additions are not a Transformers upgrade.

## Outcome

The declared minimum is now 5.17.0. All twelve non-QA artifact graphs remain
identical to baseline, including every production wheel/hash; QA preserves all 67
existing artifacts with only the separate three-package HTTPX2 addition. Six native
environment checks passed against renewed inputs: CPU amd64, emulated CPU arm64,
OpenVINO, modern CUDA, legacy CUDA and the new QA graph. They retain Transformers
5.18.0. No model identity, preprocessing or production transport source changed.

Full QA passed 953 tests with seven existing optional/platform skips and unchanged
coverage floors. The complete installed 70-package QA graph passed strict OSV with
no known vulnerabilities/skips. See the separate [migration outcome](httpx2-test-clients.md)
and [validation archive](validation/httpx2-test-clients-2026-10-03.json) for evidence.

GitHub MCP reconfirmed the same PR head remains open/unmerged after validation.
Delivery is directly on master; the original PR is not merged and no release,
tag, version bump or separate branch is created. A minimum alone still does not
prove every hypothetical version above it; complete locks remain the deployment
contract.
