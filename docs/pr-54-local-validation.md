# Open PR 54: local Requests floor adoption

Assessment date: 2026-10-03. Baseline: `2e3813aa1b2da6caf2729ce9eea8d95946ba011f`.

## Selection and design

GitHub MCP returned 14 open PRs. Excluding already adopted or superseded proposals and the unsupported Ubuntu 25.10 proposal left seven eligible candidates: 54, 52, 48, 47, 46, 34 and 27. A uniform random draw selected [PR 54](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/54), head `f4e1c530c79bdce09961d1b6227a9fbf248d6457`: raise Requests' minimum from 2.33.0 to 2.34.2.

The [official Requests release history](https://requests.readthedocs.io/en/latest/community/updates/) confirms stable 2.34.2 and its inline typing correction. Every committed runtime profile already selects the same 2.34.2 wheel. Adopt the floor locally and update affected input attestations after validating their unchanged inventories against the stricter requirement. Preserve every wheel URL, version and hash.

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| Retain the old floor | No input changes | Update intent understates the tested minimum | Reject |
| Adopt 2.34.2 with unchanged reviewed wheel graphs | Matches selected PR; no native SDK drift | Affected input attestations must be renewed and checked | Adopt |
| Refresh all packages at the same time | Broader maintenance opportunity | Introduces unrelated model/native changes | Separate reviewed refresh |

## Outcome

The floor is adopted locally with a one-line requirement change. Six input attestations are renewed after checking that the only normalized input change is the selected floor and that every reviewed Requests wheel is already 2.34.2. All wheel URLs, versions, hashes and lock text remain identical; this is not a fresh dependency resolution.

The six actual existing CPU amd64/ARM64, QA, OpenVINO, modern CUDA and legacy CUDA environments pass pip check and exact inventory verification against the new repository input contracts. No native SDK changes or new backend builds are claimed. The full locked suite passes **714 tests**, with two optional production-weight cases skipped; coverage remains **94.31% / 88.71%** above unchanged floors. Remote transport regressions run as part of that suite.

The accompanying CI iteration has its own [execution](ci-execution-contracts.md), [native artifact](ci-native-artifacts.md), [release cleanup](release-cleanup.md) and [validation archive](validation/ci-contracts-2026-10-03.json) records. The original PR remains open/unmerged at pre-delivery readback. Delivery uses master; no release, tag, separate branch or version bump is created.
