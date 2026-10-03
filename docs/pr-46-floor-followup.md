# PR 46 renewed FastAPI floor: local adoption

On 2026-10-03 GitHub MCP listed ten open PRs. PR 46 now raises `fastapi>=0.142.1,<1` to `fastapi>=0.142.2,<1`; its head is `24bc812e60e2b59c036ae232d3afec0ec4232e2d`, base `f45ee654295f313e428063707dd049bda4c79d99`, open/unmerged. This is a new floor after the earlier [PR 46 adoption](pr-46-local-validation.md).

Eligible pool: `[46]`. Uniform random draw: `0.6636218198823813`, selecting 46. PRs 53/35 are superseded by reviewed runtime graphs; 42/34/33/27 are already adopted; 41/40 target replaced scanner wrappers; 44 targets an unsupported expired Ubuntu interim base. No new graph is selected merely to create a larger pool.

[PR 46](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/46) is implemented locally without merging its upstream PR. The [official FastAPI release collection](https://github.com/fastapi/fastapi/releases?after=0.44.1), discovered through search, identifies stable 0.142.2 on September 30 and an automatic OpenTelemetry startup fix. The production and QA locks already contain that release.

Pros: the declared minimum matches the reviewed patched environment. Cons: a floor does not lock downstream arbitrary installations. Recommendation: adopt the floor, retain complete production locks, renew affected input attestations, verify unchanged wheel records/text and exact native inventories. New Windows graphs are a separate validation feature, not this PR's dependency upgrade.

## Outcome

The new minimum is applied. Six affected input attestations are renewed; all nine existing Linux profiles' **446 wheel records/hashes and lock text remain unchanged**. Exact installed inventories and pip checks pass in QA, CPU amd64/arm64, OpenVINO and both CUDA images. Full locked QA passes **947 tests** with seven existing skips; coverage remains **95.22% / 90.12%**, above unchanged floors. Native Windows framework/transport checks also pass with FastAPI 0.142.2. The separate [Windows record](windows-native-validation.md) and [archive](validation/windows-native-2026-10-03.json) give precise environments and limits.

GitHub MCP reconfirmed the original PR remains open/unmerged at the same head on delivery. No release, tag, version bump or separate branch is created.
