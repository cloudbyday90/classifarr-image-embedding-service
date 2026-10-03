# Open PR 45: local Pydantic minimum adoption

Assessment date: 2026-10-02. Baseline: `15a39337b272ef4db11eccdf58f1116f7bbea2cb`.

## Selection and official evidence

GitHub MCP returned 16 open PRs. Five already adopted or superseded proposals (33, 35, 40, 41, 42) were excluded. A uniform random selection among the other 11 chose [PR 45](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/45), at head `c284f29ed859b76210179220d8675be0f46c7029`. Its only patch changes `pydantic>=2.11,<3` to `pydantic>=2.13.5,<3`.

GitHub MCP followed the PR's actual [official 2.13.5 release](https://github.com/pydantic/pydantic/releases/tag/v2.13.5), published August 28, 2026. It includes validator reuse, garbage-collector traversal and smart-union fixes. The service's existing audited native images already contain Pydantic 2.13.5 and pydantic-core 2.46.5. Raising the declared floor records that tested runtime requirement; it is not a claim that a service vulnerability was reproduced.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Keep the old minimum | Allows older environments | Does not express the current tested version | Reject |
| Adopt the PR's minimum and retain the major-version ceiling | Small scoped change; aligns with the tested runtime | Ranged transitive selection is still not a lock | Adopt |
| Hash-lock all backend environments in this PR | Stronger reproduction | Separate multi-platform/vendor-wheel and refresh policy | Next iteration |

Application and validation additions are Python; any JavaScript we author must use ESM under the user's clarified requirement. No upstream PR merge, comment, release, tag or version bump is created.

## Validation and outcome

The final Python 3.12 suite passed **561 tests**, with two optional production-weight cases skipped. Coverage is **94.31% lines / 88.71% branches**, above unchanged floors. Both rebuilt images resolve **Pydantic 2.13.5 / pydantic-core 2.46.5**, and shipped native/server smokes pass `pip check`, non-root startup, authentication and graceful shutdown. Actual protected maximum-batch routes preserve schema/vector contracts on CPU/OpenVINO CPU.

Strict installed-environment OSV audits report zero known vulnerabilities and no skipped packages across **56 CPU, 58 OpenVINO and 67 QA packages**. Raising this minimum does not replace transitive hash locks; those are the next recommendation. Official GitHub MCP `releases/latest` readback reconfirmed **2.13.5** as stable, non-draft and non-prerelease on October 2, 2026. GitHub MCP also reconfirmed PR 45 is **open and unmerged** at the same head. Only its one-line dependency change is implemented locally; the accompanying platform work has separate [capacity](capacity-calibration.md), [deadline](embedding-deadlines.md) and [initialization](model-initialization.md) records. Delivery is directly on local/remote master with no release or tag.
