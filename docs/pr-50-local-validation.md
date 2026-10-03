# Open PR 50: local Uvicorn minimum adoption

Assessment date: 2026-10-02. Baseline: `eb61eb63167438cb1c14f18a600d56574552c937`.

## Selection and official evidence

GitHub MCP returned 15 open PRs. Already adopted/superseded proposals 33, 35, 40, 41 and 42 were excluded; PR 44 targets Ubuntu 25.10, whose support ended July 9, 2026 according to the official source linked in the [update policy](runtime-update-policy.md). A uniform random draw among the remaining nine (54, 53, 52, 50, 48, 47, 46, 34, 27) selected [PR 50](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/50), head `0ee0e5d8ffe7d7de57764f3fcab304b95165f76e`, open/unmerged.

Its one-line patch raises `uvicorn[standard]>=0.41,<1` to `uvicorn[standard]>=0.54.0,<1`. GitHub MCP followed its actual [official release page](https://github.com/Kludex/uvicorn/releases). Existing audited CPU/OpenVINO/QA environments already contain Uvicorn 0.54.0; the minimum records the tested server requirement without silently upgrading the native inference stack.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Keep the older minimum | More older environments allowed | Does not express current tested server behavior | Reject |
| Adopt the minimum and major-version ceiling locally | Small explicit maintenance change; current startup/shutdown contract | Requires refreshed locks and future reviewed updates | Adopt |
| Merge the upstream PR | Provider records the original PR merge | User requested local implementation/testing | Do not use |

Authored additions remain modular Python; any authored JavaScript must use ESM. No upstream PR comment/merge, release, tag or separate branch is created.

## Validation and outcome

The one-line Uvicorn minimum is adopted locally and included in all refreshed runtime/QA locks at **0.54.0**. Full locked QA passes **625 tests**, with two optional production-weight tests skipped; coverage stays at **94.31% lines / 88.71% branches** above unchanged floors. All five built backend profiles pass actual authentication/readiness/startup and graceful shutdown through the shipped server, together with exact dependency inventories and native projection checks. CPU/OpenVINO production-model/API parity and OpenVINO export/reload pass, supplementing the small smoke models.

Strict OSV audits cover all installed runtime/QA packages and isolated tooling without known vulnerabilities or skipped packages. ARM uses emulation, CUDA 13 uses the RTX 5070 Ti, and legacy CUDA is checked in CPU fallback mode. The separate [dependency record](dependency-locks.md) contains the complete counts, wheel hashes and validation scopes. Authored changes remain modular Python; no JavaScript/CommonJS source is introduced.

GitHub MCP reconfirmed the selected immutable head remains **open/unmerged** before delivery. Implementation and tests are local; the original PR is not merged. Delivery is directly to `master` with the changelog under Unreleased, without a separate branch, release, tag or version bump. Next pin remaining CI/reusable references and verify their native tool artifacts.
