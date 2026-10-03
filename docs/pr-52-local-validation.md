# PR 52 local pytest floor adoption

Assessment: October 3, 2026. Baseline: `8f57c727c7f8efbcaf72c27df716bffc5e4eb16c`.

GitHub MCP enumerated twelve open PRs. Already adopted checkout, Gitleaks, OSV, HTTPX and pytest-cov changes, superseded Transformers/Torch floors, and unsupported Ubuntu 25.10 were excluded. One uniform draw from **52, 48, 46, 27** selected [PR 52](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/52), head `be852cf5df1e86701e7aadd50f3f6b4d94dd6648`.

Adopt only `pytest>=9.1.1`; preserve all unrelated floors. QA already locks this version. Prove the sole dependency-input change, renew the QA input hash and keep all wheel identities/hashes and lock bytes unchanged. The official [pytest changelog](https://docs.pytest.org/en/latest/changelog.html), discovered through web MCP, documents the released 9.1.1 fixes and distinguishes the future 9.2 draft.

The stricter minimum matches the tested runner and includes its regression fixes. A floor alone does not freeze dependencies; complete target locks and exact native inventories remain necessary, at the cost of deliberate refresh review. Recommend the floor with existing locks rather than an unrelated SDK refresh. No original PR merge or fresh resolution is intended.

## Outcome

The sole normalized dependency input delta is `pytest>=9.0.3` to `pytest>=9.1.1`. The QA manifest renews only the development-input hash; all **67 artifact records**, every generated lock's bytes and all other manifests remain unchanged. All **nine** input contracts pass, and native non-root QA verifies the exact installed 67-wheel inventory with a clean pip check.

Full locked Linux QA passes **832 tests**, with seven existing platform/optional skips and **94.65% / 89.34%** line/branch coverage. Native Windows process/network validation also uses pytest 9.1.1 and passes 41 cases. The accompanying [input lifetime design](request-lifetimes.md) records runtime changes, research, pros/cons and measurements; the [archive](validation/request-lifetimes-2026-10-03.json) retains identities and log/source hashes.

GitHub MCP reconfirms PR 52 is open and unmerged at its selected immutable head. Delivery uses master without a new branch, original PR merge, release, tag or version bump. Recommend retaining the stricter floor together with the complete QA wheel contract.
