# Open PR 51: local OSV reusable workflow update

Assessment date: 2026-10-02. Repository baseline: `b6b6bc0bc7f308bcf5f3c3e41ca89049d0a24f6c`. Status: implemented and locally validated on master; the original PR remains open and unmerged.

## Selection and source identity

GitHub MCP returned 17 open PRs. Excluding the three still-open changes already adopted locally (33, 40 and 42) left 14 eligible candidates. A uniform random selection chose [PR 51](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/51): update the non-Dependabot OSV reusable PR workflow from 2.3.5 to 2.6.0. Inspected PR head: `8274224d70968747f4c4d51d179602f649ba7fc8`; base: `97f2d9578fca851abac08472477c3458bc9ae6b5`. MCP reconfirmed open/unmerged status after local validation.

MCP fetched the actual upstream workflow at both the discovered tag and its verified commit. The contents match. The [immutable official workflow](https://github.com/google/osv-scanner-action/blob/a345acffa64b0eaede81a3d9aae6141214d9c8fc/.github/workflows/osv-scanner-reusable-pr.yml) is pinned in `.github/workflows/osv-scanner.yml` to `a345acffa64b0eaede81a3d9aae6141214d9c8fc`, annotated `v2.6.0`. URLs came from GitHub MCP and official documentation search, not guessed links.

## Design and recommendation

The updated workflow checks that a scanner step reporting failure produced a nonempty results file before comparing old/new results. This distinguishes a completed scan that found vulnerabilities from failure to produce a report. A missing report must fail the job instead of yielding a misleading clean comparison. We adopt the upstream change through its reusable workflow; there is no locally copied workflow implementation to maintain.

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| Keep 2.3.5 | No caller change | Misses upstream report-completion guard and later scanner fixes | Reject |
| Adopt 2.6.0 at its full commit | Gets the guard; freezes the reviewed workflow file; smallest PR-equivalent change | Nested image is still a mutable upstream tag; future updates need review | Adopt |
| Fork the workflow and pin every nested dependency | Can freeze the complete execution graph | Adds ownership of upstream logic, reporting, SARIF and update maintenance | Defer to remaining CI dependency review |

[GitHub's official action-version guidance](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/find-and-customize-actions) recommends full commit SHAs verified against the original repository and explains the update tradeoff. Caller pinning freezes the YAML, not the container it selects: the upstream nested action points at scanner image `v2.6.0`. This residual supply-chain dependency remains a next-item candidate, alongside the older release workflow ref.

The caller's event filters and job permissions retain their existing scope: ordinary PR scans have contents/actions read and security-events write; Dependabot uses the previously reviewed direct scanner digest with contents read. Tag-only release scanning is separate. This iteration creates no release, tag, upstream PR merge or external review/comment. New application code remains modular Python; no CommonJS or new JavaScript is introduced.

## Local validation and outcome

The actual 2.6.0 scanner/reporter image resolved to digest `sha256:71ad04ab2f8798be47870f9b18817ad317c2f8f2f97aa6726ba10d5578bc174a`; local executions used that digest. Its version commands confirmed OSV 2.6.0 / SCALIBR 0.5.2 and reporter 2.6.0. Existing upstream reporter flags remain accepted, with a visible deprecation warning for `--output`.

The exact guard script extracted from the official workflow passed eight cases in a native Linux QA container. Missing and empty reports after either failed scan return failure; completed nonempty vulnerability reports proceed to comparison. Both-failure cases and the no-failure case were included. These checks exercise upstream shell behavior, not a rewritten model of it.

Native scanner controls queried current public advisory metadata using isolated fixtures outside the repository: `idna==3.20` returned exit 0 with a clean JSON report; deliberately old `idna==3.9.0` returned exit 1 with a complete vulnerability report. The intentionally vulnerable package was never installed in the service. The actual reporter then ran offline against those reports with the workflow's `--gh-annotations=true --fail-on-vuln=true` options:

| Old report | New report | Expected and actual exit | Meaning |
|---|---|---:|---|
| Clean | Clean | 0 | Clean comparison passes |
| Clean | Vulnerable | 1 | Newly introduced vulnerability fails |
| Vulnerable | Same vulnerable report | 0 | No newly introduced vulnerability |

Workflow lint passed with external ShellCheck/Pyflakes disabled. The caller ref matches the MCP-fetched upstream file and the verified tag commit. Application validation independently passed 456 tests and unchanged coverage floors; strict installed-package OSV audits found zero known vulnerabilities or skips in the 56-package runtime and 67-package QA image.

These are local component, syntax and policy checks. GitHub-hosted checkout/cache/artifact upload, runner permissions, SARIF integration and the complete reusable job were not executed locally. Existing baseline vulnerabilities intentionally remain outside a differential reporter's new-vulnerability failure policy; installed-package audits provide separate environment evidence.

## Next recommendation

Keep this minimal verified caller update. Add production-model/preprocessor and cached-IR revision contracts next, then complete dependency locks and review remaining workflow/nested container refs. Track the current priorities in the [recommendation stack](recommendation-stack.md).

## Subsequent CI iteration

The [CI execution](ci-execution-contracts.md), [native artifact](ci-native-artifacts.md) and [release cleanup](release-cleanup.md) records now implement the remaining reference, tool-download and credential review. This document retains its original iteration outcomes; current priorities are in the [recommendation stack](recommendation-stack.md).
