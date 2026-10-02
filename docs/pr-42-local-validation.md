# Local implementation of open PR 42

Decision date: 2026-10-01. Status: implemented and locally validated.

## Discovery and scope

GitHub MCP discovered 16 open PRs. Excluding PR 28, already implemented in the prior task, a uniform random selection from the remaining 15 selected [PR 42](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/42): upgrade `actions/checkout` from v6 to v7. Separate MCP metadata confirmed open/unmerged state and retrieved the six-file, eight-location patch.

Inspected PR head: `c174b4d4ce6c8a54e3d5583998aa0fa08414c3ee`; PR base: `a764ad47d9ba29d5bb674ed07c04134283811292`. Local implementation starts from the later tested master commit `4ce8599684d834021723886467eae7ca6a9a393d` and copies the affected checkout changes without merging the PR.

## Official research and recommendation

Sources were discovered through web search or returned as canonical links by GitHub MCP. The MCP also fetched release metadata and source files at the exact commit.

- [Official checkout documentation](https://github.com/actions/checkout/blob/v7/README.md): v7 uses ESM and rejects customized fork checkout in trusted `pull_request_target`/`workflow_run` events unless explicitly opted in. Its Node 24 runner requirement is at least 2.327.1, with 2.329.0 needed for authenticated git commands inside container actions.
- [Official changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md): 7.0.1 includes follow-up guard, ref whitespace, and git configuration fixes.
- [GitHub workflow security guidance](https://docs.github.com/en/enterprise-cloud@latest/code-security/tutorials/secure-your-organization/protect-against-threats): full commit references prevent tag movement from substituting action code; keep token permissions minimal.
- [Verified v7.0.1 commit](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1): MCP commit metadata and `git ls-remote` independently matched the release and major tag to this SHA.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Copy mutable `@v7` reference exactly | Matches PR text and receives automatic patch updates | A moved tag changes executed code without repository review | Use v7 behavior, with stronger pinning |
| Pin full SHA for verified v7.0.1 | Immutable action code and follow-up fixes; dependency bots can propose reviewed updates | Patch updates require a repository change | Adopt across all eight checkout locations |
| Stay on v6 | Avoids major-version migration | Misses new ESM and fork-checkout protections | Reject |

The implementation deliberately refines the PR's mutable v7 tag to `3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1`. Existing event triggers, fetch-depth settings, and job permissions are preserved. No unsafe-checkout opt-in or JavaScript code is added. Other action pins are a separate follow-up.

## Local validation and outcome

`actionlint -shellcheck= -pyflakes=` passed before and after the workflow edits. Those optional external linters were disabled; this checks workflow syntax and expressions, not remote action execution.

A temporary clone of the official v7.0.1 release matched the pinned SHA. Its package metadata reports version 7.0.1 and `type: module`; `node --check dist/index.js` passed on Node 24.18.1. Running the actual bundled action with a synthetic trusted-event fork payload exited with the expected refusal before checkout. The test used a placeholder token, created no checkout, and cleaned its verified temporary directory. No credentials or GitHub writes were involved.

The complete service suite passed 236 tests after the admission refactor. Full hosted-runner execution and authenticated checkout were not simulated locally. The original PR remains open and unmerged; no review, comment, closing keyword, release, or package-version bump is part of delivery to `fix/bounded-batch-admission`.
