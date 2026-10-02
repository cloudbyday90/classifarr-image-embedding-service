# Local implementation of open PR 49

Decision date: 2026-10-02. Status: implemented and locally validated.

## Discovery and design

GitHub MCP returned 18 open PRs. Excluding the three present already implemented PRs (33, 40, 42; historical 28 was absent) left 15 candidates; a uniform random index selected [PR 49](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/49). MCP confirmed it is open/unmerged and fetched the two-workflow setup-python v6 → v7 patch. Inspected head: `2057f813b324b0c5f4da78a1ee925c22f11493df`; base: `97f2d9578fca851abac08472477c3458bc9ae6b5`.

The [official tagged action metadata](https://github.com/actions/setup-python/blob/v7.0.0/action.yml) uses Node 24 setup and cache-save entries. The [tagged package](https://github.com/actions/setup-python/blob/v7.0.0/package.json) declares ESM. Its [verified README](https://github.com/actions/setup-python/blob/5fda3b95a4ea91299a34e894583c3862153e4b97/README.md) describes the ESM migration and Node 24's runner minimum 2.327.1. Repository refs `v7` and `v7.0.0` both resolved to `5fda3b95a4ea91299a34e894583c3862153e4b97`.

Apply the update to both current setup-python uses, pinned to that complete commit with a v7.0.0 comment. Preserve Python 3.12, pip caching/dependency paths, checkout pins, permissions, and release triggers. Neither workflow uses the removed pip-install input.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Keep v6 | No action migration | Retains older internals and a mutable major reference | Reject |
| Use mutable v7 | Exact Dependabot proposal | Executed code can change when a tag moves | Reject |
| Verified v7.0.0 commit pin | ESM action; immutable reviewed code; existing inputs retained | Pin updates require review; runner compatibility still matters | Adopt |

## Local validation and outcome

Both workflow uses are pinned to `5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0`. `actionlint -shellcheck= -pyflakes=` passed; optional shellcheck/pyflakes were not run. A metadata check verified both full pins, preserved Python 3.12 selectors and supported input names against the exact tagged action.

A temporary detached clone matched the pinned SHA. Under Node **24.18.1**, `node --check dist/setup/index.js` and `node --check dist/cache-save/index.js` passed. Both actual bundled ESM entries executed successfully with an isolated Windows runner/tool-cache fixture and an existing Python **3.12.13** installation. Setup resolved the 3.12 selector, wrote the expected Python path/version outputs and environment files; the post entry completed. Only runner-file fixtures were written; caching and toolchain downloads were disabled, and the subprocess environment contained no GitHub token.

Hosted Linux runner execution, authenticated cache restore/save and distribution downloads were not simulated. The runner minimum remains a deployment requirement. Service-suite, coverage, packaged-runtime/audit and copyright outcomes are in the [recommendation stack](recommendation-stack.md). GitHub MCP reconfirmed PR 49 is open and unmerged after local validation, with the inspected head unchanged.

Work is directly on master, per user instruction. The original PR will remain open/unmerged; no review/comment, closing keyword, release, or tag is created.
