# Project dependency-migration skill

Design and assessment: 2026-10-03.

## Purpose and design

The project now has several distinct native dependency graphs. A routine bot PR
can update a declared floor already satisfied by the runtime, while a framework
deprecation can require an additive QA client without replacing SDK or production
probe classes. The new project skill makes that distinction explicit.

[Classifarr Dependency Migrations](../.agents/skills/classifarr-dependency-migrations/SKILL.md)
is discovered from the existing `.agents/skills` directory. Its short entrypoint
routes actual lock work to one reference and reuses the maintained lock/installer
scripts. It introduces no executable duplicate resolver or release authority.
Normal automatic discovery remains enabled, with an explicit invocation prompt in
`agents/openai.yaml`.

The [official skill guidance](https://learn.chatgpt.com/docs/build-skills?translationFallback=ja-JP)
was fetched on October 3. Instructions and supporting resources are used only where
they change project decisions; a standalone version lookup does not trigger the
full migration workflow.

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Extend the resource-lifetime skill with every dependency rule | One entrypoint | Attracts unrelated dependency work and obscures ownership | Reject |
| Add a second resolver and hard-coded environment catalog | Repeatable command | Duplicates maintained profile logic and drifts | Reject |
| Focused instruction skill using existing native tools | Narrow discovery, current profiles, explicit evidence limits | Still requires judgment on affected boundaries | Adopt |

Final recommendation: owner-first dependency review, constrained native graph
changes, isolated exact installation, contract tests and honest outcome records.

## Evaluation and outcome

The HTTPX2 API migration and PR #53 floor adoption are the first concrete use
cases. The skill validator passes without scaffold placeholders. Maintainer
evaluation against actual artifacts distinguished a declared floor from the
unchanged Transformers 5.18.0 runtime, preserved all twelve other profile graphs
and all 67 existing QA artifacts, and produced exactly three reviewed additions.
The old image failed the fallback-warning gate; the new complete graph passed
953 tests, unchanged coverage floors, six native inventories and a full strict
70-package advisory audit. SDK and probe imports still use their own HTTPX classes.

A standalone version lookup routes out of this workflow; a Windows transport
change routes to native validation; a GPU memory-sizing request routes to capacity
calibration. These are manual routing checks, not an independent agent benchmark.
The [migration record](httpx2-test-clients.md), [PR outcome](pr-53-local-validation.md)
and [validation archive](validation/httpx2-test-clients-2026-10-03.json) preserve the
observed evidence. No model API or external transmission is introduced by the
skill.
