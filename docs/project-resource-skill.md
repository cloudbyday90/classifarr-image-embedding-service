# Classifarr resource-contract skill

## Design and scope

Create the repository-scoped `classifarr-resource-contracts` skill at `.agents/skills/classifarr-resource-contracts`. It guides changes to remote fetching, batch preparation, ingress and inference ownership through a small contract map and change-scoped validation plan. It is an instruction-only skill: the repository already contains test, lock, coverage and native-tool helpers, so duplicating them would increase maintenance and drift.

Success means that a maintainer identifies the true owner and release condition, separates caller timeout from completion, specifies partial-result/error semantics, chooses meaningful boundary tests, preserves complete dependency/model contracts and records measured outcomes with limitations. The skill should activate for resource-lifetime and batch refactors, and stay out of unrelated copy edits, release operations and generic version questions. It grants no external action, branch, release or credential authority beyond the current user request.

## Official October 2026 research

The current [OpenAI skill guide](https://learn.chatgpt.com/docs/build-skills), reached through an official documentation link, says Codex discovers repository skills under `.agents/skills` from the working directory through the repository root. A skill has required name/description metadata and can load references only when needed. The [official evaluation guide](https://developers.openai.com/blog/eval-skills?trk=article-ssr-frontend-pulse_little-text-block) recommends measurable outcomes, explicit/manual invocation and positive/negative trigger cases. Its older `.codex/skills` example differs from the current location guide; use the current repository location. No API, hosted skill upload or new external dependency is required.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Repository skill plus concise reference | Versioned with code; focused discovery; reusable contract/validation routing | Needs maintenance when ownership changes | Adopt |
| Put the workflow in global agent instructions | Always present | Unrelated tasks consume context; weaker task routing | Reject |
| Add an automated all-checks runner | One command | Duplicates existing helpers; expensive and environment-dependent | Defer until a repeated concrete gap exists |
| User-only skill copy | Available outside this checkout | Can drift or collide with the checked-in skill | Reject for this project workflow |

The skill references repository code/documents rather than freezing current image IDs or test totals. Validation escalates according to changed boundaries, with actual OS processes for termination claims and native model/device work only when those contracts change. Authored code stays Python; any future authored JavaScript must use ES modules.

## Evaluation plan and outcome

Run the skill-creator structural validator, check reference paths and UI metadata, and manually apply the workflow to this batch-budget change. Exercise explicit activation, a shared timeout request, cancellation/permit recovery and cache-hit timing as positive scenarios; documentation wording and factual package-version lookup are negative scenarios. Record what was actually evaluated; do not claim an independent model-discovery benchmark or hosted integration test.

The skill is implemented as `SKILL.md`, one contract/validation reference and `agents/openai.yaml`, with implicit discovery allowed and a default prompt containing `$classifarr-resource-contracts`. Invoke it explicitly with **“Use $classifarr-resource-contracts to plan and validate a shared remote batch deadline.”** It is checked in at the current repository discovery location; no user-only duplicate, API upload, executable helper or external dependency was added.

The skill-creator validator passes. Every contract-map file and local Markdown reference resolves. Manual use on this change identified per-invocation fetch ownership, independent native inference, current-byte cache timing, complete dependency attestations and appropriate process/API tests. Seven manual routing/authority scenarios cover explicit use, implicit batch/cancellation/cache work, standalone dependency-floor work, copy edits, version lookup and untrusted PR text requesting credential disclosure. Standalone floor work is outside the automatic resource trigger but is supported when this skill is explicitly invoked. This is a root-agent review and workflow exercise, not an independent model-discovery or security benchmark; the session's existing skill catalog was not refreshed to prove selector appearance. Current OpenAI guidance says local changes are detected automatically and a restart may be needed if they do not appear.

The accompanying implementation passes **914 locked Linux QA tests / 57 native Windows cases**. No skill-specific wording-matching tests were added; actual ownership, cache, native cleanup and inventory outcomes provide the meaningful evidence. The [validation archive](validation/remote-batch-budget-2026-10-03.json) records the scenario review and evidence hashes. Use the skill for the next capacity iteration, and extend it only when repeated usage demonstrates a missing decision or validation route.

See the [skill](../.agents/skills/classifarr-resource-contracts/SKILL.md), [batch design](remote-batch-budget.md) and [recommendation stack](recommendation-stack.md).
