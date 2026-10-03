# Project capacity calibration skill

Assessment date: 2026-10-03. Scope: measured deployment sizing and capacity-probe changes.

## Research and choices

Current official [skill discovery documentation](https://learn.chatgpt.com/docs/build-skills), discovered and opened through web tools, describes repository `.agents/skills` discovery and concise name/description metadata. The [OpenAI evaluation guide](https://developers.openai.com/blog/eval-skills?trk=article-ssr-frontend-pulse_little-text-block) recommends evaluating observable outcomes. The existing resource-contract skill addresses runtime ownership changes; the new skill focuses on experimental evidence and sizing decisions.

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| Put every benchmark instruction in the ownership skill | One entrypoint | Unrelated runtime refactors load hardware-specific experiment detail | Add focused routing only |
| Complementary instruction-only capacity skill | Clear selection; reuses tested helpers; project-maintained measurement map | Requires interpretation; automatic selection is not guaranteed | Adopt |
| Duplicate the probes inside the skill | Self-contained executables | Divergent runners and another validation surface | Reject |

## Design

[SKILL.md](../.agents/skills/classifarr-capacity-calibration/SKILL.md) identifies capacity questions, rejects missing GPU/CPU fallback claims, validates canceled native outcomes, and distinguishes sampled, lifetime and allocator peaks. A [reference](../.agents/skills/classifarr-capacity-calibration/references/measurement-contracts.md) maps existing helpers and focused validation. UI metadata permits normal implicit discovery. Existing resource-contract references route capacity questions to the sibling skill. Neither skill adds external publication authority or changes user-selected Python/master delivery.

## Validation and outcome

The skill-creator validator accepts the package and local links resolve. Manual scenario review covers CUDA without GPU, transient anonymous output, canceled mixed callers, shared GPU activity, dependency-only changes and documentation-only corrections. Required outcomes are refusal to mislabel CPU as GPU, lower-bound/contextual measurement interpretation, native outcome/child ownership verification, and avoiding unrelated hardware work. This is a manual instruction evaluation, not an independent model invocation or automatic-discovery benchmark. The existing resource-contract skill is present in the session catalog; the new skill may require a new session to refresh discovery.

The executed mixed probe and unavailable-GPU, filesystem and detached-failure regressions provide evidence for the workflow's concrete checks. Native measurements and final operating recommendations live in [deployment capacity](deployment-capacity.md), without copying volatile hardware values into skill instructions.
