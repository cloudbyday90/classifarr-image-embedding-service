# Quota identity resource-contract skill extension

Assessment: 2026-10-03. Extend the existing repository resource-contract skill
with conditional quota guidance, alongside its authentication/body-order reference.

## Design and tradeoffs

The [skill](../.agents/skills/classifarr-resource-contracts/SKILL.md) routes quota
changes to [identity contracts](../.agents/skills/classifarr-resource-contracts/references/quota-identities.md).
The reference carries the project-specific failure modes: raw credential rotation,
empty Bearer keys skipped by SlowAPI, shared-key principal semantics, public
decorator selection and native forwarding trust. It ties storage ownership and
expiry to existing modules rather than introducing another checklist framework.

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Separate generic rate-limit skill | Broad reuse | Duplicates ownership guidance; may attract unrelated deployment redesign | Defer |
| Conditional project reference | Precise triggers and current module/test mapping | Requires maintaining framework-version evidence | Adopt |
| Skill-only policy without regressions | Small prose | Does not prove the empty-key or header-rotation fix | Reject |

Final stack: discoverable resource-contract skill, one scoped reference, shared
Python credential/identity services and observable ASGI/native regressions.
The skill grants no extra authority for external writes, deployment or releases.

## Evaluation and outcome

The implementation exercise covers public/private/dev modes, rotated and empty
headers, valid credential precedence, independent apps, mounts, expiration and
trusted/untrusted actual server forwarding. It deliberately preserves separate
endpoint quotas and uses a nonsecret principal so quota-exhaustion logs do not
retain the shared key. The existing authentication suite checks unchanged early
rejection and admin enforcement.

Focused QA passes 151 cases, including eight native parser/proxy cases; full locked
QA passes 1,051 with seven existing skips. All 29 new quota cases pass, schema bytes
are unchanged and the new service modules have full line/branch coverage. Skill
metadata, reference links, style, scoped types and copyright validation pass.
This is maintainer scenario evaluation with regressions, not an independent-agent
skill benchmark. The security workflow's independent investigation/review assesses
the code boundary separately.

[Implementation research and outcomes](public-probe-rate-limits.md) hold the
recommendations, alternatives and platform limitations.
