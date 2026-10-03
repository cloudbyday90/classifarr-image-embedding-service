# Authentication resource-contract AI skill extension

Assessment: 2026-10-03. Extend the existing resource-contract skill with a focused
authentication reference rather than add a competing general security workflow.

## Design

The [resource-contract skill](../.agents/skills/classifarr-resource-contracts/SKILL.md)
routes body-order and protected-router changes to
[early authentication contracts](../.agents/skills/classifarr-resource-contracts/references/early-authentication.md).
The reference explains the matched-route check, shared policy, deliberate second
dependency validation and existing ingress/response ownership. It records the
actual public endpoints and mandatory admin enforcement under mounts and prefixes.

Require observable zero-body rejection, configured-key failure, credential
precedence, wrong non-ASCII refusal, unchanged OpenAPI and authenticated recovery.
Native `Expect: 100-continue` probes exercise both HTTP/1 parsers and TLS. Separate
the proven boundary from HTTP/2 transport and reverse-proxy buffering claims.

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| New generic authentication/security skill | Wider trigger | Duplicates repository owners and overreaches this platform task | Defer |
| Focused reference in resource-contract skill | Shares current admission/cleanup rules; concrete routing and native evidence | Requires maintaining framework-boundary guidance | Adopt |
| Encode auth behavior in a skill-only checklist | Quick prose | No executable proof; can hide body-order regressions | Reject |

Final stack: existing discoverable skill, conditional reference, shared Python
router/policy services and executable ASGI/native evidence. No new authority for
PR, deployment, credential or external-message operations is introduced.

## Evaluation and outcome

The actual implementation exercise checks header-only 401/503, router selection,
independent app settings, mandatory prefixed/mounted admin policy, error-response
ownership and correct-key recovery. Negative controls preserve unsupported-method
routing and existing declared-size/ingress rejection. Unicode wrong credentials
test the comparison boundary. The capacity probe now refuses any unauthenticated
body receipt instead of uploading a complete body before checking 401.

Metadata/reference validation passes. All 59 new authentication cases pass,
including 24 native parser/TLS cases, and both real-model socket replays now require
zero-byte unauthenticated rejection. Full locked QA passes 1,022 cases with seven
existing skips. The OpenAPI comparison is byte-identical. This is maintainer
scenario evaluation with regressions, not an independent-agent benchmark.
Separate [authentication design and outcomes](early-api-key-authentication.md)
hold current research, validation history and implementation evidence.
