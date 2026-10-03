# Stable public-probe and embedding quota identities

Assessment: 2026-10-03. Delivery baseline:
`65597eaa3e1e925abcecb02b3a1b6f22727cb455`. Python remains the application language;
this iteration creates no JavaScript, branch, release or original PR merge.

## Reproduced boundary

At the baseline, `security.make_limiter` returns any nonempty X-Api-Key or Bearer value as the quota
identity. Public `/health` intentionally requires no key. An offline experiment
used one client address, a two-request/minute health limit and eight requests:

| Invalid credential headers | Observed statuses | Storage identities |
|---|---|---|
| Same value four times | 200, 200, 429, 429 | One |
| Four distinct values | 200, 200, 200, 200 | Four more |

Rotating an unverified header bypassed the shared public-probe quota and created
five stored identities in total. No body, model work or external traffic was used.
This demonstrates the selection problem; it is not a stress test or a quantified
memory-exhaustion claim. Early protected-route rejection does not authenticate
public health headers and therefore does not resolve this independent boundary.

An independent investigation found another representation: `Bearer ` with an
empty suffix produced an empty identity, which pinned SlowAPI skipped entirely.
Four health and four dev embedding requests each returned 200 under a two-request
quota and created no storage keys. The locked baseline reproduction also returned
200 for a legitimate protected model request. These are offline bounded controls.

## Recommendation and tradeoffs

[SlowAPI's official examples](https://github.com/laurentS/slowapi/blob/master/docs/examples.md),
discovered through web search/fetch, describe configured quota keys and endpoint
scope. Apply a stable, verified identity policy instead of accepting arbitrary
header strings as new public-probe callers.

Current official sources were discovered and fetched through web/GitHub MCP:

- [SlowAPI API](https://slowapi.readthedocs.io/en/latest/api/) supports per-decorator
  identity selectors. Attach the public policy to its decorators, preserving
  separate endpoint scopes and the default protected selector.
- [SlowAPI source](https://github.com/laurentS/slowapi/blob/master/slowapi/extension.py)
  shows quota evaluation depends on nonempty key and scope. Local evidence uses
  the actual locked 0.1.10 implementation, rather than assuming latest source
  and installed code are identical.
- [Uvicorn proxy guidance](https://uvicorn.dev/deployment/) assigns forwarding
  interpretation to explicit trusted peers. The application consumes the resulting
  client address; it does not trust arbitrary forwarding headers itself.
- [Limits 5.8.0 strategies](https://limits.readthedocs.io/en/stable/strategies.html)
  documents fixed-window reset and boundary bursts. Retain the existing strategy;
  verify expiry against the actual pinned runtime rather than retuning windows.

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Retain arbitrary header identities on health | Existing behavior | Reproduced quota bypass and extra keys | Replace |
| Key public probes by effective client address | Stable bucket despite any credential header, including valid keys | Shared NATs; server proxy trust still matters | Adopt |
| Invalid credentials use address; valid protected traffic uses one verified principal | Closes dev/empty-key sibling paths; no secret values in storage/logs | Shared service key shares quota across clients; one extra small comparison | Adopt |
| Hash each unverified credential | Hides raw values | Rotation still creates new identities; empty input needs separate handling | Reject |
| Add distributed storage or a global probe bucket | Coordinates replicas or caps all probes | Deployment dependency or cross-client starvation; does not itself fix identity verification | Defer |

## Design and recommendation stack

`credentials.py` owns existing header selection and byte comparison, shared by
authentication and `rate_limits.py`. Public decorators explicitly select
`client_address_key`. The default selector validates the configured shared key,
then returns a fixed nonsecret service-principal label. Everything else falls
back to a nonempty namespaced address. No mutable request marker or path allowlist
is introduced; settings and storage belong to their app factory.

Keep all quota values and separate health/readiness and single/batch counters.
Public probes require no key. Dev traffic stays accepted, but arbitrary headers
cannot manufacture new quota identities. Valid dev credentials use the same
verified principal as production. Protected invalid requests retain early 401/503,
admin remains mandatory and model/vector schemas remain stable.

Retain the existing server forwarding configuration. NAT/proxy clients may share
a public-probe quota; operators must explicitly trust their actual proxy peers.
Changing address fields after trusted forwarding is a server/deployment boundary,
not credential-header rotation. The bounded experiment does not establish a
global storage ceiling, distributed protection or workload capacity.

Final stack: modular Python credential and identity services, matched-router
authentication, explicit public decorators, existing SlowAPI/limits storage and
server proxy trust, locked dependencies and ASGI/native evidence.

## Outcome

Focused locked QA passes 151 cases, including eight actual httptools/h11,
trusted/untrusted-peer and repeated-forwarding-header cases. Both public probes
charge rotating credentials to one address; correct forwarded clients remain
independent only through trusted peers. ASGI checks cover dev single/batch paths,
empty credentials, NAT/port behavior, valid principal sharing, storage/log secrecy,
mounts, independent apps, separate counters and deterministic expiry recovery.

The original bounded reproduction now returns 429 for all four rotated headers
after the fixed-header quota is exhausted and retains one identity instead of
five. Empty-Bearer health and dev embedding both return 200, 200, 429, 429 and
retain their two separate endpoint identities; the legitimate protected model
control remains 200. Both original triggers therefore no longer reproduce.

Full locked Linux CPython 3.12.3 QA passes **1,051 tests / seven existing skips**,
including 29 new cases (21 policy/ASGI and eight native sockets), in 127.44 seconds.
Coverage is **95.32% lines / 90.27% branches**, above unchanged 89.42% / 79.37%
floors. Credential and quota modules have full line/branch coverage. Existing
auth, ingress, body/deadline, native TLS and model-contract fixtures remain green.
Canonical OpenAPI is byte-identical with SHA-256
`74ed5f5e2ed776296fd487c09edb905f0ad9498bb55c0a0cb3b07ac6fb422517`.
All thirteen lock profiles and the exact 70-wheel QA inventory pass; all 26 lock
artifacts are byte-identical to the baseline.

Ruff, scoped Pyright, copyright, skill metadata and local Markdown links pass.
CodeQL 2.27.1 Python security-extended covers 85 Python files with no new findings.
Two unchanged context results remain: explicit `--show-key` output and nonsecret
0644 config publication. No suppression changed. Its final extraction uses the
candidate source import paths; an earlier invocation using image import paths was
superseded. Only Python security queries ran; workflow extraction is not a fresh
Actions security analysis.

The security fix workflow supplied separate independent investigation and candidate
review. Investigation identified the empty-key representation; review found no
concrete surviving bypass/regression. The reviewer could not execute stale local
venvs; the parent ran the complete locked checks instead. Native evidence is Linux
HTTP/1 with fake model fixtures, not hosted Windows, HTTP/2, an operator proxy or
new CPU/OpenVINO model-capacity measurements. Model/runtime graphs and production
budgets are unchanged, so unrelated performance experiments were not repeated.
No suitable unapplied open PR was available; delivery stays on master with
Unreleased updates and no version bump.

Fresh hash-verified Gitleaks 8.30.1 finds no secrets in the candidate source or
88-commit baseline history, with full redaction and unchanged rules. Native checks
use immutable QA image
`sha256:90384e69d6c5b548a42835358e5a60c0cbfdc0326d66e35f6bc0cc1538180803`
plus read-only tracked/nonignored source. Exact source hashes, commands, JUnit,
coverage, before/after controls and scanner evidence are retained in the target's
standalone security artifact collection. Repository MDs contain the portable
design/outcome summary. Committed history is checked again before push.

Next: measure actual operator reverse-proxy buffering and target workload headroom,
then observe hosted Windows execution and remaining accelerator/platform gates.
The [early-authentication outcome](early-api-key-authentication.md) retains the
earlier body-order evidence; the [quota skill design](project-quota-identity-skill.md)
records the focused workflow extension separately.
