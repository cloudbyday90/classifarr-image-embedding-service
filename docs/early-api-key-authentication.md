# API-key rejection before body receipt

Assessment: 2026-10-03. Baseline: `ded0a63843c728abfaed29e8a8e6c8a05e80ebb8`.

## Design and recommendation

The native socket probe observed a complete 16 MiB valid JSON body received before
an unauthenticated request returned 401. FastAPI reads the route body before
solving its ordinary dependencies. Authenticate matched protected routes before
delegating to that body handler, using the same header policy as the dependency.

Use a small `authenticated_router.py` factory with a scoped `APIRoute` subclass.
The model, embedding, batch and admin routers select it; public health/readiness,
documentation and ordinary routing remain outside it. There is no duplicated
global path allowlist, body read, authentication cache or mutable request marker.
Retain the router dependency for OpenAPI security and normal validation. Successful
requests perform the same inexpensive key comparison twice.

Keep X-Api-Key precedence over Bearer, query-key refusal, dev-mode behavior,
constant-time content comparison and the existing 401/503 detail messages. Compare
UTF-8 bytes so a non-ASCII wrong credential produces 401 instead of a comparison
type error. Bind mandatory admin enforcement explicitly at router construction,
including deployments with a root path or a mounted prefix.

Early failures close unread HTTP/1 connections; HTTP/2 responses omit the
connection-specific header. Existing ingress admission and declared-size rejection
still run first. Keep ingress through error-response completion and transport
cleanup, then prove recovery. Correct credentials retain body limits, upload
deadlines, schemas and native ownership. Invalid JSON without credentials now
returns 401 before parsing instead of exposing a validation error.

## Official October research and tradeoffs

Sources were discovered through web/GitHub MCP on October 3:

- [FastAPI custom route handlers](https://fastapi.tiangolo.com/how-to/custom-request-and-route/)
  support wrapping a matched route before its original handler; router-local
  selection avoids imposing authentication on public probes.
- [Starlette exceptions](https://starlette.dev/exceptions/) places handled HTTP
  exceptions within routing. Middleware alternatives must construct responses
  themselves and maintain their own route-selection policy.
- [Python HMAC comparison](https://docs.python.org/3.12/library/hmac.html) supports
  byte inputs and avoids content-dependent short-circuit comparison; string inputs
  require ASCII. This does not promise equal timing for different lengths.
- [Uvicorn server behavior](https://github.com/Kludex/uvicorn/blob/main/docs/server-behavior.md)
  documents receive-driven buffering and `100 Continue`. Verify header-only
  rejection on actual parsers; a proxy may acknowledge or buffer uploads first.

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Keep only ordinary route dependencies | Existing API wiring | Complete body is read before auth | Replace timing |
| Global ASGI authentication middleware | Earliest application check | Duplicated route/method/public policy and response handling | Defer |
| Scoped authenticated route factory | Matched-route policy, shared validator, existing exception handling and schema | Two small checks on accepted requests; outer limits can reject first | Adopt |

Final stack: modular Python/FastAPI routers, shared header-only key dependency,
scoped early route check, existing ingress/body/deadline/transport services,
complete locked dependencies and native socket evidence. No new dependency,
authored JavaScript, production budget retuning or release is required.

## Outcome

The final locked Linux QA run passed **1,022 tests / seven existing optional or
platform skips**, including 35 new ASGI policy/ownership cases and 24 native
parser/TLS cases. Coverage is **95.31% lines / 90.30% branches**, above unchanged
89.42% / 79.37% floors. The new route module and authentication policy are fully
covered. Baseline/current canonical OpenAPI are byte-identical. Style/type,
copyright, skill metadata and all thirteen dependency profiles pass.

Native header-only requests declare the 16 MiB ceiling and `Expect: 100-continue`.
Both httptools and h11, with TLS and without it, return the exact 401 or configured
key failure 503, observe **zero body bytes**, send no interim 100 and close the
connection. Correct Bearer requests on fresh connections recover embedding and
batch results. ASGI regressions cover prefixes/mounts, credential precedence,
wrong non-ASCII values, query refusal, dev mode, routing, independent app settings,
outer limit precedence and retained ingress during stalled error-response send.

Real CPU and OpenVINO CPU socket replays pass with both pinned models resident,
one worker/owner and existing 4 GiB / 8 GiB no-swap containment. Authentication
returns 401 after **zero receipt**, replacing the earlier 16 MiB observation.
Eight staged uploads still hold at the body ceiling minus one byte; excess requests
reject on headers, seven canceled callers settle, retained maximum requests match
direct vectors, concurrent mixed batches queue and both remote children are reaped.
All final ingress, queue owners/waiters and read locks are zero, with no memory/PID
limit-hit or OOM deltas. Sampled socket-phase RSS reaches 2.94 GiB CPU / 4.55 GiB
OpenVINO. Shared page charging affects cgroup totals; these runs are not a paired
memory-savings or backend-speed comparison. CPU overlapped initial QA and OpenVINO
overlapped CodeQL. Existing immutable images used read-only source overlays;
production rebuilds and hosted execution are not claimed.

The initial concurrent QA run passed 1,018 cases but four existing 1.2-second remote
startup/network deadline cases expired before reaching their fixture server. The
same nine-case network suite passed in isolation, then the complete fresh QA run
passed after model/scanner work finished. No deadlines, tests or production network
code were loosened. Keep native timing experiments separate from competing work.

CodeQL Python security-extended covers 83 files with no new findings. Two unchanged
context results remain: explicit `--show-key` output and nonsecret 0644 publication.
Verified Gitleaks 8.30.1 finds no secrets in the source or 87-commit baseline history,
with full redaction and unchanged rules. Exact CPU/OpenVINO/QA inventories validate
56/58/70 hashed wheels and pass `pip check`; all 26 lock artifacts are byte-identical
to the baseline. Secret scan results, source/image/journal hashes, exact native cases, initial/final
QA records, schema identity and limitations are retained in the
[validation archive](validation/early-api-key-authentication-2026-10-03.json).
The [skill extension](project-early-auth-skill.md) and [PR availability](open-pr-availability.md)
have separate design/outcome records. No suitable unapplied PR was available.

Next fix [public probe quota identity](public-probe-rate-limits.md): a bounded
experiment reproduced quota bypass by rotating unverified credential headers on
public health. Then measure actual reverse-proxy buffering and target hardware.
Native evidence here is Linux CPython 3.12 and HTTP/1; hosted Windows and HTTP/2
transport remain separate gates. Production budgets and lock artifacts are retained;
delivery stays on master with Unreleased changes and no release or version bump.
