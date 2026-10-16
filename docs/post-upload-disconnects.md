# Post-upload embedding disconnects

Assessment: 2026-10-16. Baseline: `8dad842ee51106aa3bea17eb98ae138b2959fe98`.

## Problem and researched decision

The previous cooperative checkpoints stopped future chunks when an inference
waiter was cancelled. The matched embedding routes did not receive again after
FastAPI parsed the body. A caller closing its socket could therefore retain its
waiter until the embedding deadline, even though the executor could already
accept a stop signal. This change connects those two existing boundaries.

Official sources were located through web MCP on October 16, 2026. The
[ASGI HTTP specification](https://asgi.readthedocs.io/en/stable/specs/www.html)
provides a disconnect receive event and permits send failures to race that event.
It also uses disconnect after response completion. The
[Uvicorn server contract](https://uvicorn.dev/server-behavior/)
confirms that receive after a completed response returns disconnect. These
facts require limiting observation to the period before response generation.

The [FastAPI Request reference](https://fastapi.tiangolo.com/reference/request/)
exposes the underlying receive channel; its disconnect polling implementation
also reads that channel. Our inference from these sources is to give one
blocking watcher exclusive ownership after body parsing, instead of polling
concurrently with upload handling. This is a project design choice, not an ASGI
requirement that every route cancel on disconnect.

The [Python asyncio task documentation](https://docs.python.org/3.12/library/asyncio-task.html)
distinguishes cancellation of a waiting task from shielded work and recommends
retaining task references. The existing executor remains the inference owner;
this helper only owns the route's wait and its receive task.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Retain deadline-only detection | Smallest task count; existing behavior | Abandoned callers retain queued work until the deadline | Replace for valid embedding work |
| Watch all requests in middleware | Central location | Competes with body consumers; response completion can look like abandonment | Reject for this scope |
| Watch only parsed embedding waits | One receive owner; prompt queued cancellation; uses existing cooperative checkpoints | Two bounded helper tasks per active embedding request; cannot interrupt active native work | Adopt |

## Design

`request_disconnect.py` races a receive watcher against a callable that awaits
the existing executor or batch-window submission. Both routes enter it only
after FastAPI parses their typed body and after route validation. Empty explicit
batches do not start a watcher. Authentication, quota checks and upload/ingress
limits remain in their existing order.

Only `http.disconnect` is accepted after the consumed body. Unexpected ASGI
events fail the operation rather than silently discarding more body data.
A result already ready when the race is inspected wins, including its ordinary
exception. Every exit cancels pending helpers and settles both before response
serialization. AnyIO shielding protects this cleanup from an enclosing cancel
scope; repeated direct asyncio cancellation is deferred until settlement and
then re-raised. There are no detached receive tasks or per-request mutable
embedder flags.

On disconnect the route returns an empty internal 499 response to complete the
ASGI call on the already disconnected connection. It does not log a service
failure or publish a new timeout policy. Successful/error schemas and connected
client responses retain their existing behavior. The existing response-send
deadline, terminal drain and abort ownership remain responsible after this
helper exits.

Cancellation removes queued admission promptly. A dispatched owner keeps its
permit and read lock until the active operation returns and closes resources.
Its existing per-owner stop event skips future cooperative work. Cancelling one
coalesced member leaves live members attached; only the last departing member
signals the shared owner. A stuck native call still requires process-level
containment. A current remote fetch remains bounded by its existing fetch budget.

The native Windows gate now includes the actual matched-route socket cases.
Its target-native graphs add SlowAPI 0.1.10, Limits 5.8.0, Deprecated 3.0.0 and
the matching CPython Wrapt 2.5.0 wheel, preserving every previous artifact.
The tests use the application's own lifespan context, so no extra lifespan
package is needed. These Windows environments still omit production model
runtimes and do not establish Windows inference support.

## Validation outcome

The final frozen-source Linux QA run passes **1,203 tests**, with seven existing
platform/optional skips. Coverage is **95.86% lines / 91.24% branches**, above
unchanged **89.42% / 79.37%** floors. The new helper has 100% line/branch coverage.
Current and baseline OpenAPI JSON are byte-identical under the same locked graph
and settings, SHA-256
`204c4510b3f884af2a2d2e32b3226b994841006f833fadc233ae8eb35d99f86c`.

The 41 added Linux cases comprise nine helper cases and 32 real socket cases.
They cover both HTTP/1 parsers, asyncio plain/TLS and uvloop plain HTTP, active
and queued closes, live/last coalesced members, a slow complete upload and
successful keep-alive reuse. They assert detached ownership before releasing
the controlled worker, native stop observation, settled admission/read locks
and recovered vectors. Helper cases cover ready-result/error races, unexpected
receive events/errors, asyncio/AnyIO/deadline cancellation and repeated parent
cancellation during cleanup. Existing full-suite authentication, ingress,
remote-child, deadline, cache and chunking regressions also pass.

Two fresh Windows 11 x64 environments pass the entire mandatory native gate:
**120 passed / 16 permitted setup skips / 32 explicit uvloop deselections each**,
on CPython 3.13.3 and 3.14.2 with Proactor sockets. All 18 new supported Windows
socket cases execute per interpreter. Each exact 37-package inventory and pip
check passes. Those interpreter patches are locally available evidence, not a
claim about current hosted Server 2025 or latest patch execution.

The first Windows attempt correctly failed collection because the new actual-app
test imported an unavailable lifespan helper. Using the native app lifespan and
adding the reviewed SlowAPI middleware graph resolves the actual dependencies;
no mandatory case is skipped. The first Linux matrix also exposed the existing
opaque uvloop/TLS drain refusal. The final matrix tests supported combinations;
production asyncio and existing refusal regressions remain intact.

Exact native installed inventories and pip checks pass for QA (70), CPU
amd64/arm64 (56 each), OpenVINO (58), CUDA and legacy CUDA (75 each). ARM64
executes under QEMU, not physical ARM performance testing. Strict OSV audits
cover all five amd64 installed graphs, the isolated 29-package auditor and both
37-package Windows lock graphs, with complete inventory validation and zero
known vulnerabilities or skipped packages. CPU ARM64's verified package/version
inventory matches audited CPU amd64; no native ARM auditor execution is claimed.
Wheel-byte checks remain separate from OSV's hash-presence audit behavior.

Ruff and scoped Pyright pass. Native CodeQL 2.27.1 / Python queries 1.8.11
security-extended indexes all 97 production/helper Python files and retains only
the two earlier contextual setup alerts: explicit `--show-key` output and readable
nonsecret defaults. No new alert or suppression is introduced. This invocation
does not claim a separate Actions query analysis.
The [validation archive](validation/disconnect-2026-10-16.json)
preserves commands, source identities, exact inventories and outcome summaries.
The socket workload uses controlled checkpoint-aware workers and deterministic
vectors. It proves transport and ownership behavior; no new production-weight,
accelerator throughput, Docker forwarding or host-capacity result is claimed.

## Recommendation stack

Keep Python, direct Docker publishing, one inference owner, native chunk size
eight, existing budgets and the independent owner lifecycle. Adopt matched-route
disconnect observation with native parser/TLS and coalescing gates. Next run the
existing poster/landscape PNG/JPEG/WebP calibration on the deployment host to
choose caller pressure and chunk size. Treat the Ubuntu/Python base migration
as a separate native wheel/backend project; see the [PR adoption record](pr-61-local-validation.md).
