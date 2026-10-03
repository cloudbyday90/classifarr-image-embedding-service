# Early authentication contracts

Use this reference for API-key/body ordering, protected router changes or framework
upgrades affecting the early route check. General token architecture, secret setup
and unrelated security reviews do not need this workflow.

Read `src/image_embedder/authenticated_router.py`, `security.py` and the affected
router factories. `authenticated_router(auth)` wraps a matched FastAPI route before
its ordinary body handler; it retains the dependency for schema security and
normal validation. Both stages use the same policy. Do not replace this with a
mutable request-state marker or a duplicated global path allowlist.

Health/readiness and documentation are public. Models, embedding and batch routes
follow `require_api_key`; the admin router binds `always_required=True`, including
root-path and mounted deployments. Preserve X-Api-Key precedence, Bearer parsing,
query-key refusal, byte-based constant-time comparison and configured-key failure.
Wrong non-ASCII credentials must produce 401 without a comparison type error.

Authentication failures must not receive, parse or dispatch body work. HTTP/1 early
errors close unread connections; HTTP/2 omits the connection-specific header.
Admission and declared-size limits remain outside routing and may reject first.
Keep ingress until response completion/transport cleanup. Accepted requests retain
upload deadlines, body/image budgets and detached-native ownership.

Validate observable boundaries: a forbidden ASGI receive, stalled error-response
ownership, routing redirects/405/404, prefixed/mounted admin protection, independent
app settings and unchanged OpenAPI. Run actual header-only `Expect: 100-continue`
requests against httptools/h11, with TLS and without it. Require no interim 100,
zero observed body bytes, server closure and authenticated recovery. These probes
do not establish proxy buffering or HTTP/2 transport support.

Focused checks:

```sh
python -m pytest tests/test_auth.py tests/test_auth_early.py tests/test_auth_sockets.py tests/test_request_input_limits.py tests/test_ingress.py tests/test_capacity_socket.py -o addopts=''
```

Use the existing locked QA image,
source snapshot and counters. Production capacity changes need the calibration
workflow; authentication-only edits do not justify a new benchmark framework.

Design and current evidence: [early API-key authentication](../../../../docs/early-api-key-authentication.md).
