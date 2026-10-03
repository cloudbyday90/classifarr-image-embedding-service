# Response send lifetimes

Design date: 2026-10-03. Baseline: `c2f78ef16d02775a12b97b7ff4bc12c53975e135`. Python and the existing HTTP schemas are retained.

## Evidence and options

[Uvicorn flow control](https://uvicorn.dev/server-behavior/) pauses ASGI sends when the write buffer crosses its high watermark. Inspection of our locked Uvicorn 0.54.0 shows the final body write can return with buffered bytes remaining. [ASGI HTTP](https://asgi.readthedocs.io/en/latest/specs/www.html) defines no portable transport-abort event. [Python transport documentation](https://docs.python.org/3.12/library/asyncio-protocol.html) distinguishes flushing `close()` from buffer-discarding `abort()`, followed by `connection_lost()`. [AnyIO cancellation guidance](https://anyio.readthedocs.io/en/latest/cancellation.html) supports shielded asynchronous cleanup and preserving external cancellation. These sources were discovered with web MCP; upstream action provenance was resolved with GitHub MCP. Research reflects October 3, 2026, not future changes during the rest of the month.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Proxy timeout alone | Mature edge protection | Does not establish application transport cleanup or owner ordering; direct access differs | Complementary deployment protection |
| Pure ASGI cancellation | Portable; small | No socket abort; final buffered write can escape the timer | Insufficient alone |
| Uvicorn transport adapter with total send budget | Covers buffered terminal writes, streaming gaps and all application responses; confirms closure before cancellation | Depends on narrow pinned Uvicorn internals; HTTP/1 implementation | Implement and test against the locked server |
| Explicit asyncio loop and observable TLS socket buffers | Accounts for queued ciphertext; avoids premature terminal completion | Forfeits automatic uvloop selection; deployment performance requires calibration | Use in the shipped launcher |
| Replace the HTTP server | Could offer native write deadlines | Broader protocol/operational migration without local evidence | Defer |

## Design

`RESPONSE_SEND_TIMEOUT_SECONDS` / `[server].response_send_timeout_seconds` defaults to 30 seconds and must be positive and finite. The monotonic budget starts at response headers and includes body production, every send and terminal user-space buffer drain; chunks never reset it. Expiry aborts the connection and waits for its closure callback before cancelling cooperative application work. It sends no replacement status after headers. Ingress remains held through drain and cancellation cleanup; native inference ownership retains its existing independent rules.

A small response guard and separate HTTP protocol adapter wrap each connection's loaded ASGI application. The shipped launcher selects asyncio explicitly and retains automatic httptools/h11 parser selection. This covers application errors, ingress rejections, upload timeouts and probe responses outside the FastAPI middleware stack. Uvicorn's own parser errors/concurrency refusal bypass the application and are not claimed to receive this guard. Final drain means the local transport buffer is empty, not proof the peer consumed or acknowledged every byte. Cooperative cancellation and OS cleanup are not hard real-time guarantees. Background tasks after completed sending have no new deadline.

Native terminal-write tests reproduced premature completion with TLS on both loops: the TLS buffer can be empty while the underlying socket still owns encrypted data. The adapter includes CPython's `_ssl_protocol._transport` buffer in its drain check. uvloop's Cython TLS object hides that transport, so alternate opaque TLS transports fail closed at terminal drain with an explicit error instead of accepting an unverifiable completion. Use asyncio for TLS; plaintext uvloop remains covered as an alternate deployment. This private CPython boundary, Uvicorn's per-connection `app`/`flow`/transport callbacks and protocol factory acceptance must be retested on upgrades. No parser implementation is copied or patched.

Use `python -m image_embedder.server` for these guarantees. A stock alternate Uvicorn invocation does not install the adapter. HTTP/2/zttp and WebSockets are outside this HTTP/1 adapter; future server upgrades must repeat native protocol tests. Keep edge timeouts and service supervisor shutdown budgets aligned.

## Validation and outcome

Final locked Linux QA passes **888 tests**, with seven existing platform/optional skips and one upstream TestClient HTTPX deprecation warning. Coverage is **95.15% lines / 89.91% branches**, above unchanged **89.42% / 79.37%** floors. No dependency wheels, version, release or tag changes are made.

The **56 added cases** cover configuration, total budgets, errors/external cancellation, trailers, real httptools/h11 sockets, asyncio/uvloop, TLS, final single writes, continuous streaming, producer gaps, disconnects, FastAPI request-resource cleanup during Starlette streaming, keep-alive reuse and successful byte preservation. Reduced test socket buffers create controlled Linux backpressure with 8 MiB terminal fixtures. Windows loopback can drain that write into OS buffers despite an unread client; those cases validate empty local buffers and release rather than claiming peer receipt or forcing an artificial abort. Production socket watermarks and API responses remain unchanged.

The guard also checks the monotonic clock before and after sending, so a delayed watcher cannot disarm an already overdue response. Repeated aborts are idempotent after real connection loss, including Python 3.14 TLS objects which clear internal state on closure. Native Windows Uvicorn 0.54.0 checks pass **31 cases** with eighteen expected uvloop skips on Python 3.14.5, AnyIO 4.12.1 and the host's older FastAPI/Starlette stack; this is supplementary evidence, not the locked deployed Linux graph. Future streaming resources must be explicitly owned by request cleanup: cancellation while an async iterator is suspended at a yield does not establish immediate iterator finalization. Existing API responses are JSON.

The native tests assert connection loss while ingress is still active, removal from Uvicorn's connection set, completion of request tasks and restored admission capacity. A server uses two real spawned workers in each existing locked CPU/OpenVINO/CUDA image; actual `/health` and `/models` requests and SIGTERM shutdown pass with source overlays. This checks the pickleable factory and shipped launcher without rebuilding images or repeating unchanged production-model inference. Native CodeQL Actions has zero alerts; Python retains the two documented setup-helper alerts, with no new response-lifetime alert or suppression. Exact figures, source identities and additional checks are archived in the [validation record](validation/response-send-lifetimes-2026-10-03.json).

## Recommendation stack

Implement the bounded HTTP/1 send adapter now. Next add a shared total remote-fetch budget across sequential batches, preserving kill/reap and inference ownership. Then calibrate target accelerator/concurrent-input, PID and temporary-storage budgets and establish recurring native Windows coverage. Keep Python; reconsider larger server or worker changes only when profiling supports their added complexity.
