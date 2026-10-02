# Input body and image allocation limits

Assessment date: 2026-10-02. Baseline: `97f2d9578fca851abac08472477c3458bc9ae6b5`. Status: implemented and locally validated.

## Observed behavior and research

I inspected `main.create_app`, the embedding routes, and `ImageEmbedder` at the baseline. JSON bodies had no application size limit. `_decode_base64` checked bytes after allocation; `_image_from_bytes` immediately converted to RGB. `embed_batch` retained every uncached compressed payload and decoded image. Count admission and worker ownership are already bounded, but do not bound these allocations.

Official URLs were discovered with web search and GitHub MCP. [Starlette's middleware documentation](https://www.starlette.io/middleware/) describes actual ASGI byte counting, including missing or understated Content-Length. Its [release notes](https://www.starlette.io/release-notes/) introduce body limits in 1.6.0. Require this version explicitly rather than depending on a transitive version that may lack the API.

[Pillow's Image documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=paste) explains lazy opening and its warning/error thresholds for decompression bombs. Check dimensions before load/conversion and preserve Pillow's own protection; do not mutate global `MAX_IMAGE_PIXELS` or process-wide warning filters. [Python documents strict base64 decoding](https://docs.python.org/3.12/library/base64.html); preserve `validate=True` after a length check that requires no decoded copy.

[CLIP's official processor documentation](https://huggingface.co/docs/transformers/model_doc/clip) describes shortest-edge resizing before center crop. A narrow input can create a large intermediate despite a small source pixel count. Bound both source and projected resize pixels without changing accepted-image preprocessing. [Uvicorn documents read flow control](https://www.uvicorn.org/server-behavior/) and stops buffering remaining bodies after response completion; application limits complement server/proxy deployment limits.

## Options and recommendation

| Option | Pros | Cons and residual limits | Decision |
|---|---|---|---|
| 1. Proxy-only body limit | Rejects uploads before application work; operationally useful | Does not protect direct access, base64 decode, image expansion, or remote batches | Complementary deployment control |
| 2. Application body limit and modular image/batch budgets | Consistent enforcement across entry points; bounded retained image inputs; keeps inference contracts | More settings; oversized inputs are rejected; native/model/parser allocations are not a hard RSS limit | Adopt |
| 3. Process-isolated decoders and strict OS memory quotas | Stronger containment of native codecs and stuck work | IPC, copying, worker restart, and deployment complexity | Defer until workload evidence warrants it |

Option 2 preserves successful response schemas and model math. The body limit runs before JSON retention; inline image/batch checks run before admission. Worker checks protect remote sources and direct embedder calls. Budget exhaustion for a known inline batch rejects the request; image failures discovered during processing remain ordered per-item batch errors. Rejected items do not reach the processor or cache. Runtime images remain non-root and remote URLs remain disabled by default.

Keep the body limiter inside SlowAPI's BaseHTTP wrapper. Local raw-ASGI tests found that putting the limiter outside it groups the receive exception and FastAPI maps that to HTTP 400; the selected ordering preserves 413 for missing, invalid, understated, and negative Content-Length as well as streamed bodies. Rate limiting can reject an upload before reading any body. The framework may use plain text for its header-based body refusal; image-limit refusals use the existing JSON detail structure.

## Selected contracts

| Setting | Default | Boundary |
|---|---|---|
| `MAX_REQUEST_BODY_BYTES` | 16 MiB | Raw HTTP bytes, including JSON/base64 overhead |
| `MAX_IMAGE_BYTES` | 10 MiB | One compressed/decoded-base64 image |
| `MAX_BATCH_IMAGE_BYTES` | 32 MiB | Aggregate compressed image bytes resolved by one batch, including cache hits |
| `MAX_IMAGE_PIXELS` | 16,000,000 | Source pixels and projected CLIP resize pixels, independently |
| `MAX_BATCH_IMAGE_PIXELS` | 32,000,000 | Sum of source and projected resize pixels for retained uncached batch images |

Limits are positive integers and cannot be silently disabled. Exact ceilings are accepted. Raise the body ceiling deliberately for large inline batches; default body overhead permits a single image near the existing 10 MiB image limit, not 32 maximum-sized inline images.

Small services own inline checks, per-batch accounting, and Pillow/base64 loading. Each batch checks bytes before retaining another payload and pixels before RGB conversion. A remote image may temporarily occupy up to one `MAX_IMAGE_BYTES` scratch allocation before aggregate checking; retained compressed payloads remain within the batch ceiling. Decoded images close on success and failure. Existing inference owners retain queue capacity throughout timed-out work, so HTTP expiry cannot start an extra worker over its retained images.

These are input-work limits, not exact process-memory quotas. JSON object overhead, codec metadata, tensor copies, model weights, cache entries, native libraries, multiple processes, and concurrent body parsing require deployment sizing and server/proxy concurrency limits. Settings are application-local; there is no module-global budget or warning-policy mutation. The platform remains Python; any future JavaScript must use ES Modules.

## Validation and outcome

The complete Python 3.12 CPU suite passed 315 tests with pytest-cov 7.1.0. Line coverage is 94.74% and branch coverage 88.48%, above the unchanged 89.42%/79.37% ratchet. The input accounting module and inference executor each have 100% line/branch coverage. One upstream Starlette/httpx TestClient deprecation warning remains visible.

Regressions cover exact/oversized headers, absent/invalid/negative/understated headers, streamed bodies, rejection before admission, base64 refusal before decoder invocation, source/resize refusal before RGB conversion, extreme aspect ratios, aggregate exhaustion with later smaller items, cached/no-cache paths, and configuration rejection of nonpositive/noninteger ceilings. Event-controlled tests verify that a timed-out worker owns its live image until work finishes and then closes it. Coalesced requests retain their individual 200/413 mapping. A real tiny CLIP model compares single and batch service embeddings with equivalent direct native calls; each uses the corresponding batch shape to account for float32 kernel differences.

The rebuilt CPU runtime passed its shipped non-root/offline native projection, writable-cache, authenticated startup, and graceful-shutdown smoke. A separate live Uvicorn probe, with a 1,024-byte body ceiling, returned 413 for both declared and chunked 1,500-byte uploads, 400 for a smaller undecodable image, and 200 for authenticated model discovery. Lifespan shutdown completed, including Uvicorn's expected SIGTERM re-raise. The Starlette 1.6.0 minimum passed 28 input-limit and integration tests; the current installed runtime uses 1.7.0. This iteration rebuilds CPU; the prior five-profile validation remains documented in the [backend record](backend-build-recommendation.md).

Strict OSV audits of the actual CPU runtime (56 packages) and pytest-cov test environment (63 packages) found zero known vulnerabilities and zero skipped packages. Audit tools ran in isolated temporary storage without changing the shipped image. Ruff checks/formatting for new modules and tests, scoped Pyright checks of all seven changed runtime modules against installed library types, workflow lint, copyright, compilation, and whitespace checks passed. The rate-limit handler adapter preserves the existing SlowAPI response while satisfying Starlette's generic exception-handler type contract.

## Next recommendation

The subsequent [remote destination iteration](remote-image-destinations.md) validates redirects and pins connections to approved public addresses; its separate record contains reproduction and verification evidence. Remote access remains disabled by default. The [aggregate ingress iteration](aggregate-ingress.md) now bounds concurrent HTTP retention and adds deployment memory/worker controls. Next add production-model/cache fixtures and complete dependency locks. This document's 315-test results describe the earlier input-budget iteration; current results and priorities are in the [recommendation stack](recommendation-stack.md). No release or upstream PR merge is created.
