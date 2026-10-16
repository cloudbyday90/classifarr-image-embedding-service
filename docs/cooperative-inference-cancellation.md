# Cooperative inference cancellation

Assessment: 2026-10-16. Baseline: `c38092c926c894c3b7da819b7f83dbf75000df9b`.

## Research and design

The previous CPU experiment retained 14.957 seconds of native work after its
HTTP 504 began. Existing execution ownership correctly retains permits and read
locks, but a batch currently executes in one model call. Add cooperative
checkpoints and bounded native chunks without abandoning active native work.

Official URLs were discovered through web MCP search and fetched on October 16.
[Python asyncio](https://docs.python.org/3.12/library/asyncio-task.html) explains
that shielding preserves an owner task while its waiter receives cancellation.
[Python threading](https://docs.python.org/3.12/library/threading.html) provides
thread-safe events and does not support forcibly stopping threads.
[AnyIO threads](https://anyio.readthedocs.io/en/stable/threads.html?highlight=BlockingPortal)
supports copied context and voluntary thread checks. Its scope-based cancellation
check alone cannot detect cancellation of this service's separately shielded
waiter; that is a project-specific ownership distinction.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Cancel or abandon the active thread | Prompt waiter return | Does not stop native work; risks releasing owned resources | Reject |
| Separate inference processes | Hard-stop capability | Model duplication, IPC and recovery costs | Defer |
| Per-owner thread-safe stop event and bounded forward chunks | Skips future work; preserves active-call ownership and existing models | Extra forward calls; one active chunk can still overrun; needs backend parity and coalesced-member evidence | Adopt |

Use a context-bound event per execution owner, never a mutable shared embedder
flag. Set it when the last relevant waiter leaves. Check outside per-item error
handlers before input/decode/model preparation and before each forward chunk.
An active chunk completes and validates normally; a later checkpoint skips
remaining work. The existing executor retains admission and its read lock
through all cleanup, including decoded-image closure and supervised child reaping.

Coalesced groups already cancel their executor submission only after their last
member leaves. Preserve that rule; a cancelled member cannot stop a live sibling.
Use a separate positive `EMBED_INFERENCE_BATCH_SIZE`/`model.inference_batch_size`
setting, initially eight, independent of API item ceilings and collection size.
Preserve ordered partial errors, cache identity, aggregate byte/pixel budgets,
normalization, model revisions and native backend selection. Cancelled incomplete
batches publish no partial batch cache entries; completed calls retain existing
cache behavior. Do not change HTTP schemas, deadlines, admission or memory limits.

Distinguish cooperative cancellation from native failure in optional observations.
Cancelled incomplete results have unknown full-batch vector verification, not
successful throughput. Actual native errors remain failures. A real cancellation
probe must validate the active chunk privately and prove no subsequent chunk
starts, rather than demand a result for deliberately skipped work.

## Validation and outcome

Full locked Linux QA passes **1,162 tests**, with seven existing skips, in
124.476 seconds. The **30 new cases** cover active/final chunks, input/decode/
model/processor checkpoints, a saturated worker limiter, independent owners,
coalesced groups with one versus zero live callers, ordered invalid vectors,
cache recovery and incomplete observation evidence. Negative controls fail when
checkpoints are removed or a committed detached chunk is corrupted. An active
native error remains an error even after a stop request.

Coverage is **95.79% lines / 91.08% branches**, above unchanged **89.42% / 79.37%**
floors. Execution, cancellation, observation and batch-forward modules each have
complete line/branch coverage. Baseline/current source produce byte-identical
OpenAPI JSON under the same final graph and settings. Native CodeQL indexes all
96 Python files and retains only the two existing contextual setup alerts;
Python security-extended queries run, not Actions queries. Scoped Pyright checks
all 12 affected runtime/probe files with zero errors/warnings using actual
current framework/OpenVINO definitions. Ruff, copyright, skill metadata, relative
links, secret and whitespace checks pass without new exclusions or suppressions.

Both immutable publisher models pass the controlled cooperative scenario on CPU,
OpenVINO CPU and actual CUDA 13 execution on the **RTX 5070 Ti**. For each model,
one committed eight-image chunk completes and matches private references at
1e-4, **24 future native images are skipped**, and a queued one-image recovery
also passes. Capacity/readers remain one during the barrier, recovery waits, and
both settle to zero. The probe observes closure of nine forwarded/recovery
images per model; deterministic regressions additionally check closure of all
prepared images. Full cancelled-batch parity remains unknown. These controlled
barriers do not establish kernel interruption or a hard cancellation bound.

Default eight-image chunks pass single-reference and batch 1/8/32 raw/normalized
parity on all three backends. Real direct-loopback HTTP/1 poster/landscape
JPEG/PNG/WebP profiles pass **8/8 cases each** on CPU and OpenVINO, one caller
with 32-image requests. Their slowest observations are **33.609 seconds** and
**17.134 seconds**, respectively, under the unchanged 45-second deadline. Every
complete native result is privately verified and every HTTP client is reaped.

| CPU serial comparison, same reviewed graph | Native cap 32 | Native cap eight |
|---|---|---|
| ViT-L-14, 32-image batch | 20.501 s | 42.514 s |
| ViT-B-16, 32-image batch | 4.827 s | 12.326 s |
| Whole-process observed peak RSS | 2.38 GiB | 1.99 GiB |

These are sequential finite samples on a shared host, with initialization,
thread scheduling and warm-cache effects; they are not a controlled causal
speedup comparison. They reveal a batching-efficiency cost that must remain
visible. Keep eight as the initial bounded-work default after the selected HTTP
profiles pass; compare 8/16/32 on the deployment host when choosing a different
tradeoff. Larger chunks reduce checkpoints and may improve full-batch efficiency
while retaining more work in an indivisible call. Whole-cgroup charging includes
clients and can fall below RSS because shared pages were charged elsewhere.
Do not subtract independent peaks or replace earlier cold-export sizing evidence.

CPU/OpenVINO mixed-input experiments explicitly use API batches of eight and two
8 MiB remote fixtures. Each has three fully verified batches, one cooperative
stop, **three reaped children**, retained permits/readers while the active child
settles, and valid queued recovery. The second remote input is never fetched
after cancellation. No OOM/task-limit events or residual temporary bytes are
observed. These are ASGI/controlled-child experiments, not external-network
capacity measurements. Probe intra/inter-op Torch settings are 8/16; production
host settings and broader concurrency require separate replay.

The [native archive](validation/cooperative-inference-2026-10-16.json) records
source/image identities, commands/log/journal hashes, full outcomes, graph/audit
proof and limitations. The [PR 58 record](pr-58-local-validation.md) covers the
independent floor-only adoption. The [skill record](project-cooperative-inference-skill.md)
documents instruction routing and evaluation. No release or version bump is
created; no original PR is merged.

## Recommendation stack

Keep Python, direct Docker, one owner and existing containment. Validate chunk
parity and cancellation before selecting chunk size from target-host pressure.
More frequent checkpoints reduce potentially wasted future work but can reduce
batch efficiency; they cannot interrupt a hung active native call. Preserve
independent hardware and external-client-address gates.

Next propagate **post-upload HTTP disconnects** into matched embedding waiter
cancellation through one bounded receive owner. Current embedding routes await
inference without monitoring that receive channel. The discovered
[ASGI HTTP specification](https://asgi.readthedocs.io/en/stable/specs/www.html)
defines a separate disconnect event; our source-based inference is that an early
socket close can otherwise leave stop signalling delayed until the existing
deadline. Preserve body consumption, response/ingress ownership and live shared
callers, and require native socket evidence before claiming this boundary fixed.
