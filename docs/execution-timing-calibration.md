# Queue and native-call timing calibration

Assessment: 2026-10-15. Baseline: `e5b55d03ac61ce36c16bf611bc5017c8c01c4955`.

## Research and design

The previous representative CPU two-caller maximum-batch experiment returned
504 for a second 32-image poster request. Its HTTP duration could not distinguish
admission, model-read locking, worker dispatch and the synchronous model call.

Official URLs were discovered and fetched through web services.
[Python's clock documentation](https://docs.python.org/3.12/library/time.html)
supports nanosecond performance-counter differences for elapsed measurements;
process and thread CPU clocks measure different scopes. Use one server clock
for causality, with relative offsets rather than combining client/server epochs.
[AnyIO's worker API](https://anyio.readthedocs.io/en/stable/api.html)
distinguishes thread completion from cancellation of its waiter. Keep the existing
owner until it finishes, including work surviving an HTTP deadline.
[PyTorch's profiler guidance](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
offers operator-level traces but describes additional profiling overhead.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Increase deadline from HTTP totals | Small configuration change | Does not identify the bottleneck; retains ingress longer | Defer |
| Full operator profiler for every request | Detailed kernel/operator costs | Extra overhead and backend-specific tracing; does not replace admission evidence | Reserve for a demonstrated native bottleneck |
| Optional owner-phase observer and bounded probe timelines | Separates admission, locks, worker wait and native-call wall time; follows detached owners | Adds trusted internal hook; native call still includes preprocessing and serialization | Adopt |

The application accepts an optional internal execution observer; it is absent
by default and has no HTTP setting or public trace endpoint. Trusted observers
must return promptly and be thread-safe. Ordinary observer failures cannot
replace results/errors or prevent permit cleanup; exception text and callable
arguments/results are never passed into observations or diagnostic logs.

The representative probe binds only its finite expected request labels to
per-experiment contexts. Each owner retains its own trace across task/thread
handoffs and HTTP expiry. Admission, read-lock, dispatch, native completion,
waiter detachment and terminal owner events remain separate. Probe storage is
bounded by the existing two-model/four-fixture/three-size/eight-caller/two-repeat
ceilings. Unknown labels create no timeline. Missing stages remain unknown,
and incomplete owners or successful responses lacking native evidence fail
calibration. A 504 never becomes successful throughput.

Keep Python, direct Docker publishing, one owner, current admission, the
45-second embedding deadline and existing byte/pixel/memory containment.
Native-call durations include decoding, preprocessing, model execution and
result construction plus probe result verification; they are neither kernel-only
timings nor thread CPU time.
Existing client RSS versus whole-cgroup limitations remain in force.

A small probe facade forwards to the existing embedder owner and validates every
returned native batch against encoded-byte references before returning. Its
bounded timeline records only a verification boolean. This closes a discovered
diagnostic gap: client-only validation cannot check a result after its HTTP
waiter expires. Received HTTP vectors are also checked independently. Unknown
unstarted work remains unknown, and neither observer nor journal receives raw
vectors, image payloads or keys. Numeric mismatch diagnostics use a fixed error
message rather than dumping vectors into server logs.

## Validation and outcome

Full locked Linux QA passes **1,132 tests**, with seven existing skips, in
124.227 seconds. The 20 new cases cover observer failures, separate contexts,
admission cancellation, read-lock and worker-limiter waits, detached success and
failure, missing/out-of-order evidence and private native-result verification.
The client independently rejects corrupted HTTP vectors. Application coverage
is **95.46% lines / 90.56% branches**, above unchanged **89.42% / 79.37%** floors;
both changed execution modules have complete line/branch coverage.

The final real-model experiments use both immutable publisher models, four
portrait/landscape/codec fixtures, alternating raw/normalized results and separate
HTTP clients. CPU retains 4 GiB/no swap; OpenVINO retains 8 GiB/no swap. Each has
one owner, Torch intra/inter-op settings 8/16, ingress eight, queue 100 with a
60-second admission limit, a 45-second embedding deadline and no result cache
or collection window. The calibration quota is 1000/minute only inside the probe.

| Final profile | Outcome | Slowest client observation | Native evidence |
|---|---|---|---|
| CPU, four callers × 32 images | Failed in the first ViT-L-14 JPEG burst: one 200, one observed 504, two cancelled client siblings whose server status is 504 | 45.059 s | Two native calls verified; two owners cancelled before native start |
| CPU, four callers × eight images | 32/32 passed | 32.863 s | All 32 native results verified |
| OpenVINO CPU, two callers × 32 images | 16/16 passed | 41.808 s | All 16 native results verified |

For the failed CPU request, admission consumes **28.784 seconds**, the native
call **31.167 seconds**, and the owner **59.959 seconds**. Native work continues
**14.957 seconds after response start**, then passes private vector validation.
Two other requests expire in admission and never dispatch native work; their
unknown native duration is retained. Every owner/read lock/queue/ingress settles,
the client is reaped, and no OOM/task-limit event or remaining temporary bytes
are observed. The failure remains a failed capacity result.

An earlier timing-only two-caller CPU run passes 16/16, with a 30.466-second
slowest response: its second JPEG owner spends 15.531 seconds in admission and
14.883 seconds in native work. Earlier four-caller runs are also preserved in
the archive. Those runs predate private native-result verification and cannot
prove parity for a detached result. Shared-host variability, warmer page-cache
charging and the added verification make these observations unsuitable for a
paired performance claim. They establish neither sustained throughput nor a
production-host percentile. Whole-cgroup memory includes the HTTP child and
shared-page accounting can fall below process RSS; do not subtract independent
peaks or replace the earlier cold-memory sizing evidence.

Initial QA on a Windows Docker file share misses an unchanged 1.2-second remote
child startup/redirect assertion; a focused replay also misses redirect/TLS
receipt. A byte-verified copy on a native Linux volume passes all four phases,
then the full suite passes without changing assertions, budgets or filters.
[Docker documents empty-volume population and `volume-nocopy`](https://docs.docker.com/engine/storage/volumes/);
the frozen source volume uses that option to prevent an image's older checkout
from being copied implicitly. The original failures remain in the evidence.

Native CodeQL security-extended analysis indexes all 93 Python files and retains
two existing setup-helper alerts, with no new alerts or suppressions. Workflow
files are indexed but this run executes Python queries only. Ruff, scoped
Pyright, copyright, skill metadata, coverage, secret and whitespace checks pass.
The [PR 57 record](pr-57-local-validation.md) covers the actual six-profile
Transformers migration, strict audits, model parity and distinct IR identities.
The [validation archive](validation/execution-timing-2026-10-15.json) contains
phase records, exact source/artifact identities, configurations, setup failures,
resources and both initial/final outcomes without vectors, images or keys.

## Recommendation stack

1. Keep Python, direct Docker publishing, one owner and existing containment and
   deadlines. Replay smaller caller/batch profiles on the deployment host before
   choosing operating pressure; extra smaller requests also consume more quota.
2. Next investigate **cooperative cancellation between native micro-batches**:
   stop undispatched chunks once their request has no live waiter, while keeping
   the active native call, read lock, images and capacity owned until settlement.
   Coalesced batches require live-member accounting; one cancelled member must
   not discard another's result. This is a design/testing task, not permission
   to kill threads or release a running owner. The observed detached tail makes
   this more valuable than raising deadlines blindly.
3. Use a scoped native profiler only if finer preprocessing/operator attribution
   will change that decision. Larger deadlines retain ingress longer; extra
   owners duplicate resident models and require measured memory headroom.
   Physical ARM/GPU capacity, external client-address preservation and hosted
   platform evidence remain separate gates. No reverse proxy is required.

## October 16 follow-up

The recommended [cooperative inference change](cooperative-inference-cancellation.md)
now introduces bounded forward chunks and last-waiter checkpoints. The October
15 timings above describe the previous whole-batch path, not the new default.
Incomplete cooperative stops retain unknown full-batch parity; the separate
cooperative probe validates the committed chunk and queued recovery privately.
