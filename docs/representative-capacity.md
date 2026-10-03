# Representative poster and codec calibration

Assessment: 2026-10-03. Baseline: `b43272ba9f380f7488f1a70e5dafa70a91754b04`.
The operator selected poster-style portrait inputs, landscape inputs and
PNG/JPEG/WebP cases. Direct Docker deployment and Python remain unchanged.

## Research and alternatives

Official URLs were discovered and fetched through web services.
[Pillow's format guidance](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html?highlight=gif)
describes codec-specific encoding controls; decode the encoded fixture rather
than treating a lossy source tensor as its reference. Record exact encoded
hashes and installed codec versions because compression changes bytes.
[Linux process accounting](https://docs.kernel.org/filesystems/proc.html?highlight=memavailable)
distinguishes current RSS from process high-water RSS. The
[cgroup memory controller](https://docs.kernel.org/admin-guide/cgroup-v2.html?highlight=bpf_cgroup_device)
also charges page cache, kernel and socket memory; separately sampled process
peaks cannot be subtracted to obtain service-only cgroup memory.
[Python subprocess guidance](https://docs.python.org/3.12/library/asyncio-subprocess.html)
warns that captured output can buffer in memory. A stream buffer setting alone
does not bound total retained output or guarantee child cleanup.
[Docker's volume guidance](https://docs.docker.com/engine/storage/volumes/)
supports mounting an existing volume subdirectory separately. The local
OpenVINO experiment keeps publisher assets read-only while mounting only the
application-owned `ov_ir` subdirectory writable for its persistent process locks.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Reuse only padded square ceilings | Retains reproducible overload evidence | Omits poster shapes, codecs and normal body sizes; HTTP client shares model RSS | Keep as separate stress scenario |
| Add bounded representative fixtures and a separate HTTP client process | Exercises selected shapes/codecs and native endpoints; reports client RSS independently | Synthetic content; same cgroup still includes both processes and sampling/reference overhead | Adopt |
| New external benchmark platform or production traffic replay | Can isolate complete service cgroup and observe real distributions | Additional infrastructure and operator data/access; real traffic can contain private images | Defer until operator evidence is available |

## Design and recommendation stack

1. Extend `capacity_probe.py` with a representative scenario, reusing its pinned
   models, memory/resource readers, direct server and flushed journals.
2. Use deterministic poster JPEG and RGBA PNG, landscape WebP, and a small
   high-entropy PNG control. Keep explicit byte/pixel/body and concurrency bounds;
   compute native single-image references from each encoded payload.
3. Generate and send requests in a fresh isolated HTTP client interpreter. Pass a
   disposable key through bounded stdin, never command arguments or journals.
   Fix transport to loopback; use bounded responses, a finite child lifetime and
   kill/reap cleanup on timeout, failure or cancellation.
4. Validate ordered vectors in normalized and unnormalized requests for 1/internal/API batch
   sizes, both resident models and two concurrent callers by default. Record
   per-case durations, request sizes, observed server bytes, separate client RSS
   and settled queue/ingress ownership. These are finite burst observations.
5. Retain one worker, current limits, hashes and coverage floors. Use target
   workload evidence before changing concurrency or containment.

This scenario adds no proxy, application API behavior, JavaScript or dependencies.
Poster dimensions are reproducible synthetic fixture choices, not a claim about
the operator's production distribution. Whole-container accounting retains the
client; model-owner RSS still includes references and sampler overhead. Real
remote-client address preservation and hosted/accelerator gates remain separate.

The selected fixture dimensions are 500×750 for JPEG/RGBA PNG posters, 750×500
for landscape WebP and 224×224 for the high-entropy PNG control. Each has two
different encoded variants. The API's normalization flag applies to the whole
request; callers alternate normalized/unnormalized requests, preserving the
existing endpoint contract. A calibration-only `1000/minute` embedding quota
permits the bounded experiment; the shipped `30/minute` quota stays unchanged.
The client has a 300-second total lifetime, at most 1 MiB input/256 KiB report,
2 MiB per response, 1–8 callers within HTTP admission and 1–2 repeats. Fixture
bytes, source/resize pixels and whole bodies are checked before model ownership.
Subprocess launch and kill/reap cleanup can add OS latency beyond the experiment
deadline; ownership is retained until cleanup completes. The fixed child creates
no descendants. HTTP refusals or vector mismatches retain completed/canceled case
records and observed statuses/bytes, emit settled ownership and sampled/kernel
memory evidence, then fail the CLI. Hard child timeout/output/launch failures
retain phase error and memory evidence; a final client report may be unavailable.
Refusal or vector mismatch never counts as successful throughput.
`--batch-sizes` permits only a subset of the configured 1/internal/API ceilings
in this scenario, supporting comparisons without changing application limits.

## Outcome

The two-caller CPU full-ceiling experiment failed as intended when its second
32-image ViT-L-14 poster JPEG request returned 504 at 45.08 seconds. The first
32-image request passed in 29.22 seconds; both bodies were observed in full.
Five requests passed before the failure, the client was reaped and all ingress,
waiting, native and read-lock owners settled. There were no OOM, memory-limit or
task-limit events. The original pre-diagnostic failure is also retained locally.

One CPU caller passed all 24 profile/model/batch combinations at the default
45-second deadline. Its slowest successful request took 23.14 seconds. CPU
containment remained 4 GiB with no swap. These separate runs had different host
load/cache conditions and do not establish a paired performance improvement.
Two CPU callers with `--batch-sizes 1 8` passed all 32 cases; the slowest request
took 16.24 seconds. This gives a bounded caller/batch option under the current
deadline on this host. It does not change the application's maximum of 32.
OpenVINO CPU, with both cached FP32 models resident and 8 GiB/no-swap containment,
passed all 48 two-caller cases at 1/8/32. Its slowest request took 39.25 seconds;
that leaves little deadline margin on a busier host. Cached XML/BIN/manifests were
verified against both current contracts before and after the run and stayed
unchanged. The initial fully read-only cache setup failed on its persistent lock;
the corrected retry grants write access only to the existing `ov_ir` subdirectory.

| Experiment | HTTP/vector outcome | Model-owner peak RSS | Whole-cgroup kernel peak | Client peak RSS | Sampled tasks |
|---|---|---:|---:|---:|---:|
| CPU, two callers, 1/8/32 | Five pass, second 32-image poster request 504 | 2.50 GiB | 1.04 GiB | 100.75 MiB | 51 |
| CPU, one caller, 1/8/32 | 24/24 pass | 2.52 GiB | 1.07 GiB | 100.83 MiB | 59 |
| CPU, two callers, 1/8 | 32/32 pass | 2.13 GiB | 0.68 GiB | 100.75 MiB | 59 |
| OpenVINO CPU, two callers, 1/8/32 | 48/48 pass | 4.16 GiB | 2.68 GiB | 100.71 MiB | 41 |

All four experiments use pinned offline models, one worker/owner, zero cache and
batch window, a 45-second embedding budget, 1024 PID containment and 128 MiB
tmpfs. Successful cases require ordered finite vectors within `rtol=atol=1e-4`,
matching encoded hashes and exact server-observed body sizes. Every experiment
reaps its HTTP client and settles ingress, waiting, native and read-lock owners;
header-only unauthenticated requests return 401 with zero application body bytes.
There are no memory/OOM/task-limit events or observed temporary-file allocations.

Memory is from warm/cache-specific containers on this Docker host, with other
projects present. Shared pages can be charged outside a later experiment, making
cgroup peaks lower than process RSS. These readings do not establish dedicated
production memory requirements; retain existing 4 GiB CPU/8 GiB OpenVINO limits
and earlier cold-export evidence. Sampled task peaks are lower bounds; this v1
PID controller provides no lifetime peak. The approximately 101 MiB client peak
belongs to a separate interpreter and must not be subtracted from independent
kernel/sample peaks.

The full QA container needs an explicit executable temporary mount for its
existing launcher stub. Two local setup attempts retained Docker's `noexec`
tmpfs and failed that fixture alone; a native mount/exec diagnostic and focused
launcher replay confirmed the correction. Calibration retains `noexec` tmpfs.
Docker documents the explicit mount flags in its
[tmpfs guidance](https://docs.docker.com/engine/storage/tmpfs).

Full locked QA passes **1,112 tests**, with seven existing platform/optional skips,
including 26 new fixture, allocation, private IPC, output/response, cancellation,
receipt, metadata/vector and deadline regressions. Coverage remains **95.38%
lines / 90.42% branches**, above unchanged **89.42% / 79.37%** floors. The canonical
OpenAPI byte hash is unchanged. Exact QA/CPU/OpenVINO inventories contain
70/56/58 reviewed wheels, strict pip checks pass and all 13 profiles/26 lock
artifacts remain unchanged. Ruff, scoped Pyright, workflow syntax, skill metadata,
copyright and relative documentation-link checks pass.

The [native validation archive](validation/representative-capacity-2026-10-03.json)
records image identities, configurations, source/journal hashes, resource scopes,
all case results and limitations. Early CPU diagnostic runs used CRLF source;
the three differences from final LF probe files are verified as line-ending-only
changes. The passing CPU smaller-batch and OpenVINO experiments, full QA and
static validation use final source bytes. Every changed committed file matches
that final QA snapshot. Twelve untouched baseline files, including six Python
files, retain checkout CRLF versus committed LF differences; these are verified
as line-ending-only, and the archive separates QA and committed source hashes.

CodeQL 2.27.1 with Python security-extended queries 1.8.11 scans all 90 Python
source files and reports no new alerts. Its two unchanged baseline alerts concern
the explicit `--show-key` output in `generate_env.py` and readable nonsecret
configuration publication in `secret_publication.py`. No suppression changes
were made. Workflow extraction does not establish an Actions security-query
result. Fresh hash-verified Gitleaks 8.30.1 scans find no secrets in the final
source snapshot or 90-commit baseline history. Two initial heuristic matches
were public OpenVINO contract digests; representing them explicitly as archived
SHA-256 fields resolves those matches without allowlist changes. These static
checks cover the reviewed source/history, not every deployment condition.

No suitable unapplied PR was available. The subsequent requested
[aging-PR cleanup](aging-pr-cleanup.md) closes all eight reviewed proposals
without merging, and a fresh GitHub MCP collection returns zero open PRs.

## Next recommendation

Investigate CPU queue-wait versus native execution time for concurrent maximum
batches, then replay the measured caller/batch profiles on the actual deployment
host before changing deadlines, owners or admission. Prefer one caller up to 32
or two callers up to 8 as the current locally tested CPU options. A larger HTTP
deadline would retain ingress longer and needs target-host evidence; additional
workers would duplicate model ownership. Preserve direct Docker topology and
current containment. Observe external-client address sharing separately.

| Operating choice | Pros | Cons | Recommendation |
|---|---|---|---|
| CPU, one caller up to 32 | All selected profiles pass with current deadline | Coordinates callers; reduces concurrent request submission | Locally validated starting option |
| CPU, two callers up to 8 | All selected profiles pass with more deadline margin | More requests to process a large library | Locally validated concurrent option |
| Longer deadline or additional model owners | Could accommodate more queued work | Longer ingress retention or duplicated native memory; benefit unmeasured | Defer pending queue/native and target-host evidence |
| OpenVINO CPU, two callers up to 32 | Complete selected workload passes | Slowest request already takes 39.25 seconds; larger containment | Retain 8 GiB and replay on target hardware |
