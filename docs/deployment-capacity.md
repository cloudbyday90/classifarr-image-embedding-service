# Concurrent input and accelerator capacity

Assessment date: 2026-10-03. Baseline: `f45ee654295f313e428063707dd049bda4c79d99`. This extends the [earlier CPU/OpenVINO calibration](capacity-calibration.md); it does not change production owner counts.

## Official research and alternatives

Sources were discovered with web search and GitHub MCP, then opened. The [kernel v2 controller](https://docs.kernel.org/admin-guide/cgroup-v2.html) distinguishes current/peak task counts and limit-hit events. The [v1 PID controller](https://docs.kernel.org/6.0/admin-guide/cgroup-v1/pids.html) independently mounts task accounting and documents hierarchical limits. Thread creation consumes task capacity, so parent process counts alone are insufficient. [Python filesystem observations](https://docs.python.org/3.11/library/os.html#os.statvfs) provide block and inode availability. Filesystem occupancy can capture anonymous worker output that a directory walk misses. [Docker tmpfs guidance](https://docs.docker.com/engine/storage/tmpfs) documents temporary storage and possible swap persistence; tmpfs is part of the memory budget, not extra RAM.

[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) distinguish live tensor allocations from allocator reservations. [Allocator peak documentation](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_reserved.html) explains peak resets and conservative async-allocator accounting. [Synchronization](https://docs.pytorch.org/docs/main/generated/torch.cuda.synchronize.html) waits for outstanding device work. Device-wide free memory includes other applications; it must not be presented as this service's allocation.

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| Infer CUDA/PID/storage capacity from CPU RSS | No probe changes | Different resources and ownership; misses anonymous files and GPU reservations | Reject |
| Extend separate bounded offline probes | Real models, protected API, child lifecycle and independent counters; no request-path sampling overhead | Sampling can miss short peaks; fixture/hardware-specific evidence | Adopt |
| Raise concurrency or impose tight production PID/tmpfs limits immediately | More parallel work or stronger containment | Unmeasured native thread/export costs can break supported backends | Defer pending target measurements |

## Design

Keep memory reporting separate from task/storage and accelerator readers. Resolve the actual PID controller membership on v1/v2; absent counters remain unknown. Record parent threads, cgroup tasks/limit/events, filesystem used/available bytes and available inodes. Aggregate sampled maxima and minima without retaining unbounded samples. Report kernel lifetime peaks separately.

CUDA probes require available CUDA and verify both model devices; CPU fallback fails. Synchronize at phase boundaries and reset allocator peaks only in this dedicated experiment. Record live/reserved/peak allocator bytes and device-wide free/total bytes. Keep resident models, current numerical checks, cache disabled and one native owner.

The mixed scenario uses authenticated in-process ASGI requests with two remote images and remaining inline images, followed by a concurrent maximum inline batch. A controlled loopback HTTP fixture serves deterministic PNGs padded to bounded byte pressure. Only the probe's isolated child command maps the fixed fixture authority to loopback; production URL policy and worker launch remain unchanged. The actual parent subprocess, protocol, total/shared deadline, anonymous output and kill/reap paths execute. Cancel a second mixed caller after remote dispatch and verify its retained owner blocks a live waiter until cleanup. Validate ordered vectors against the same owner's references and record settled queue state and child closure. This covers application admission/input ownership, not socket/proxy buffering.

Run fresh non-root containers with offline verified model assets, dropped capabilities, no new privileges, finite memory/no swap, a generous experimental PID bound and dedicated temporary filesystem. These experiment ceilings are not production sizing recommendations. Extend manual CPU/OpenVINO calibration; CUDA requires operator-selected GPU hardware.

## Outcome and recommendations

Four offline production-weight experiments passed using the approved catalog revisions and complete locked native environments. Both models remained resident; batches 1/8/32 and 974-square source PNGs validated against same-owner single-image vectors at `rtol=atol=1e-4`. The mixed API workload sent two 8-MiB padded remote PNGs plus 30 inline images and queued a maximum inline batch behind it. A second mixed caller canceled after child dispatch; its actual native result still passed parity checks. Every mixed experiment validated four native batches, reaped all four children with closed stdin and settled with zero owners/waiters/readers. Unauthenticated requests returned 401.

| Experiment | Peak RSS GiB | Cgroup lifetime peak GiB | Sampled peak tasks / parent threads | Peak tmpfs used MiB | Successful concurrent pair seconds |
|---|---:|---:|---:|---:|---:|
| CPU mixed + canceled remote owner | 2.55 | 1.14 | 42 / 42 | 8.004 | 45.05 |
| OpenVINO CPU mixed + canceled remote owner | 4.42 | 4.60 | 26 / 25 | 8.004 | 23.69 |
| NVIDIA CUDA mixed + canceled remote owner | 1.94 | 1.45 | 25 / 25 | 8.004 | 3.47 |
| OpenVINO private cold exports + maximum batches | 6.34 | 4.86 | 58 / 58 | 0 | Not an API pair |

CUDA executed on an RTX 5070 Ti with driver 616.92, Torch 2.14.1+cu130 and the native allocator. Maximum allocator phase peaks were **2.04 GiB allocated / 2.37 GiB reserved**. Sampled CUDA allocation was lower than the allocator peak, demonstrating why sampling alone cannot establish VRAM capacity. Device-wide free memory is archived separately and includes unrelated applications. Torch used eight intra-op and 16 inter-op threads; each experiment used one worker/owner and the 45-second embedding deadline. CPU and CUDA containers had 4 GiB/no swap; OpenVINO had 8 GiB/no swap. All used 1024-task and 128-MiB `noexec,nosuid,nodev` tmpfs experiment bounds, and no memory/PID limit-hit or OOM counters increased. Anonymous output occupied 8 MiB plus one filesystem block, then temporary occupancy returned to baseline after reaping. Private cold IR was removed on exit while original entries remained available.

The [measurement archive](validation/deployment-capacity-2026-10-03.json) records immutable image identities, configurations, source/journal hashes, per-phase aggregates, final accounting and validation evidence. These are native **source-overlay** runs using existing locked images; production image rebuilds and hosted workflow execution are not claimed. CPU measurement preceded a type-only reader annotation correction; its immutable source snapshot is recorded separately. CUDA/OpenVINO used the delivered executable probe code. Shared-host CPU calibration overlapped CodeQL work; OpenVINO/CUDA runs overlapped routine QA. These are observations, not paired backend speed rankings. The CPU pair approached 45 seconds; measure individual client/queue/deadline headroom under target load before operating near the maximum batch ceiling.

Measurements exercised native cgroup v1; v2 and independently mounted v1 PID controllers have deterministic tests. This kernel exposes no PID lifetime peak, so sampled task peaks remain lower bounds. Parent thread counts include probe sampling/fixture overhead. Temporary storage metrics cover the dedicated `/tmp` filesystem, including anonymous remote output; model/IR volume growth is a separate storage budget. Cold IR lives on that volume. Shared page-cache charging can make cgroup peaks lower than RSS. Padded PNGs do not cover every codec/byte/pixel combination, and in-process ASGI does not measure socket/proxy buffering. Intel GPU and CUDA ARM remain separate gates.

Keep Python and modular service/probe files, one worker/owner, existing ingress/byte/pixel and upload/remote budgets, 4 GiB CPU/CUDA and 8 GiB OpenVINO containment, with operator overrides based on target measurements. The 1024-task/128-MiB bounds have substantial headroom for these experiments but are not a new production default. Preserve exact dependency/model/IR contracts and monitor temporary capacity independently of persistent cache storage. Next add recurring native Windows setup, fetch-worker and response-transport validation.

The complete locked Python 3.12 QA suite passes **928 tests / 7 existing optional skips**, with **95.22% lines / 90.12% branches** coverage above unchanged floors. Fourteen new cases cover split controller discovery, missing counters, reserved filesystem capacity/inodes, anonymous-file allocation, aggregate minima/peaks, unavailable CUDA, synchronized allocator boundaries, mixed native children and hidden detached failures. Scoped Ruff/Pyright, workflow syntax, copyright, Markdown links and both skill validators pass. Native CodeQL reports only the same two existing setup-context alerts (explicit key display and nonsecret 0644 publication); no probe finding or suppression was added. Configured native Gitleaks source/history scans find no secrets. The final QA filesystem explicitly permits the setup test's executable local stub; earlier tmpfs execution-denied failures were test-container configuration errors.

## Real socket follow-up

The [separate socket design and outcome](socket-upload-capacity.md) now extends
these historical ASGI measurements with native CPU/OpenVINO HTTP/1 uploads at
configured body/image ceilings, server receive counters, disconnect settlement
and real vector recovery. Keep the original observations above intact; direct
socket evidence still leaves the operator's proxy buffering and target hardware
as separate gates. The [authentication follow-up](early-api-key-authentication.md) now rejects before body ingestion and [quota identities](public-probe-rate-limits.md) are stable across unverified headers. Next measure actual operator proxy buffering, effective-address trust and target workload headroom.

The operator subsequently confirmed direct Docker port publishing with no reverse
proxy. The [direct deployment follow-up](direct-deployment-trust.md) makes address
trust explicit, validates the published port and refreshes bounded socket
headroom. No proxy is added; actual workload and remote-client address behavior
remain deployment-specific measurements.
