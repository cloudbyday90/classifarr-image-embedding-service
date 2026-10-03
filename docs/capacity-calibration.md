# Production capacity calibration

Assessment date: 2026-10-02. Baseline: `15a39337b272ef4db11eccdf58f1116f7bbea2cb`. Python and delivery directly on master are user decisions.

## Evidence and official research

The default API allows 32 images and retains both catalog models. Previous production probes measured single/two-image work, with a 4.40 GiB large OpenVINO cold-export peak. Those observations cannot establish maximum-batch or multi-model capacity.

Official sources were discovered through web search and GitHub MCP. [Docker resource guidance](https://docs.docker.com/engine/containers/resource_constraints) recommends testing memory requirements and documents equal memory/combined-swap limits as preventing swap. The [kernel v1 memory controller](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html) distinguishes process RSS from controller accounting and describes limit-hit counters. The [kernel v2 controller](https://docs.kernel.org/6.12/admin-guide/cgroup-v2.html) exposes current/peak memory and hierarchical memory events. The [OpenVINO performance-hint guide](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html) explains that throughput modes can increase loading time and memory; its nightly status does not establish a tested change to this deployment.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Infer capacity from two-image RSS | No additional tooling | Misses activations, multiple models, page cache and detached owners | Reject |
| Separate bounded production workload and memory-reporting modules | Reproducible workloads; real loaders/queue; no request-path sampling overhead | Large weights and hardware time; observations remain workload-specific | Adopt |
| Add an exporter service or rewrite the runtime | Can isolate export retention | New protocol and ownership before evidence justifies it | Defer |

## Design

Implement small Python modules under `scripts/` for read-only process/cgroup observations, workloads and CLI orchestration. Use pinned production assets in an offline cache, disable embedding caching, and require a finite container memory limit. Archive flushed JSONL phase records so completed measurements survive a later process failure. Sampling retains bounded aggregate peaks rather than every sample.

Compare serial cold/warm loads with both models resident, the configured internal/API batch ceilings, overlapping model loads as a separately labelled pressure experiment, and cancellation of an already dispatched owner followed by a waiting live computation. Cancellation must retain its permit until native work completes. Validate batch output against the same owner's single-image result without loading an extra reference model into capacity measurements. Cold OpenVINO export uses a private temporary IR root rather than invalidating existing entries.

Keep worker/admission settings and numerical contracts stable until measurements support a specific change. Report process high-water marks separately from sampled stage peaks and cgroup lifetime peaks. Read cgroup v1/v2 according to process membership and mount roots; missing counters are unknown, never fabricated zeroes. Routine tests use synthetic filesystem/workload controls and never download model weights.

## Validation and outcome

The final Python 3.12 suite passed **561 tests**, with two optional production-weight cases skipped in routine execution. The 68 additional cases exercise memory-controller discovery/accounting, bounded sampling, workload parity/ownership, deadline compatibility and initialization serialization. Coverage is **94.31% lines / 88.71% branches**, above unchanged **89.42% / 79.37%** floors. CPU/OpenVINO image builds and shipped non-root native/server smokes passed. Strict OSV covered **56 CPU, 58 OpenVINO and 67 QA packages**, with no skips or known vulnerabilities.

Native experiments used the same approved publisher revisions, Torch **2.14.1+cpu**, Transformers **5.18.0**, Pydantic **2.13.5** / pydantic-core **2.46.5**, and OpenVINO **2026.4.0**. One process used Torch's actual defaults of eight intra-op and 16 inter-op threads, one inference owner, both models resident and no embedding cache. Deterministic 974-square PNG fixtures approach the aggregate pixel ceiling: a 32-image batch accounts for **31,963,264 source/resize pixels**. Each scenario validates batches 1/8/32 against same-owner single vectors at `rtol=atol=1e-4`, including mixed normalization. Loaded devices are checked as `cpu` or `ov:CPU`.

| Native case | Outcome | Process peak GiB | Cgroup peak GiB |
|---|---|---:|---:|
| Baseline CPU protected maximum batch, 15-second deadline | 504 at 15.021 s; native owner retained | 2.57 | 1.11 |
| Baseline OpenVINO protected maximum batch, 15-second deadline | 504 at 15.016 s; native owner retained | 4.38 | 4.29 |
| Guarded CPU overlapping cold-load requests, then batches | Passed; large batch 24.90 s | 2.51 | 1.03 |
| Guarded OpenVINO overlapping private cold exports, then batches | Passed; large batch 22.43 s | 6.31 | 4.83 |
| CPU detached large batch followed by live base-model batch | Passed; live dispatch waited 17.92 s | 2.55 | 1.03 |
| OpenVINO detached large batch followed by live base-model batch | Passed; live dispatch waited 14.40 s | 4.38 | 2.83 |
| Final CPU protected maximum batch, 45-second default | 200 in 13.923 s; complete vector parity | 2.56 | 1.04 |
| Final OpenVINO protected maximum batch, 45-second default | 200 in 12.786 s; complete vector parity | 4.36 | 2.81 |

An intermediate 30-second deadline still produced a CPU 504 at 30.043 seconds, followed by successful live work and settled ownership. That observation prompted the finite 45-second default. All protected-route experiments also enforce unauthenticated 401 and oversized-batch 413 behavior. Detached experiments cancel only after dispatch, observe one retained permit/read lock and one live waiter, run actual maximum batches, and finish with zero owners/waiters/readers. Before the initialization guard, overlapping OpenVINO cold requests reproduced a Transformers import failure; the guarded native replay passes. See the separate [deadline](embedding-deadlines.md) and [initialization](model-initialization.md) records.

All archived containers used `--network none`, the shipped non-root user, dropped capabilities, no new privileges and equal memory/combined-swap limits: **4 GiB CPU / 8 GiB OpenVINO**, with zero additional swap. Completed probes reported no memory-limit hits or OOM kills; the failed baseline overlap exited with ImportError rather than OOM. Private cold IR was removed while original cache entries remained usable. The [measurement archive](validation/capacity-2026-10-02.json) retains configuration, immutable image identities, source/journal hashes, stage samples, lifetime peaks, counters, ownership and strict installed-package inventories. Weights and raw build/runtime logs remain outside Git.

These timings are observations on a shared host, not paired throughput benchmarks. A small QA run overlapped the guarded cold-export replay. Host load, first-touch costs and page-cache charging vary; shared file pages may be charged outside a warm container, so its cgroup peak can be lower than process RSS. No host cache flush was performed. Missing v2/native counters remain unknown, and this host exercised cgroup v1; v2 semantics have synthetic tests. These fixtures approach the pixel ceiling while compressing efficiently, so they do not establish all concurrent HTTP-body/byte-limit combinations, accelerator VRAM, Intel GPU execution or hosted CI behavior. The initial CLI/backend mismatch was caught and excluded from the archive; backend selection is now mandatory and verified after loading.

Keep one worker and current ingress/admission budgets, CPU 4 GiB and OpenVINO 8 GiB containment, with explicit operator overrides. Retain serialized cold initialization and tune embedding/client/proxy budgets on target hardware. The next implementation is **hash-locked dependency profiles and a reviewed update policy**; rerun model/artifact and capacity contracts when native versions change. Continue accelerator and concurrent-input calibration before increasing owner counts.
