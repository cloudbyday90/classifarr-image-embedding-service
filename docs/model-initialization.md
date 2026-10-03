# Process-owned model initialization

Assessment date: 2026-10-02. Baseline: `15a39337b272ef4db11eccdf58f1116f7bbea2cb`. Discovered during [capacity calibration](capacity-calibration.md).

## Evidence and official research

A fresh offline OpenVINO container requested both approved model aliases concurrently through one ImageEmbedder. It failed with `ImportError` importing `CLIPImageProcessorPil` from Transformers 5.18.0. This was a native runtime failure, with no OOM kill or memory-limit hit. Existing per-alias locks prevented duplicate loading of the same alias, but different aliases could initialize shared library state concurrently.

GitHub MCP fetched the exact [Transformers 5.18 lazy-import implementation](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/utils/import_utils.py). It dispatches deferred imports from shared package objects. The observed failure supports serializing application-owned initialization; it does not prove a complete upstream root cause or establish a security vulnerability. Official [Python RLock guidance](https://docs.python.org/3.13/library/threading.html), discovered through web search, describes reentrancy and recommends context-manager acquisition/release.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Per-alias locks alone | Different cold models can load concurrently | Reproduced cross-alias import failure; concurrent export peaks overlap | Reject |
| Eagerly import every dependency at startup | May avoid this import race | Adds startup work and does not serialize conversion/library initialization | Reject as sufficient remediation |
| Small process-wide initialization guard, with direct warm-cache returns | Protects shared library setup across aliases and owners; limits simultaneous exports | Cold initialization requests wait; native work still cannot be forcibly stopped | Adopt |
| Separate exporter process/service | Stronger lifecycle separation | New protocol and artifact ownership, before current evidence requires it | Defer |

## Design

`model_initialization.py` owns one process-scoped RLock, a context manager and a typed decorator. ImageEmbedder acquires its per-alias lock followed by this guard only after the warm-cache fast path. Shared processor, vision-model and OpenVINO loaders also hold the guard; reentrancy allows nested loading without deadlock. All model owners in the process use the same guard. Warm compiled/native inference does not acquire it when the model is already cached.

This is a bounded responsibility module, not a new application singleton. It creates no cross-process protocol or model registry. Separate worker processes still own separate memory and locks; deployment remains one worker by default. Existing versioned IR file locks protect cache publication between processes. A native conversion stuck inside the guard requires the existing supervisor/container containment; an HTTP timeout is not cancellation of initialization.

## Validation and outcome

Four event-controlled regression cases pass: distinct aliases within one owner, separate owners, nested guard acquisition/exception release, and a warm cached model returning while another owner initializes. The final suite passed **561 tests**, with two optional production-weight cases skipped; coverage is **94.31% lines / 88.71% branches**. Scoped typing and source lint pass for the new module and affected loaders.

The actual OpenVINO cold-overlap replay now initializes both approved models with loaded devices verified as `ov:CPU`, then passes single-reference and batch 1/8/32 parity. The guarded load phase took **93.427 seconds** and the whole experiment peaked at **6.31 GiB process RSS / 4.83 GiB cgroup memory**, within the unchanged 8 GiB/no-swap container. No memory-limit hits or OOM kills occurred. A small QA run overlapped this cold replay, so the duration is an observation rather than a comparative performance result. CPU overlapping-load requests also pass, along with actual detached/native-owner experiments on both profiles.

The [capacity archive](validation/capacity-2026-10-02.json) includes the failed baseline ImportError, guarded replay and immutable image identities. Existing original IR entries remain usable after private cold export. No IR schema/version bump is required because serialization/numerical policy is unchanged. Retain one worker and the process guard; consider an exporter process only if target-host cold retention or initialization lifetime requires stronger isolation.
