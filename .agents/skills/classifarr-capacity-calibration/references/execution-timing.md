# Timing queued embedding requests

Use the representative probe's `execution_timings` when a concurrent batch
approaches or exceeds its deadline. Keep the requested workload, exact runtime
graph and containment visible; a library migration and a timing comparison
must not silently use different model/runtime versions.

The optional trusted observer belongs to the existing `InferenceExecutor`.
The shipped application supplies no observer. The probe binds only expected
labels, uses bounded per-experiment records and a single server performance
counter, and follows owner completion after the HTTP waiter leaves. Reuse
`capacity_execution_timing.py`; do not install a mutable global timer or infer
queue duration by subtracting independent client/worker clocks.

Read `events_ns` and `durations_seconds` together. Admission wait ends on permit
acquisition; read-lock wait and AnyIO worker-dispatch wait have distinct bounds.
Native-call wall time includes image preparation, inference and result
construction plus private probe verification. It does not identify kernels or all CPU usage across native
threads. Missing phases remain unknown. An admission cancellation without a
native start is not a zero-cost inference. A detached native tail is retained
work, even if its HTTP response already reports 504.

Require complete owner/native outcomes and settled queue/readers/ingress, even
on failed calibrations. Require `native_result_verified` for completed native
calls, including detached owners; unstarted work stays unknown. The probe facade
checks native values privately without passing vectors into the generic observer;
the HTTP client independently checks received values. A successful response without its complete
timing chain invalidates timing evidence. Keep actual HTTP/vector failure as
failure; do not extend deadlines merely to make an experiment pass. Observation
callbacks must be quick and thread-safe; ordinary callback failures must not
change native results, cancellation or capacity release.

If a hard client/server failure prevents a final report, retain the available
phase error, resource and cleanup evidence; missing timing evidence stays
unknown. Do not label that run a completed successful calibration.

Fresh-child startup budgets can expose host file-share latency. For such
validation on Windows Docker, compare a byte-verified frozen source copy on a
native Linux volume. Use `volume-nocopy` for a fresh source volume so an image's
older checkout cannot populate it implicitly. Keep original failures and do not
weaken deadline assertions to hide infrastructure effects.

Attribution currently covers explicit `/embed-batch` requests. Do not assign
one coalesced owner to every member request or treat native-call wall time as
GPU kernel time. Use a scoped backend profiler only when phase evidence shows
that deeper native detail would resolve the question. Keep profiler overhead
separate from unprofiled capacity results.

Evaluate: a 504 before native start; a 504 during native work; a shared-lock wait;
a saturated worker limiter; callback failure; missing completion; simultaneous
requests with distinct contexts; and a request for an unrelated dependency floor.
Use event-controlled ownership regressions and real socket/model evidence for
the affected path. Preserve the operator's Python/direct-port decisions and
target-hardware gates.

See [design and outcomes](../../../../docs/execution-timing-calibration.md).
