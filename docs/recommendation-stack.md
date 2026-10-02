# Reliability and security recommendation stack

Assessment date: 2026-10-01. These recommendations combine local review, reproducible probes, and official documentation linked in the individual design records.

## Priority and tradeoffs

| Priority | Recommendation | Benefits | Costs or limits | Outcome |
|---|---|---|---|---|
| 1 | Track inference owners independently of HTTP request deadlines | Restores concurrency and lock accounting during slow work, cancellation, and retries | Running Python threads cannot be forcibly killed | Implemented; see [execution design](inference-execution.md) |
| 1 | Preserve server signal ownership and drain application work in lifespan | Allows Uvicorn shutdown to reach the new drain logic and prevents cleanup racing inference | Drain, server, and container deadlines must be configured together | Implemented as part of execution lifecycle ownership |
| 1 | Apply randomly selected open PR 28 locally | Raises the pytest minimum to include upstream fixes | Lower-bound dependency policy remains | Implemented and tested locally; see [PR record](pr-28-local-validation.md) |
| Completed | Bound batch-window admission with shared FIFO tickets | Enforces one waiting budget, visible pending members, immediate cancellation, and live payload preparation | Collection reserves a slot; compatible joins share a ticket; counts do not bound decoded bytes | Implemented; see [admission design](bounded-batch-admission.md) |
| Completed | Apply randomly selected open PR 42 locally, pinning checkout v7.0.1 | ESM action, safer trusted-event fork handling, immutable CI dependency | Patch updates require review; hosted runner not exercised locally | Implemented; see [PR record](pr-42-local-validation.md) |
| Next 1 | Repair backend image/dependency contracts and add build smoke checks on PRs | Detects missing CUDA bases and incompatible Torch/runtime selections before release | CI cost; GPU correctness needs suitable hardware | Next task; see [backend design recommendation](backend-build-recommendation.md) |
| Next 2 | Bound HTTP bodies, decoded pixels, and aggregate batch memory | Limits memory expansion and payload retention beyond the admission count | Requires documented budgets and compatibility decisions | Separate input-processing change |
| Next 3 | Add offline real model/preprocessor contract tests | Detects regressions hidden by mocked backends | Additional test runtime and backend-specific environments | Separate verification change |
| Next 4 | Pin remaining CI action dependencies and review workflow permissions | Makes other CI dependencies immutable and limits credential exposure | Version maintenance; permissions need job-specific validation | Separate workflow change |

## Final recommendation

Keep the existing Python/FastAPI/AnyIO stack. Use application-scoped admission, queue, execution, and batch modules with explicit ownership; keep routes focused on HTTP behavior. Test event-controlled workers so a quick 504 cannot conceal work that continues in the background. Preserve authentication and remote-image validation, avoid logging payloads or secrets, observe detached failures, and keep JavaScript dependencies on ESM-compatible releases.

The former unbounded pending queue is removed. Both endpoints use the same FIFO admission budget; queued coalescer members count individually. Compatible requests can share an admitted collecting group's slot without extending its window. Collection reserves capacity and is included in `in_flight`. These semantics, including zero-wait admission and early inherited deadlines, are documented in the admission design and README.

Mixed groups now rebuild live payloads immediately before thread submission, after acquiring the shared lock. Canceled jobs leave retained membership promptly; dispatched work remains owned until its thread completes. Cross-endpoint FIFO ordering and live result association are covered by regressions.

The next task is backend build validation: a fresh registry check still returns "not found" for the configured CUDA base. Avoid replacing that tag without matching the Torch wheel, runtime, driver, and architecture contracts. Follow with body/decoded-memory limits, real model contracts, and remaining CI pins. This task does not constitute a full repository security audit or proof that the complete system is secure.

## Delivery outcome

Execution ownership, lifecycle integration, bounded admission, and both selected PR patches are implemented. The admission change passed 236 tests. Final coverage reached 92.80% lines and 85.48% branches, above the unchanged committed floor of 89.42% and 79.37%. The executor retains 100% line and branch coverage. Focused Ruff/Pyright, copyright, compilation, dependency consistency, and actionlint checks passed. The pinned checkout's ESM/Node 24 and unsafe-fork refusal smoke checks passed locally. Detailed validation and platform limits are recorded in the individual documents.

`README.md`, configuration comments, the prior implementation plan's status, and `CHANGELOG.md` under Unreleased are updated. Delivery targets `fix/bounded-batch-admission`. No release, default-branch merge, or PR merge is part of this change.
