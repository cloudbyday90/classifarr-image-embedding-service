# Reliability and security recommendation stack

Assessment date: 2026-10-01. These recommendations combine local review, reproducible probes, and official documentation linked in the individual design records.

## Priority and tradeoffs

| Priority | Recommendation | Benefits | Costs or limits | Outcome |
|---|---|---|---|---|
| 1 | Track inference owners independently of HTTP request deadlines | Restores concurrency and lock accounting during slow work, cancellation, and retries | Running Python threads cannot be forcibly killed | Implemented; see [execution design](inference-execution.md) |
| 1 | Preserve server signal ownership and drain application work in lifespan | Allows Uvicorn shutdown to reach the new drain logic and prevents cleanup racing inference | Drain, server, and container deadlines must be configured together | Implemented as part of execution lifecycle ownership |
| 1 | Apply randomly selected open PR 28 locally | Raises the pytest minimum to include upstream fixes | Lower-bound dependency policy remains | Implemented and tested locally; see [PR record](pr-28-local-validation.md) |
| 2 | Bound admission before the optional batch window retains payloads | Makes queue limits, deadlines, and waiting telemetry meaningful under burst traffic | Must distinguish individual requests from a single batched inference permit | Next recommended task |
| 3 | Repair backend image/dependency contracts and add build smoke checks on PRs | Detects missing CUDA bases and incompatible Torch/runtime selections before release | CI cost; GPU correctness needs suitable hardware | Separate packaging change |
| 4 | Bound decoded pixels and aggregate batch memory | Limits memory expansion from small compressed inputs | Requires a documented image/batch memory budget and compatibility decisions | Separate input-processing change |
| 5 | Add offline real model/preprocessor contract tests | Detects regressions hidden by mocked backends | Additional test runtime and backend-specific environments | Separate verification change |

## Final recommendation

Keep the existing Python/FastAPI/AnyIO stack. Add an application-scoped execution module with explicit ownership, reuse the queue and read/write lock, and keep route modules focused on HTTP behavior. Test event-controlled workers so a quick 504 cannot conceal work that continues in the background. Preserve current authentication and remote-image validation, avoid logging payloads or secrets, and observe detached failures.

After this repair, address bounded batch-window admission. The current unbounded pending queue can retain requests outside `EmbedQueue`, report zero waiting requests, and keep processing after callers have expired. Discarding canceled jobs before dispatch is useful mitigation but does not establish an admission bound. Acceptance for the next task should cover max-queue zero, finite queue limits, cancellation, accurate pending telemetry, and fair interaction with `/embed-batch`.

The implemented group watcher now removes executor admission when all group clients expire; a mixed group can still retain an expired item while serving live clients. Rebuild its live payload at actual dispatch as part of the next task, while keeping one inference permit per group and preserving each client's result mapping.

Deployment packaging and decoded-memory limits remain material follow-ups. This task does not constitute a full repository security audit or proof that the complete system is secure.

## Delivery outcome

Execution ownership, lifecycle integration, and the selected PR patch are implemented. All 216 tests passed; line and branch coverage rose to 92.04% and 84.83%, above the unchanged committed floor. Two independent reviews found and verified the cancellation edge-case repairs. Detailed validation and platform limits are recorded in the execution and PR documents.

`README.md`, the prior implementation plan's status, and `CHANGELOG.md` under Unreleased are updated. Delivery targets the `fix/inference-execution-ownership` feature branch. No release or PR merge is part of this change.
