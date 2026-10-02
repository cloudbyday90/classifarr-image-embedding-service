# Bounded batch-window admission

Decision date: 2026-10-01. Status: implemented and locally validated.

## Problem and official research

The optional coalescer retained jobs in an unbounded queue before acquiring inference capacity. Collected jobs were absent from queue telemetry, bypassed the waiting limit, and could contain expired payloads when a mixed group eventually dispatched.

Official sources were discovered with web search and retrieved with the web tool on 2026-10-01. These are live documents, not a claim about future releases.

- [Python 3.12 asyncio queues](https://docs.python.org/3.12/library/asyncio-queue.html): zero queue size means unbounded; a bounded queue's `put` can block. Bounding that queue alone would not bound the request tasks retaining payloads while blocked on `put`.
- [Python 3.12 tasks and deadlines](https://docs.python.org/3.12/library/asyncio-task.html): deadlines use the event loop clock, and cancellation requires cleanup. Preserve the separately owned inference task from the preceding repair.
- [Starlette thread pool](https://www.starlette.io/threadpool/): the AnyIO pool is shared with synchronous endpoints and dependencies. Increasing thread capacity does not address unbounded admission.
- [OWASP API4 resource consumption guidance](https://api-security.owasp.org/editions/2023/en/0xa4-unrestricted-resource-consumption/): constrain memory, execution time, payloads, and batch work in addition to request frequency. This repair bounds retained job count; decoded image bytes and HTTP body budgets remain separate work.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Bounded asyncio queue alone | Small change | Blocked producers retain payloads; collected jobs escape its bound; zero means unlimited | Reject |
| Separate coalescer limit | Isolates the change | Competing endpoint waiting rooms and telemetry can exceed the configured global limit | Reject |
| Shared FIFO admission tickets with weighted batch members | One limit across endpoints; bounded retained jobs; explicit permit ownership and cancellation | Collection briefly reserves a computation slot; requires careful ownership transfer | Adopt |
| External broker and isolated workers | Durable scheduling and process isolation | Adds deployment, model initialization, IPC, and operational complexity | Defer |

## Design and compatibility

Add a focused admission module, reuse the existing queue facade and read/write lock, and let the executor accept an admitted ticket. A group reserves one computation slot before collection. Waiting group members each count toward `IMAGE_EMBEDDER_MAX_QUEUE`; a compatible member can join an already admitted collecting group without consuming another inference slot. A group never exceeds `EMBED_BATCH_MAX_SIZE`.

With concurrency C, batch maximum B, and waiting maximum Q, retained admitted single-image jobs and explicit-batch requests are bounded by C × B + Q. Direct and explicit-batch requests occupy one ticket each, and explicit batches can hold multiple images. This is a request-count bound, not a byte-memory bound. Admission tickets progress FIFO across both endpoints; compatible joins can share an earlier ticket but cannot extend its collection deadline or exceed its member limit. A waiting group inherits its first member's queue deadline, so later members can expire sooner than a fresh individual ticket would.

`in_flight` includes computation slots reserved for collection and shared-lock acquisition. `waiting` counts individual requests awaiting capacity, including batch-window members. With Q=0, a new group fails with HTTP 429 when all slots are occupied; compatible requests may still join an admitted open group. Queue expiry returns HTTP 504, closed submission HTTP 503, and response schemas remain stable.

Caller cancellation removes retained membership immediately. The last client leaving cancels undispatched work or detaches a running worker. Rebuild payloads and result association from live jobs after shared-lock acquisition, immediately before thread dispatch. The executor alone releases a transferred permit after real worker completion. Shutdown settles groups and drains existing owners without freeing running-worker resources.

The service remains Python; no CommonJS or new JavaScript is introduced. Keep the current Python/FastAPI/AnyIO stack with application-scoped modules.

## Acceptance and outcome

Implemented `admission.py`, the queue facade, executor transfer/late preparation, and per-model bounded groups. Added ten admission tests and ten coalescer/HTTP tests. Updated two existing error-path tests to exercise public submission and real accounting instead of the removed private dispatcher. Existing thread-ownership and shutdown regressions remain in place.

The tests cover shared FIFO ordering, Q=0, finite burst limits, waiting telemetry, queue deadlines before collection, canceled payload removal, mixed-model grouping, result association, partial acquisition, shutdown, and continued permit ownership after HTTP 504. A burst probe with C=1, B=2, Q=3 admitted five of twenty jobs and rejected fifteen with HTTP 429; both headers and health showed three waiting requests. A mixed group blocked on the shared lock excluded its expired middle item and delivered the remaining results in order. Group task references survive until completion, including cancellation before coroutine startup.

Shutdown closes the shared admission pool as well as executor-owned tasks. Closure errors are reconstructed per caller so the pool retains no growing caller traceback. Ticket ownership prevents departing clients from freeing a running worker's capacity.

Local validation uses Windows, Python 3.14.5, pytest 9.0.3, AnyIO 4.12.1, real threads, and ASGI requests with controlled fake inference. The complete suite passed 236 tests. Final coverage reached 92.80% lines and 85.48% branches, with the executor at 100% for both. Ruff correctness/import checks and Pyright passed for the four modified service modules; new modules pass formatting checks. Copyright, compilation, installed dependency consistency, workflow lint, and the unchanged coverage ratchet passed.

No real GPU/model download, hosted GitHub runner, or native Linux signal subprocess was exercised locally. Per-process request admission does not bound HTTP body parsing, decoded pixels, every explicit-batch image's memory, or the combined capacity of multiple Uvicorn workers. Keep authentication and URL restrictions, set deployment memory/body budgets, and address backend builds and decoded-image limits next. This is a specific availability improvement, not a full repository security audit.
