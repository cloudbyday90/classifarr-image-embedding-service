# Inference execution ownership

Decision date: 2026-10-01. Status: implemented and locally validated.

## Problem and scope

Before this change, both embedding routes wrapped a thread await in `asyncio.wait_for` and released their queue and shared-read permits in the request task's `finally` block. A deadline canceled that await but could not stop the thread. An event-controlled local probe demonstrated two active workers with concurrency configured to one, after two successive HTTP 504 responses. Queue telemetry incorrectly reported zero in-flight work.

This change covers single embedding, explicit batch embedding, and thread dispatch from the optional batch window. It preserves authentication, remote-image restrictions, cache semantics, response schemas, and existing error mappings. The service remains Python; no JavaScript or CommonJS is introduced.

## Official research

Sources were discovered through web search and opened through the web tool on 2026-10-01, rather than constructed from assumed paths. Documentation is live; these observations do not claim future releases or complete historical snapshots.

- [Python 3.12 coroutines and tasks](https://docs.python.org/3.12/library/asyncio-task.html): `wait_for` cancels its awaited operation on expiry; `shield` protects a separately retained task from its caller's cancellation. The service must keep strong references to its tasks and observe their outcomes.
- [AnyIO thread documentation](https://anyio.readthedocs.io/en/stable/threads.html?highlight=BlockingPortal) and [API reference](https://anyio.readthedocs.io/en/stable/api.html): canceling an await does not forcibly terminate its thread. A thread's continued resource use must remain accounted for. Keep the existing AnyIO thread implementation, with an independently owned task around it.
- [FastAPI lifespan documentation](https://fastapi.tiangolo.com/advanced/events/): application lifespan owns shared resources and their teardown. Register the executor per application and drain it during teardown.
- [Uvicorn server behavior](https://www.uvicorn.org/server-behavior/) and [settings](https://www.uvicorn.org/settings/): the server manages graceful process shutdown and its own timeout. Let Uvicorn retain process-signal ownership; the application's drain budget is separate from the server and container shutdown budgets.

## Alternatives

| Option | Advantages | Disadvantages | Decision |
|---|---|---|---|
| Shield the existing request operation wholesale | Small patch; avoids releasing a running worker's permits | Expired queued requests can still start; no task registry or shutdown policy | Reject |
| Application-scoped execution service with tracked tasks | Preserves prompt HTTP deadlines, actual concurrency, existing models, and queue telemetry; reusable by both routes | Requires explicit admission, detached-error, and shutdown handling; cannot kill a hung native thread | Adopt |
| AnyIO task group and cancellation scopes throughout the API | Structured lifetime and backend abstraction | Current queue/coalescer are asyncio-based; scope conversion is broader than this repair, and raw asyncio cancellation still needs care | Defer |
| Isolated inference worker processes | Can terminate irrecoverably stuck work and isolate failures | Model duplication, GPU/process initialization, IPC, startup cost, and recovery complexity | Reconsider if hard execution termination becomes a requirement |

## Design

Add a focused `execution.py` service, instantiated by `create_app`, rather than a global singleton. Each submission has an owner task and a dispatch marker. The owner acquires the existing concurrency permit and shared lock, dispatches synchronous work, and releases both only after that work returns or raises.

The request awaits a shielded owner. If the caller times out or is canceled before dispatch, cancel and settle the admission task so that it cannot later start work. If dispatch has begun, detach the caller and let the owner complete with its permits intact. Keep strong task references and consume detached exceptions without logging request arguments, image data, or credentials.

Owner tasks return an internal success/error envelope; attached callers unwrap errors into the existing HTTP mappings. Local Python 3.14.5 testing showed that shielding an exception-raising owner can report a detached failure to the event loop even when a done callback retrieves the exception. Returning the envelope avoids duplicate/unobserved failure reporting and preserves controlled, payload-free logging. This is an internal representation, not an API schema change. The [official CPython implementation](https://github.com/python/cpython/blob/main/Lib/asyncio/tasks.py), discovered during research, corroborates the shield reporting path; the local executable regression is the acceptance evidence.

Routes retain their existing deadline and HTTP error mapping. The batch coalescer uses the same service, so stopping the dispatcher cannot release the permits of its still-running thread. Settle collected and pending job futures on stop and skip already canceled jobs before dispatch. Bounded batch-window admission remains a separate follow-up.

Each dispatched group watches its client futures. If its last client leaves, cancel the group's executor submission: this removes waiting admission or detaches already dispatched work. A group with a live client continues. Cancellation is requested only once so concurrent caller expiry and shutdown cannot interrupt partial-permit cleanup. Admission closed by shutdown returns HTTP 503, including the race where an owner is canceled before its coroutine first runs.

On teardown, stop batch submissions, close executor admission, cancel work still waiting to dispatch, and wait up to `shutdown_timeout_seconds` for dispatched tasks. A drain timeout leaves ownership intact and skips GPU cleanup while inference is active. Python cannot forcibly terminate these threads; operational hard-stop limits remain the responsibility of the process supervisor. Application lifespan must not replace Uvicorn's signal handlers.

## Acceptance criteria

1. Both routes return HTTP 504 while an event-blocked worker remains active, with `in_flight=1` and its shared-read lock retained.
2. With concurrency one and no waiting room, retries receive HTTP 429 until the original worker exits; actual peak work never exceeds one.
3. Caller cancellation and admission deadlines remove undispatched work without leaking capacity.
4. Worker failures release capacity and detached failures produce no unobserved-task exception.
5. Shutdown rejects new work, settles waiting work, drains running work, and retains permits when the drain budget expires.
6. Batch-window stop does not orphan job futures or abandon running-worker accounting.
7. Concurrent caller cancellation and shutdown do not cancel partial-permit cleanup a second time.
8. The last batch client leaving during admission removes that group's waiting work; a remaining client still receives its result.

## Implementation and outcome

Implemented `execution.py`, application wiring, both embedding routes, batch dispatch, and lifespan. Added `worker_fakes.py` and three focused execution test modules. Independent reviews discovered the repeated-cancellation cleanup race and canceled-group admission gap; both were reproduced, repaired, and covered by regressions. A subsequent immediate-close scheduling regression verifies HTTP 503 semantics before owner startup.

Validation on Windows with Python 3.14.5, pytest 9.0.3, and AnyIO 4.12.1:

| Check | Result |
|---|---|
| Complete suite, `python -m pytest -q` | 216 passed |
| Focused execution, HTTP, and batch-window tests | 31 passed |
| Line / branch coverage | 92.04% / 84.83%; committed ratchet passed |
| New executor line / branch coverage | 100% / 100% |
| Ruff correctness and import checks | Passed for modified service modules and new test modules |
| Ruff formatting | Passed for new Python modules |
| Pyright | Zero errors in execution, batch, lifespan, and both embedding routes |
| Copyright, compilation, whitespace, and installed dependency consistency | Passed |

The application factory's existing SlowAPI exception-handler type diagnostic was independently reproduced from an archived HEAD checkout with the same interpreter. It is outside this repair; no new type suppression was added. Whitespace checking recognizes the repository's existing CRLF endings. Validation exercised actual threads and ASGI requests with controlled fake inference; no real GPU/model download or native Linux signal subprocess was exercised locally.

The next architectural task is bounded batch-window admission. Its pending queue remains unbounded. A group with both expired and live clients can still include an expired item prepared before admission; the follow-up should rebuild live items at dispatch and preserve result association. These limits are not claimed as solved here. Hard termination of a stuck native thread still requires process supervision or a future process-isolated executor.
