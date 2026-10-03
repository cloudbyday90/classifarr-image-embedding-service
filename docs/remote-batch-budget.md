# Shared remote batch fetch budget

## Decision and design

Reviewed on **2026-10-03**. Give each `embed_batch` invocation a lazy, shared monotonic deadline, starting immediately before its first remote fetch. Default `REMOTE_BATCH_FETCH_TIMEOUT_SECONDS` / `[image].remote_batch_fetch_timeout_seconds` is **30 seconds**. Each child receives the earlier of its existing 15-second per-image deadline and the shared deadline. Both explicit `/embed-batch` requests and automatic multi-item batch-window groups use this path. A coalescer with one live job retains the existing single-image path and its per-image budget.

The shared interval includes remote interpreter startup, DNS, transport, redirects, reads, result validation, cleanup and time between remote items in the byte-resolution/cache phase. Inline decoding and cache lookup between remote fetches consume elapsed time; inline items before the first remote fetch do not start the clock. Image decoding, model initialization, preprocessing and inference follow byte resolution and retain their existing ownership and embedding deadline. This is a fetch-phase bound, not a hard whole-batch execution bound.

Preserve ordered partial results. On exhaustion, the active fetch is terminated/reaped, then it and subsequent remote items report `Remote batch fetch timed out`. Do not launch another child for an expired budget. Successful earlier remote inputs and valid inline inputs still proceed to cache lookup/inference. A per-image timeout before shared expiry remains `Remote image fetch timed out`. Cached remote embeddings still require fetching current bytes, so they also consume this budget. Each invocation owns its budget; no mutable setting, singleton, global clock or thread-local deadline is introduced.

## Official research and alternatives

Sources were discovered through web search/open tools on October 3, rather than constructed URLs. Python documents a [monotonic clock unaffected by wall-clock updates](https://docs.python.org/3.12/library/time.html). Requests explains that its [socket inactivity timeout is not a whole-download limit](https://requests.readthedocs.io/en/latest/user/quickstart/). Python requires [killing the child and finishing communication after `communicate` timeout](https://docs.python.org/3/library/subprocess.html); OS process creation cannot necessarily be interrupted. These support retaining parent-owned termination and sharing an absolute deadline. The 30-second value and partial-result policy are project decisions, not upstream prescriptions.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Per-image limits only | Independent allowance for every image | Up to 32 sequential allowances can accumulate after caller timeout | Replace for batches |
| Shared fetch-phase deadline | Bounds cumulative remote occupancy; preserves partial success and independent native inference ownership | Later remote inputs can fail; inline/cache work between fetches uses elapsed time | Adopt, 30-second default |
| Sum only active fetch durations | Inline work never consumes allowance | Extra accounting; excludes elapsed gaps and weakens the simple deadline contract | Defer |
| Hard whole-batch deadline | One total end-to-end ceiling | Python thread cancellation cannot terminate native inference safely; needs a separate model-process architecture | Defer |
| Concurrent remote fetching | Lower latency for healthy remote inputs | More children, sockets, storage and admission accounting | Defer until deployment capacity is measured |

## Security and validation plan

Retain remote URLs disabled by default, allowlists, public numeric destinations, original-host TLS validation, redirect/address/byte ceilings, isolated environment, bounded protocol and anonymous image storage. Timeouts contain no URL or token. Shared expiry must not release inference permits before the active child is reaped. The HTTP waiter's timeout remains independent and does not cancel a dispatched owner.

Validate deterministic clocks for lazy/shared boundaries and late completion; mixed inline/remote, cache-hit, all-failed and separate-invocation behavior; native blocked DNS and controlled HTTP children; automatic groups and API ordering; detached-owner capacity recovery; positive finite configuration precedence. Run full locked QA and focused native Windows subprocess checks. OS launch/reap and scheduling latency can extend observed wall time beyond the configured deadline; no peer-receipt or hard real-time guarantee is claimed.

## Outcome

Implemented `remote_budget.py` as a small per-invocation service; the process supervisor clips its absolute deadline and refuses launch when expired. The embedder passes the budget explicitly through byte resolution without changing cache keys, settings, response schemas or model execution. Transport unit fakes now cover both call paths; fresh-child tests separately prove production supervision.

Full locked Linux QA passes **914 tests**, with seven existing platform/optional skips and **26 added cases** (19 batch/policy/native/API cases plus seven configuration cases). Coverage is **95.22% lines / 90.12% branches**, above unchanged **89.42% / 79.37%** floors; the budget module has complete line/branch coverage. Focused native Windows validation passes **57 cases** on Python 3.14.5, including real DNS, HTTP/TLS, process failures and shared batch supervision. Windows uses its existing separate host package graph; Linux QA uses the complete reviewed 67-wheel graph.

Native source-overlay CPU, OpenVINO and CUDA image probes each process 32 blocked-DNS inputs with Torch resident in the parent. A **0.9-second** shared budget settles in **0.904 / 0.905 / 0.905 seconds**, launches only one child and returns only after that child is reaped with stdin closed. No model weights were downloaded or model/device inference repeated; the changed contract is fetch preparation/ownership. The native success-then-trickle case also proves that later fetching spends the original remaining time rather than receiving another allowance. HTTP timeout checks prove 504, retained owner/429, then recovered 200 capacity.

Scoped Ruff and Pyright pass, as do copyright, input/inventory contracts, coverage, Markdown links and whitespace checks. The one changed resolver fake in the legacy invariant test retains seven identical pre-existing Ruff findings; a baseline comparison records them without suppressions or unrelated edits. Native CodeQL security-extended analysis matches all four changed runtime sources and retains only the two existing contextual setup alerts (explicit `--show-key` output and readable nonsecret config). No new alert or suppression was added. Configured Gitleaks checks are recorded in the [validation archive](validation/remote-batch-budget-2026-10-03.json).

The first full run exposed seven old fixture mismatches: an inline resolver fake lacked the new keyword and direct transport fakes covered only the single-image call path. Those fixtures were updated, their existing policy assertions retained, and the final full run above passed. The implementation stays on master with Unreleased updates and no release/tag/version bump. Next measure deployment accelerator/concurrent-input/PID/storage capacity before raising concurrency.

Related records: [request lifetimes](request-lifetimes.md), [response sending](response-send-lifetimes.md), [PR 46](pr-46-local-validation.md), [project skill](project-resource-skill.md), and [recommendation stack](recommendation-stack.md).
