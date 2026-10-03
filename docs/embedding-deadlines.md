# Independent embedding and remote-hop deadlines

Assessment date: 2026-10-02. Baseline: `15a39337b272ef4db11eccdf58f1116f7bbea2cb`. Discovered during [capacity calibration](capacity-calibration.md).

## Evidence and official research

Both baseline production profiles reproduced HTTP 504 at approximately 15 seconds for an authenticated 32-image ViT-L-14 batch. A following live request succeeded; the timed-out native computation retained its permit/read lock and settled. The admission/ownership design worked, but the single setting coupled the embedding HTTP deadline to remote-download hop timeouts. Serial large-batch observations ranged beyond 15 seconds on this host.

[Uvicorn settings](https://uvicorn.dev/settings/), discovered through official-domain web search, distinguish keep-alive and shutdown timeouts from application behavior. The embedding deadline belongs in the application, where queue/native ownership is already tracked. This change does not rely on a socket keep-alive setting to interrupt a model. Native work remains non-interruptible and keeps its permit after caller expiry.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Raise the shared timeout | One setting | Also lengthens each remote hop | Reject |
| Lower the API batch ceiling to eight globally | Fits the observed short deadline more readily | Changes accepted workloads and reduces batching opportunities on faster hardware | Keep as an operator option |
| Separate a finite embedding deadline, calibrated initially to 45 seconds | Adds headroom for measured batches; preserves remote-hop budget and queue ownership | Longer ingress occupancy; callers/proxies must align; slower hardware/queue wait can still time out | Adopt |

## Design and compatibility

`deadlines.py` owns positive finite duration validation and compatibility selection. Both embedding routes use `Settings.embedding_timeout_seconds`, including queue wait and computation. `REQUEST_TIMEOUT_SECONDS` remains the remote-hop budget, normally 15 seconds.

Set `EMBEDDING_TIMEOUT_SECONDS` or `[queue].embedding_timeout_seconds` to override the embedding deadline; environment takes precedence over TOML, and an explicit constructor value takes precedence over either. Durations may be fractional positive seconds. NaN, infinity, booleans, zero and negative values fail settings construction. Configured timeouts do not establish forcible thread cancellation.

If the new setting is absent, an explicitly supplied legacy `REQUEST_TIMEOUT_SECONDS` environment variable retains its previous embedding budget, including an explicit value of 15. A nondefault legacy TOML/constructor timeout is also inherited. Otherwise the new embedding default is 45 seconds. The shipped TOML documents the optional new key without setting it, preserving legacy environment overrides. Callers mutating a Settings instance after construction should set `embedding_timeout_seconds` for HTTP deadlines; changing the remote-hop field no longer changes the already resolved embedding field.

For example, `EMBEDDING_TIMEOUT_SECONDS=45` with `REQUEST_TIMEOUT_SECONDS=15` gives a 45-second embedding response budget and a 15-second budget per remote hop. It does not bound total request-body receive, DNS or multiple download hops; those remain separate work. One worker, eight ingress owners, existing queues, memory ceilings and detached owner cleanup remain in force. Successful API schemas and embedding values are unchanged.

## Validation and outcome

The final suite passed **561 tests**, with two optional production-weight cases skipped; coverage is **94.31% lines / 88.71% branches**. The 29 new configuration cases cover independent defaults, TOML/environment/constructor precedence, explicit legacy compatibility, fractional values and refusal of invalid durations. Existing event-controlled route, queue and cancellation tests use the independent embedding field and continue passing.

Both baseline native profiles returned 504 at about 15 seconds for the protected 32-image large-model request. A 30-second trial still returned CPU 504 at **30.043 seconds**; ownership remained correct and the following live request succeeded. The finite **45-second default** adds headroom above that observed workload. Final default-setting containers returned **200**, complete vectors and metadata, in **13.923 seconds CPU** and **12.786 seconds OpenVINO CPU**. They also verified 401 without authentication, 413 above the batch ceiling, successful live work and zero settled owners. The [capacity archive](validation/capacity-2026-10-02.json) records intermediate and final outcomes explicitly.

Timing varies on the shared host, so this is not a paired speedup claim. Queue wait, first cold initialization or slower hardware can exceed 45 seconds; explicit budgets remain available and native ownership still survives expiry. Both current images passed non-root startup, authentication, native projection, dependency and graceful-shutdown smokes. Align deployment client/proxy deadlines, then refresh capacity evidence alongside dependency locks.
