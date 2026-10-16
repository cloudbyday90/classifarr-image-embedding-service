# Cooperative inference ownership

Use for inference stop checkpoints, forward chunking or coalesced-caller changes.
Read `execution_cancellation.py`, `execution.py`, `batch_inference.py` and the
affected embedder path before editing. Keep one context-bound stop event per
execution owner. A mutable embedder flag can cancel unrelated requests.

The HTTP waiter and synchronous owner have separate lifetimes. Set the stop
event when the owner loses its waiter; coalesced groups must wait for their last
live member. Cancelling one member cannot cancel shared useful work. Check
outside per-item exception handlers so cancellation cannot become an ordinary
bad-image result. Existing supervised fetches must finish/reap before the next
checkpoint returns; no ownership release while a thread or child still runs.

Chunk size controls work per native call, independently of API item and batch
collection ceilings. Preserve aggregate byte/pixel accounting across chunks,
original result order, partial input errors, normalization and cache identity.
Do not publish incomplete-batch cache entries. A completed final chunk may
finish normally after cancellation; no later chunk is required to observe a
stop request that arrives after its final checkpoint.

An active native call is indivisible here. Keep its capacity, read lock, images
and model until it returns and cleanup finishes. A cooperative stop is neither
native failure nor a successful full-batch result. Missing full-batch vector
verification stays unknown. Native errors occurring during cancellation must
remain observable and must fail validation.

Evaluate an expired admission, cancelled worker-limiter wait, stop during fetch,
decode/model preparation and between chunks, final-chunk completion, an active
native error, two concurrent owners and coalesced groups with one versus zero
live members. Require event-controlled barriers, image closure, permit/read-lock
retention, settled owners and valid recovery. Run actual CPU/OpenVINO chunks
against private references when forwarding changes. Use the capacity probe's
cooperative scenario to prove one committed chunk completes and validates while
future chunks are skipped; label its artificial barrier separately from timing.

Keep ordinary standalone embedding calls usable without an execution context.
Do not claim kernel interruption, response-disconnect detection or a hard stop
bound. Use the existing capacity-calibration skill for target-host replay;
dependency-only maintenance does not need this cancellation workflow.

See [design and outcomes](../../../../docs/cooperative-inference-cancellation.md).
