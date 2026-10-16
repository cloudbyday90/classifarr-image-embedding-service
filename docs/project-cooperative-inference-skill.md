# Cooperative inference skill extension

## Design

Extend the resource-contract skill with a conditional cooperative-ownership
reference. This is an ownership workflow rather than another general performance
skill. The existing capacity skill remains responsible for measurement scopes.

The reference makes last-waiter semantics, thread-safe per-owner context,
indivisible active calls, aggregate input budgets and unknown incomplete results
explicit. It routes event-controlled regressions and the reusable native probe
instead of treating a cancelled HTTP await as proof of stopped inference.

Its benefit is repeatable reasoning at a fragile cancellation boundary. Its cost
is maintaining a focused reference as ownership changes. It adds no deployment,
PR mutation, publishing or external-message authority.

## Evaluation and outcome

The 30 new cases and final **1,162-pass / seven-existing-skip** suite exercise
the routed boundaries. Removing chunk checkpoints or corrupting a completed
detached chunk fails the probe's controls. Worker-limiter saturation, last versus
remaining coalesced callers, native errors during cancellation and all prepared
image closure pass deterministic regressions. Context restoration and two
concurrent owners demonstrate that stop signals remain scoped.

Both pinned models pass the controlled committed-chunk/recovery scenario on CPU,
OpenVINO CPU and the actual RTX 5070 Ti. Each case skips 24 future native images,
keeps active permits/readers until settlement and privately verifies its eight
committed vectors plus recovery. Incomplete full-batch parity remains unknown.
Real CPU/OpenVINO mixed-input runs reap the current child before stopping later
fetches. The reference routes dependency-only maintenance away from this workflow.

Skill metadata and relative-reference validation pass. This is manual evaluation
against executable evidence, not an independent agent benchmark or proof of
kernel interruption, socket-disconnect detection or universal performance.
The capacity skill's timing reference now distinguishes cooperative stops from
complete native results; measurement ownership remains with that sibling skill.
The [design/outcome](cooperative-inference-cancellation.md) and
[native archive](validation/cooperative-inference-2026-10-16.json) retain exact
evidence and the next unresolved disconnect boundary.
