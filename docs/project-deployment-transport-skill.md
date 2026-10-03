# Deployment transport AI skill extension

## Design

Extend the existing repository capacity-calibration skill with a conditional
transport reference. A separate overlapping skill would make trigger selection
less clear. The reference establishes the operator's topology, names the owned
forwarding configuration and reuses existing parser and capacity probes.

The operator uses a direct Docker port. The skill preserves that choice, separates
Docker/NAT address observations from external-client attribution, and distinguishes
a quota transport smoke from real-model capacity evidence. It adds no publishing,
production-change or PR authority.

## Outcome

The six routing/evidence cases in the reference were evaluated against the
implementation and existing evidence contracts: direct mode preserves topology;
ambient wildcard trust cannot enable forwarding; untrusted peers cannot change
address or scheme; trusted loopback peers can; Docker/NAT address preservation
remains an operator measurement; dependency-only updates do not trigger capacity
work. Native parser regressions execute the trust cases; the separate published
port smoke executes the direct topology. Real pinned CPU/OpenVINO measurements
remain separate from that smoke.

The skill creator's metadata validator passes. The extension uses one conditional
reference and the existing socket probe rather than a second benchmark runner.
It corrects the earlier body-order wording to require header-only 401 with zero
application receipt. This is a focused instruction/evidence evaluation, not an
independent agent benchmark. Implementation and measured limitations are recorded
in [the direct deployment outcome](direct-deployment-trust.md).
