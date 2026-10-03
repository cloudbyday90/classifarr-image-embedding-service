# Socket-capacity AI skill extension

Assessment: 2026-10-03. This extends the existing repository skill rather than
introducing another overlapping calibration workflow.

## Design

The [capacity-calibration skill](../.agents/skills/classifarr-capacity-calibration/SKILL.md)
now routes native upload questions to a focused
[socket measurement reference](../.agents/skills/classifarr-capacity-calibration/references/socket-upload-experiments.md).
Its purpose is to prevent a passing in-process API check from becoming an
unsupported deployment-capacity claim.

Use the existing real-model runner and resource counters. Require server receipt
before EOF, application ingress rejection, actual disconnect settlement, recovered
vectors and child reaping. Preserve fixed authorities, offline verified assets,
production budgets and scoped server/client ownership. The guidance distinguishes
late authentication, pre-EOF cancellation and proxy buffering from the boundaries
that the experiment actually exercises.

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Separate socket benchmarking skill | Narrow entry point | Duplicates model/resource/delivery rules | Defer |
| Extend calibration with a focused reference | One source of measurement policy; small trigger instructions | Agent must follow the linked reference | Adopt |
| Add a general autonomous deployment skill | Wider automation | Scope exceeds measured evidence and user authorization | Reject |

Final stack: existing calibration skill, focused progressive reference, modular
Python probe, native evidence archive and separate outcome documentation. This
skill grants no publishing, PR or production authority.

## Evaluation and outcome

Metadata validation passes. Local evaluation applies the instructions to the
actual socket experiment: one-byte EOF barriers use server counters; excess
requests reject without receipt; cancellation precedes native dispatch; retained
requests compare vectors; fixture children settle before teardown. The first
header-only authentication test exposed body-before-dependency behavior. Both the
probe and guidance now report full-body receipt instead of claiming early 401.

Negative cases refuse fixture budgets above the bounded experiment, a transport
limit that would mask application rejection, and bodies too small for valid JSON.
A caller-failure test proves temporary server closure. An ambient proxy/CA
negative control still succeeds because the fixed loopback client disables
ambient trust. Routine tests fetch no model weights.

These are maintainer scenario evaluations and executable regressions, not an
independent agent benchmark. Native results and remaining proxy/hardware limits
are recorded in [socket upload capacity](socket-upload-capacity.md).
