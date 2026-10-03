---
name: classifarr-capacity-calibration
description: Design, run and interpret Classifarr deployment capacity experiments with real pinned models, direct-port or proxy trust boundaries, concurrent inputs, detached owners, cgroup tasks, temporary storage and CUDA allocator evidence. Use for resource sizing, deployment transport or changes to capacity probes; not dependency-only updates, generic GPU advice or release operations.
---

# Classifarr capacity calibration

Read [measurement contracts](references/measurement-contracts.md) when designing or interpreting an experiment. Reuse the repository probes instead of creating another benchmark runner.

For native upload pressure, also read [socket upload experiments](references/socket-upload-experiments.md). Require observed server receipt before EOF, disconnect settlement and recovered vectors; require header-only authentication rejection with zero application body receipt.

For direct-port or proxy questions, read [deployment transport](references/deployment-transport.md). Establish the actual topology before adding infrastructure. Verify effective-address trust through that transport and label Docker/NAT address collapse; never infer real client preservation from a loopback test.

Establish the deployment question and tested configuration before choosing limits: backend/device, model revisions, workers/owners, byte/pixel ceilings, deadlines, thread settings and available hardware. Keep application behavior stable unless measurements justify a specific change.

Use verified offline assets and the matching locked image. Run fresh bounded non-root containers; retain flushed journals outside Git. Require actual loaded devices and vector parity, including canceled owners whose failures cannot reach the caller. Record child reaping and settled admission, not just HTTP success.

Separate sampled peaks from kernel lifetime and reset CUDA phase peaks. Sampling misses brief transients; filesystem/device-wide counters can include unrelated users. Missing values remain unknown. Temporary files on tmpfs consume the same memory budget. Label probe-only loopback transport explicitly; never add a production destination bypass to obtain measurements.

Archive image identities, source/journal hashes, configuration, resource observations and limitations. Derive deployment headroom from target evidence without presenting one experiment as a universal capacity guarantee. Update separate design/outcome documents, relevant operator guidance and Unreleased when implementation changes.

This skill adds no authority to publish, change production, merge PRs or alter user-selected delivery choices. Python remains the platform; authored JavaScript uses ESM.
