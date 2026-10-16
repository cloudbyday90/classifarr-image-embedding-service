# Execution timing skill extension

## Design

Extend the existing capacity-calibration skill with one conditional timing
reference. Use skill-creator guidance for narrow routing and progressive
disclosure rather than creating an overlapping performance skill.

The reference prevents several misleading conclusions: HTTP elapsed time is
not native execution time; thread CPU time excludes other native threads;
missing native start is unknown rather than zero; and an HTTP deadline does
not release a running owner. It routes deeper operator profiling only after
phase evidence justifies its cost. It adds no external-action authority.

## Evaluation and outcome

Evaluate queued versus detached expiry, read-lock and worker-dispatch delay,
callback failure, incomplete evidence and distinct request contexts against
the executable regressions and native experiments. Dependency-only floor
updates remain outside this routing. Metadata validation checks discoverability;
the event/ownership tests and real model outcomes establish behavior.

The 20 new regression cases and final **1,132-pass / seven-existing-skip** QA run
exercise the routed boundaries. Event-controlled tests distinguish admission,
read-lock and worker-dispatch waits, demonstrate ordinary callback failure
isolation, preserve separate contexts, reject incomplete evidence and check
both successful and corrupted detached results. The actual four-caller CPU
32-image run records a verified native result continuing 14.957 seconds beyond
response start, while two expired queued owners never start native work. CPU
eight-image and OpenVINO 32-image comparison profiles pass. Missing native start
remains unknown in both tests and journals.

The reference now also requires private native-result verification after HTTP
expiry, retains hard-failure evidence and documents byte-verified Linux source
volumes for Windows Docker startup-sensitive validation. Metadata validation
passes. This is manual instruction/scenario evaluation against executable
evidence, not an independent agent benchmark or proof of universal performance.
The conditional reference avoids loading timing details for dependency-only
maintenance and adds no external-action authority.

The [timing design and outcome](execution-timing-calibration.md) and
[validation archive](validation/execution-timing-2026-10-15.json) preserve the
research, pros/cons, exact results and remaining target-host gates. Extend this
reference when a measured timing boundary changes; keep generic profiling and
release operations outside its scope.
