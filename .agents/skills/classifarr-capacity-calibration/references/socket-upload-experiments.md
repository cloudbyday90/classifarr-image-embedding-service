# Socket upload experiments

Use `scripts/capacity_probe.py --scenario socket` in the matching reviewed CPU or
OpenVINO image, with verified offline assets. Under current default pixel limits,
select `--image-edge 974 --repeats 1`. Keep production budgets unchanged.

The probe accepts bounded fixture settings and requires spare server connections
above application ingress capacity. These bounds govern the experiment, not the
service configuration. Refuse unsupported fixtures rather than masking application
rejection with Uvicorn's connection limit.

The client shares a JSON prefix and streams whitespace to the body ceiling.
Client, server, model and sampler belong to the same cgroup; report the combined
measurement. A uniform PNG padded to the byte ceiling tests this specific decoded
pixel pressure, not all codecs or image shapes.

Before sampling held uploads, require server counters at exactly one byte before
EOF for every admitted request and an idle native queue. Send only headers for
the excess request and require application 503 with zero body receipt. Cancel
held callers, observe server disconnects and released ingress, finish one retained
upload and compare its vectors with direct inference. Pre-EOF cancellation does
not test cancellation of dispatched native work.

Test unauthenticated requests with a complete valid body and record received bytes.
The current route dependency authenticates after body parsing. Do not shorten the
upload budget to turn that behavior into apparent early rejection. Declared body
overflow should reject on headers without receipt.

Mixed requests use the existing fixed remote fixture. Label its child-only
transport substitution and require every child reaped before fixture cleanup.
Do not alter production destination policy to produce a passing experiment.

Archive journal/source hashes, image identity, actual devices and numeric limits.
Distinguish point snapshots, sampled peaks and kernel lifetime peaks; keep limit
events and settled owners. Proxy buffering, TLS, HTTP/2 and target hardware require
separate evidence. Read [the design and outcome](../../../../docs/socket-upload-capacity.md)
for the current run and its limitations.
