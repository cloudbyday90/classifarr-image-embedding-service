# Representative workloads and independent client RSS

Use this reference when changing image workload profiles or interpreting capacity
after the operator identifies shapes/codecs. Reuse `capacity_probe.py` with
`--scenario representative --clients 2 --repeats 1`; use the matching backend
image and a fresh bounded container. Square `--image-edge` does not apply here.
Use `--clients 1` or `--batch-sizes 1 8` to compare caller/batch pressure under
the current default ceilings. Selected sizes must belong to configured ceilings;
they do not change production limits. Keep failed full-ceiling results archived.

Establish whether inputs are operator samples, synthetic fixture choices or
production distributions. The current operator selected portrait posters,
landscape and PNG/JPEG/WebP. The fixed profiles are 500×750 JPEG/RGBA PNG,
750×500 WebP and 224×224 high-entropy PNG, each with two variants. Never claim
these dimensions or compression patterns were measured in production.

Check `capacity_image_fixtures.py` allocation guards before creating model owners:
encoded per-image/aggregate bytes, source plus resize pixels, JSON body sizes,
1/internal/API batch ceilings, bounded repeats and admitted callers. Record
encoded hashes and codec/runtime versions; compute references from encoded bytes
so lossy JPEG/WebP changes do not invalidate the comparison. Validate ordered
metadata, finite vectors and request-level normalization with both real models.

`capacity_representative.py` owns the direct temporary server and reuses native
readers/sampling. `capacity_client_process.py` owns a fresh isolated HTTP client
interpreter. Require fixed loopback URLs, no ambient loader/proxy/service-key
environment, bounded stdin/output/HTTP responses and kill/reap on failure,
timeout or cancellation, including cancellation during launch. The disposable
key belongs only in private stdin and request headers, never arguments/journals.

Report client process RSS/high-water RSS separately. The model-owner process
still includes references, server, sampler and fixture generation. The cgroup
still includes the client, shared pages, page cache and kernel/socket charges.
Do not subtract independent high-water/sample peaks to invent service-only
memory. Sampling misses transients; PID/temporary-storage evidence remains
scoped. Report finite request bursts, not sustained throughput or percentiles.
HTTP/vector failures retain partial records, status/receipt, settled owners and
memory evidence, then exit unsuccessfully. Hard child failures can lack a final
client report; require phase error, cleanup and available memory evidence instead.
OS launch/reaping can add deadline latency.

Require server-observed receipt for every request, header-only 401 with zero
application body receipt and settled ingress/queue/readers/connections. The
probe's `1000/minute` quota is experimental; the production quota is unchanged.
Keep production workers, admission, memory containment and trust boundaries
until target-host evidence justifies tuning. No proxy is required for the
operator's direct Docker deployment. External-client addresses are a separate
network measurement; loopback evidence cannot establish preservation.

Focused regressions: `python -m pytest tests/test_capacity_client_process.py
tests/test_capacity_representative.py tests/test_capacity_workload.py -o addopts=''`.
Run actual pinned CPU/OpenVINO cases for affected measurement changes, complete
locked QA, modified workflow lint and skill metadata checks. Metadata validation
alone does not establish useful skill behavior.

Evaluate these cases against instructions and executable evidence:

| Prompt/evidence | Required response |
|---|---|
| Operator asks for posters, landscape and codecs without dimensions | Use documented synthetic choices; retain unknown production distribution |
| JPEG/WebP vectors differ from original uncompressed pixels | Compare to encoded-byte native references, not originals |
| Child times out, floods output or parent cancels while launching | Bound output and lifetime; reap before ownership returns |
| Client high-water RSS is known alongside cgroup peak | Report both scopes; do not subtract them |
| A concurrent request returns 429/504 or corrupt vectors | Fail calibration; never count refused/incorrect requests as throughput |
| Loopback succeeds and user uses direct Docker | Preserve topology; leave real external-address preservation unproved |
| Dependency-only update or documentation correction | Do not trigger a new workload run without affected measurement behavior |

See [design and outcomes](../../../../docs/representative-capacity.md) and
[skill evaluation](../../../../docs/project-representative-capacity-skill.md).
