# Real socket-upload capacity

Assessment: 2026-10-03. Baseline: `6d4221715bf9a4a9820077fd8a550503c22ebc6d`.

## Design and recommendation

Extend the existing capacity probe with a `socket` scenario on the actual
authenticated HTTP/1 server. Keep production limits and owner counts unchanged.
Measure incomplete uploads at the request-body ceiling, single-image byte/pixel
pressure, maximum batches and concurrent remote/inline inputs. Use small modules
for repeatable upload bodies, passive receive counters, temporary server ownership
and experiment orchestration. No new dependency or production destination bypass
is required.

Stage one upload per ingress slot with the final byte withheld. Require server-side
receive counts before taking the pressure snapshot. Send only headers for the
excess request and prove application rejection without body receipt. Disconnect
all but one staged caller, require released ingress without native dispatch, then
finish the retained request and validate its vector against direct inference.
Check authentication with a complete valid body and record received bytes;
at this baseline the route dependency runs after body receipt. Check declared overflow,
queue settlement and server cleanup.
Concurrent mixed requests reuse the existing probe-only remote child fixture;
record that transport substitution and require all children reaped.

Fixtures use valid PNGs. A single uniform square approaches the configured pixel
ceiling and is padded to the encoded-image byte ceiling; batch fixtures use the
selected edge and alternating existing reference images. JSON whitespace reaches
the body ceiling without allocating a full body for every client. Record actual
bytes/pixels and fixture sharing. Client, server, sampler and models share one
process/container, so their resource totals are deployment experiment evidence,
not isolated server allocation measurements.

## Official October research and alternatives

URLs were discovered through web search/fetch on October 3:

- [Uvicorn server behavior](https://github.com/Kludex/uvicorn/blob/main/docs/server-behavior.md)
  describes receive-driven flow control and early response behavior. Application
  consumption can still retain a complete body; transport watermarks alone do not
  bound total admitted request memory.
- [HTTPX asynchronous streaming](https://www.python-httpx.org/async/) supports
  async byte generators and scoped client closure. [Environment guidance](https://www.python-httpx.org/environment_variables/)
  supports explicit `trust_env=False` for this fixed loopback experiment.
- [Python streams](https://docs.python.org/3.12/library/asyncio-stream.html) provides
  bounded reads and explicit write drain/close for the header-only rejection probe.
- [Kernel cgroup v2 guidance](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
  distinguishes current membership-wide usage from lifetime peaks. Retain sampled
  process peaks and memory/PID limit-event deltas separately.
- [NGINX request buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)
  and [body buffers/temp storage](https://nginx.org/en/docs/http/ngx_http_core_module.html)
  introduce another owner and buffering policy. Direct-loopback results do not
  establish reverse-proxy behavior.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Increase concurrency from existing ASGI measurements | Small operational change | Missing wire/body retention evidence | Defer |
| Add a separate general benchmark framework | Broad load generation | New dependencies and duplicated ownership/measurement logic | Reject |
| Extend current probe with controlled native sockets | Existing models/counters, observed receive and cleanup boundaries | Fixture/client overhead; direct HTTP/1 scope | Adopt |

Final stack: retained modular Python service and hash locks; existing capacity
reader/sampler; scoped asyncio/Uvicorn HTTP/1 server; fixed loopback streaming
clients; native model/vector and ownership gates; separate proxy/hardware evidence.

## Outcome

Both fresh CPU and OpenVINO CPU containers passed with the two pinned models
resident, cache disabled and one inference owner. Each staged case held eight
requests at exactly **16 MiB minus one byte received**. Both excess requests
returned 503 without receipt. Seven callers per case disconnected; the retained
request completed with 200 and direct-reference vector parity. Maximum 32-item
mixed/inline batches returned 200, an actual waiter queued, and both remote
children were reaped. Final ingress, native owners, waiters and read locks were zero.

| Profile | Socket-phase sampled RSS (GiB) | Sampled cgroup (GiB) | Kernel lifetime cgroup (GiB) | Sampled tasks / parent threads | Socket-phase time (s) |
|---|---:|---:|---:|---:|---:|
| CPU | 2.94 | 1.62 | 1.62 | 44 / 43 | 56.81 |
| OpenVINO CPU, cached IR | 4.55 | 4.75 | 4.75 | 27 / 26 | 44.09 |

At the held single-image/batch upload points, RSS was **2.16 / 2.30 GiB** on CPU
and **4.27 / 4.41 GiB** on OpenVINO. The single fixture used 16 million source
pixels and 10 MiB PNG bytes; the alternating 974-edge maximum batches
approached the aggregate pixel budget. Remote images were padded to 10 MiB each.
Sampled temporary occupancy peaked at 10 MiB plus one filesystem block and
returned to zero. Memory/PID limit-event and OOM deltas were zero. The kernel
provides no PID lifetime peak, so sampled task peaks remain lower bounds.

Runs used existing immutable native images plus a read-only source snapshot,
verified offline publisher assets, eight Torch intra-op / sixteen inter-op threads,
4 GiB CPU or 8 GiB OpenVINO/no swap, 1024 tasks and 128 MiB non-executable tmpfs.
These are experiment containment bounds. Production image rebuilds and hosted
workflow execution are not claimed. CPU overlapped routine QA; OpenVINO started
as QA finished. This shared-host experiment is not a backend speed comparison.
Whole-phase times include references, uploads, rejection, recovery and multiple
requests; they are not individual HTTP deadline measurements. Shared page-cache
charging can make CPU cgroup accounting lower than process RSS.

The full locked QA suite passed **963 tests / seven existing optional or platform
skips**. Coverage is **95.22% lines / 90.12% branches**, above unchanged floors.
Focused socket checks, style/type checks, workflow syntax, copyright, skill metadata
and the coverage ratchet pass. Security scan outcomes and immutable source/journal
hashes are retained in the [measurement archive](validation/socket-upload-capacity-2026-10-03.json).
CodeQL security-extended covers 82 Python files and ten workflows, with no new
findings. Two unchanged context results remain: explicit `--show-key` output in
`generate_env.py` and nonsecret 0644 publication in `secret_publication.py`.
Verified Gitleaks source and 86-commit history scans found no secrets.
No rules or suppressions were weakened. All thirteen dependency profiles validate;
the CPU/OpenVINO/QA installed inventories and pip checks pass, with all lock
artifacts unchanged.
The [skill extension](project-socket-capacity-skill.md) and
[open PR availability](open-pr-availability.md) have separate design/outcome records.

Authentication returned 401 after receiving the complete **16 MiB** valid body on
both profiles. Next implement API-key rejection before body receipt, preserving
public probe routes, constant-time comparison, existing error/API contracts and
ingress/shutdown ownership. Then measure the operator's actual proxy buffering,
body temp storage and target hardware before changing admission or owner counts.
Keep Python and the present resource/model/dependency stack. No suitable unapplied
PR was available; none was adopted or merged. No limits were retuned and no release
or version change was created.

## Early authentication follow-up

The [separate authentication design and outcome](early-api-key-authentication.md)
implements the next fix identified above. The current probe requires header-only
401 with zero body receipt; the measurements in this document retain the original
baseline and its 16 MiB pre-authentication observation.
