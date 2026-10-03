# Direct deployment and forwarded-header trust

Assessment: 2026-10-03. The operator confirmed direct Docker port publishing and
no reverse proxy. The next capacity recommendation therefore starts with the
direct transport and its effective client address. No proxy is added.

## Research and alternatives

URLs below were discovered and fetched through web services. Current
[Uvicorn settings](https://uvicorn.dev/settings/) enable forwarding by default,
with trust inherited from `FORWARDED_ALLOW_IPS` when present.
[Uvicorn deployment guidance](https://uvicorn.dev/deployment/) makes the actual
connecting peer the trust boundary and warns that trusting arbitrary peers lets
clients spoof addresses. The service uses that effective address for public
probe quotas, so its launcher must own this policy explicitly.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Inherit Uvicorn defaults and ambient trust | Convenient for existing proxies | A direct deployment can inherit forwarding trust unrelated to its topology | Replace in the shipped launcher |
| Disable forwarding unless explicit peers are configured | Direct client headers cannot change quota identity or scheme; one clear setting | Existing proxied users must migrate their trust configuration | Adopt |
| Add a reverse proxy | Independent ingress and TLS controls | Additional deployment, buffering and trust ownership; not needed by this operator | Defer |

[Docker port publishing](https://docs.docker.com/engine/network/port-publishing/)
maps ports using networking rules, rather than authenticating HTTP forwarding
headers. An omitted host address publishes on all host interfaces. Keep the
operator's current port mapping; document an explicit bind address or network
firewall when access needs restricting. Docker/NAT can collapse client
addresses, so quotas describe the server-observed peer rather than proving a
unique external person.

## Design and recommendation stack

1. Keep Python, the direct Docker topology, one worker and current capacity limits.
2. Give the shipped launcher an empty trusted-peer list by default. Disable
   forwarding and pass an explicit empty allowlist, ignoring ambient
   `FORWARDED_ALLOW_IPS`.
3. Support a future intentional proxy through `IMAGE_EMBEDDER_FORWARDED_ALLOW_IPS`
   or `[server].forwarded_allow_ips`, accepting explicit IPv4/IPv6 addresses and
   canonical networks. Reject wildcard trust, `/0`, hostnames, malformed values
   and accidental non-array TOML values before server startup.
4. Validate address quotas and spoofed headers through native parsers and a real
   Docker published port. Refresh bounded CPU/OpenVINO socket measurements;
   preserve the distinction between application RSS and whole-container memory.
5. Measure the operator's real workload, codecs and concurrent clients before
   increasing admission or worker counts. A proxy remains optional future work.

Environment values override TOML, including an empty value disabling trust.
Explicit proxy networks must contain only peers the operator actually controls;
a network allowlist is not a firewall. A separate Uvicorn CLI or alternate ASGI
launcher must configure its own forwarding policy. The API and OpenAPI schema
do not change.

## Outcome

The shipped server and socket-capacity fixture now pass both Uvicorn forwarding
arguments explicitly. A small `forwarding.py` validates configuration; `Settings`
loads the owned TOML/environment setting and validates direct constructors too.
There is no new JavaScript, dependency, API response or proxy service.

Native httptools and h11 checks cover direct, explicitly trusted and explicitly
untrusted peers, both public probes, repeated forwarding fields, quota exhaustion
and scheme preservation. A separate Docker Desktop 29.8.1 published-port smoke
used the locked CPU image, current source overlay, one non-root worker, 4 GiB/no
swap, a disposable key and disabled model warmup. Host-loopback requests to each
probe produced **200, 200, 429, 429** despite rotated/repeated forwarding fields
and ambient wildcard trust. Header-only unauthenticated embedding returned 401;
authenticated model listing returned 200. The real application performed no
model inference in that smoke. Shutdown exited 0 without OOM, and the temporary
container and key file were removed. No remote LAN address-preservation claim is
made from that host-loopback test.

Both real pinned model profiles then passed the existing offline socket scenario:
single/image and 32-item batch ceilings, eight held 16 MiB uploads, header-only
excess 503, seven disconnects and recovered vector parity per stage, zero-byte
unauthenticated 401, oversized-header 413, mixed inline/remote ordering, two
reaped children and settled ingress/queue ownership. CPU and OpenVINO CPU ran
sequentially in fresh locked containers with one worker/owner, cache disabled,
45-second embedding budget, 1/8/32 batches, 974-pixel square batch inputs and
20 ms sampling. Remote fixture images used 10 MiB padded PNGs; the single source
pixel fixture reached 16 million pixels. Full journals remain outside Git;
[the measurement archive](validation/direct-deployment-trust-2026-10-03.json)
retains their hashes, configuration and evidence.

| Profile | Process peak RSS | Kernel cgroup peak | Container limit | Observed unused margin | Sampled task / temp peaks |
|---|---:|---:|---:|---:|---:|
| CPU | 2.94 GiB | 3.57 GiB | 4 GiB | 0.43 GiB / 10.68% | 44 tasks / 10.00 MiB |
| OpenVINO CPU, cached IR | 4.55 GiB | 4.75 GiB | 8 GiB | 3.25 GiB / 40.63% | 27 tasks / 10.00 MiB |

Memory/task limit-event deltas and OOM counts were zero. Native cgroup v1 exposes
no task lifetime peak, so sampled task/temp observations remain lower bounds.
The probe client and sampler share the service process/container. Kernel memory
charging includes more than process RSS and shared page-cache accounting varies;
these margins describe these runs, not guaranteed spare production capacity.
Cached OpenVINO IR does not replace the existing cold-export sizing evidence.
No worker, admission, memory, timeout, dependency lock or coverage floor is tuned.

Full locked QA passed **1,086 tests / seven existing skips** in 173.75 seconds.
Coverage is **95.38% lines / 90.42% branches**, above unchanged 89.42% / 79.37%
floors. The new forwarding validator has full statement/branch coverage. Focused
checks passed 53 cases, including the native trust/parser matrix and existing
quota/socket-capacity regressions. Scoped Ruff/Pyright, copyright, skill metadata
and Markdown links pass. Canonical OpenAPI is byte-identical to the baseline.
All thirteen dependency profiles validate; exact QA/CPU/OpenVINO inventories
and pip checks pass, with all 26 lock artifacts unchanged.

Hash-verified CodeQL 2.27.1 Python security-extended extracted 86 Python files
and found no new findings. Two unchanged setup-tool context results remain:
explicit `--show-key` output in `generate_env.py` and nonsecret 0644 publication
in `secret_publication.py`. Ten workflow files were extracted; this Python query
invocation is not an Actions-language analysis. Verified Gitleaks 8.30.1 found no
secrets in the tracked/nonignored source snapshot or 89-commit history. The source
scan was scoped to that snapshot after an initial checkout-directory scan was
stopped because it also traversed ignored local environments. No rules,
suppressions, permissions or coverage gates were weakened. Final source and
committed history are checked again before push.

The first scan including the final measurement archive flagged three public
content digests: the canonical OpenAPI schema and two source files. Each value
was verified against its hashed bytes. The archive now separates path/kind from
digest fields, retaining the same digest values without adding any allowlist.

No suitable unapplied PR was available among the eight fetched open PRs; none
was selected or merged. The [availability record](open-pr-availability.md) and
[skill design/outcome](project-deployment-transport-skill.md) are separate documents.
Delivery is directly on local/remote `master`, without a branch, release, tag or
version bump.

Next measure representative codecs, concurrent callers and remote-client address
preservation on the actual operator host. The CPU fixture's 10.68% observed
margin is a reason to keep current owner/admission counts until those measurements
establish sufficient headroom. Observe the hosted Windows matrix and remaining
accelerator/platform gates separately.
