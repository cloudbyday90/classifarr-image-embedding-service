# Remote image destination validation

Assessment date: 2026-10-02. Baseline: `3364b0c96f711d3384c06adf663f5717bc0fe05d`. Status: implemented and locally validated. Work stays directly on `master`.

## Evidence and security contract

All remote sources pass through `ImageEmbedder._resolve_image_bytes → _fetch_image_bytes`: direct single requests, explicit batches with/without cache, and coalesced dispatch. The baseline approves the initial URL with `urlparse`, discards resolved addresses, and calls Requests on the original URL. An independent read-only investigator confirmed this shared boundary and its compatibility requirements.

An offline baseline probe used synthetic Requests responses and a socket that records and refuses connections. A public-image control returned its bytes. Three triggers reproduced: public-to-loopback redirect, a backslash/userinfo authority interpreted differently by the validator and Requests, and public DNS approval followed by a loopback connection attempt after a second hostname lookup. No protected destination was contacted.

The invariant is: every connection uses a validated public numeric address from the current hop's complete DNS result, and its Host/TLS identity belongs to that same validated URL. Redirects cannot skip this boundary. Existing default-disabled remote access, optional exact host allowlists, byte/pixel limits, successful embedding schemas, content-derived cache keys, and ordered batch errors must remain.

## Official October 2026 research

URLs were discovered through web search and GitHub MCP, then opened or fetched. [OWASP's SSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) recommends disabling automatic redirects and checking all IPv4/IPv6 DNS answers. [Python's URL parsing guidance](https://docs.python.org/3.12/library/urllib.parse.html?highlight=urlparse) says parsing alone is not validation and notes normalization/removal of some control characters. Reject ambiguous raw authorities before preparation.

[urllib3's supported IP connection/TLS identity APIs](https://urllib3.readthedocs.io/en/stable/advanced-usage.html) permit numeric-IP pools with explicit original-host SNI and certificate checks. Its [pool API](https://urllib3.readthedocs.io/en/stable/reference/urllib3.connectionpool.html) supports streamed bodies and disabling redirects/retries. Use these public APIs rather than patching process-wide DNS or private connection methods. [Python's address classification notes](https://docs.python.org/3.12/whatsnew/3.12.html) document classification fixes; test the actual packaged interpreter, including shared-address and mapped/translated IPv4 forms.

## Alternatives and decision

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Reject every redirect, retain hostname transport | Smallest redirect change | Breaks ordinary CDN redirects; DNS/authority disagreement remains | Incomplete |
| Numeric-address pools with manually validated redirects | One shared boundary; retains public redirects and verified HTTPS; no ambient proxy/netrc routing | Per-fetch pool cost; explicit compatibility changes; finite connection attempts can exclude later working addresses | Adopt |
| Dedicated egress proxy/worker with network policy | Stronger deployment isolation and centrally controlled routing | Additional service, operations, and trust boundary | Complementary future option |

Keep small `remote_url` and `remote_fetch` modules. Prepare URLs without a Session, reject raw controls/backslashes and credentials, normalize the authority, and apply host/address rules to the actual canonical destination. Reject empty or mixed unsafe DNS sets; preserve HTTP/HTTPS and valid custom ports. Public-address policy also rejects multicast, reserved, scoped, CGNAT, and embedded private IPv4 destinations.

Use fresh direct pools for at most four validated addresses, alternating address families while preserving resolver preference within each family. Fallback is allowed only for failure before connecting; TLS, read and HTTP failures are not retried. For HTTPS require CA verification and the original hostname for SNI/certificate identity. Send only the canonical origin-form path/query and explicit Host; ignore ambient proxies, netrc, cookies, and request headers. Use the installed certifi trust bundle, preserving normal public HTTPS verification. Do not inherit Requests CA-bundle environment overrides.

Disable automatic redirects and retries. Follow at most three redirects, revalidating each hop (including same-host, relative, and scheme-relative transitions), reject HTTPS-to-HTTP downgrade, and close redirect bodies without draining them. Stream the final decoded body under the existing `MAX_IMAGE_BYTES` ceiling. Close responses and pools on every exit. Policy failures use existing ValueError/400 handling and limits retain InputLimitExceeded/413. HTTP/network/TLS failures use sanitized RemoteFetchError/500, preserving the prior download-failure status. Suppress their original exception chains so tracebacks do not expose URLs/queries; explicit batches retain sanitized ordered per-item errors.

Compatibility changes are deliberate: credentialed/ambiguous URLs, unsafe address results, downgrade redirects, and chains beyond three are refused. Configure canonical ASCII/punycode allowlist hosts. Environment proxy/netrc/CA override behavior is removed from remote-image fetching. This is destination validation, not an OS network firewall or a total DNS/download wall-clock deadline.

## Implementation, verification and outcome

Small `remote_url.py` and `remote_fetch.py` services replace the embedder's transport implementation, leaving thin compatibility helpers and all shared entry-point ownership intact. urllib3 and certifi become explicit runtime dependencies at the verified installed-version floors. Test-only [trustme](https://pypi.org/project/trustme/1.2.1/) generates ephemeral CA/server certificates; no private keys or production-model downloads are committed.

The security fix outcome is **fixed** for the reproduced parsing, redirect and DNS-to-connection paths. The baseline public control succeeded while the three probes recorded protected targets or attempts. New regressions refuse the ambiguous authority before any pool, protected redirects before a protected connection, and same-host DNS changes before another pool; actual transport resolves the original hostname once, then connects to its approved numeric address. Legitimate public HTTP/HTTPS, custom ports, encoded paths/queries, exact allowlists and relative/cross-host public redirects continue to work.

The mandatory fresh read-only investigator and later fresh bypass reviewer followed the installed `codex-security:fix-finding` skill. The reviewer independently passed 95 focused tests and additional cache-enabled/disabled API checks. No protected-address bypass was established. Its actual HTTP/1.1 closure probe observed 20 undelivered large response sockets close immediately, with descriptors unchanged (5 → 5). It found two compatibility issues in the initial patch: download errors changed 500 → 400, and four leading IPv6 addresses could exclude IPv4. The final patch restores 500 with sanitized tracebacks and interleaves families under the same four-attempt budget. Author regressions verify both corrections; the independent review preceded those final corrections.

Final focused validation passed **126 tests** after compilation/scoped Pyright and Ruff checks. This includes real TLS success, wrong-host/untrusted-CA refusal, preserved SNI/Host, numeric connection recording, ambient environment isolation, gzip expansion ceilings, alternate address encodings, redirect budgets, all dispatch modes, error/log sanitization and resource closure. DNS/connection tests are synthetic or use an owned local-server seam inside a network-disabled container; production has no private-address exception. Current complete-suite, coverage, image and audit results are recorded in the [recommendation stack](recommendation-stack.md).

Reproducible checks: run `python -m compileall -q src tests`; scoped `pyright` with the packaged optional-backend types; `ruff check --select E,F,I --ignore E501` on the new modules/tests; `pytest tests/test_remote_destinations.py tests/test_remote_native_transport.py tests/test_remote_api_boundary.py tests/test_remote_fetch_and_security.py tests/test_embedder.py tests/test_coverage_gaps.py --no-cov`; then `pytest`, `python scripts/check_coverage_ratchet.py`, `python scripts/check_copyright.py`, and `actionlint -shellcheck= -pyflakes=`. Tests run against the Python 3.12 QA image with a read-only repository mount and external networking disabled. Build the default CPU Dockerfile and execute its shipped `scripts/smoke_backend.py --backend cpu`; audit installed site-packages using `pip-audit --strict -s osv --path ...` without suppression.

Remaining limits: DNS calls and trickled downloads do not have a total wall-clock deadline; each socket attempt/hop can consume its own timeout. Four attempts can still omit a working later address. Public-address policy is not a deployment firewall and cannot control unusual OS routing or externally configured translation. No real external CDN/production-weight fetch, Intel GPU or new all-backend rebuild was performed here. Prior backend evidence is separate. Retain deployment egress controls and address total worker lifetime as a follow-up; no full repository audit is claimed.

## Next recommendation

The subsequent [aggregate ingress iteration](aggregate-ingress.md) bounds HTTP retention and adds deployment worker/memory controls. The [production model/artifact iteration](model-artifact-contracts.md) now implements pinned preprocessing fixtures and versioned IR contracts with native peak measurements. Next calibrate maximum-batch/multi-model RSS/VRAM capacity and complete dependency locks; total remote DNS/download deadlines remain a separate design task. This record retains its earlier remote-boundary evidence. No release or upstream PR merge is created.
