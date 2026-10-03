# HTTPX2 API test-client migration

Assessment: 2026-10-03. Baseline: `2f19545409758ac524e4c3703e11ea55b8e26524`. Work is performed directly on master.

## Problem and recommendation

The current Starlette 1.7.0 TestClient warns when it falls back to HTTPX. QA already locks FastAPI 0.142.2 and Starlette 1.7.0, so the next change is the client environment and authored API tests, not a server framework upgrade. Adopt HTTPX2 2.13.1 for synchronous TestClient and asynchronous API tests. Keep the separate legacy HTTPX path required by Hugging Face and shipped production/capacity probes; those paths keep their current complete graphs and response classes.

Add HTTPX2 to the complete native Linux QA graph while constraining every existing reviewed distribution. Resolve new transitive wheels with the reviewed pip on the actual CPython 3.12 target, validate approved origins/hashes and compare the resulting graph before installation. A Transformers floor update is separate from this QA graph extension. No Windows graph or production runtime refresh is implied.

Rename authored asynchronous API test imports explicitly, with no global module alias or fallback shim. A small test helper combines LifespanManager's state-aware ASGI wrapper with an HTTPX2 AsyncClient and explicit environment isolation. Use it in shared batch fixtures and the full authenticated request-cycle test. Other tests retain deliberate explicit/raw lifecycle boundaries. Keep capacity-probe tests paired with their legacy HTTPX probe module. New regressions establish actual response class selection, lifespan state/startup/shutdown and success/error cleanup. Fail the specific Starlette legacy-client warning so missing HTTPX2 cannot silently restore the deprecated path.

## Official research and tradeoffs

URLs were discovered through web search/fetch and GitHub MCP on 2026-10-03:

- [HTTPX2 publisher package metadata](https://pypi.org/project/httpx2/) identifies stable 2.13.1, published September 23, and wheel SHA-256 `6dff50fabc270ee5fd25d845d0b078ed20564579744d6d962850975996d2f9a4`.
- [Publisher migration guide](https://pydantic.dev/docs/httpx2/get-started/migration/) documents compatible client APIs, distinct HTTPX/HTTPX2 classes, renamed loggers/User-Agent and OS trust-store verification. Compatibility is a starting hypothesis to test, not proof that mixed class objects are interchangeable.
- [Starlette release notes](https://starlette.dev/release-notes/) and [official TestClient source](https://github.com/Kludex/starlette/blob/main/starlette/testclient.py) establish HTTPX2 preference and the deprecated fallback. The actual installed 1.7.0 wheel source was checked locally before changes.
- [HTTPX2 ASGI transport guidance](https://pydantic.dev/docs/httpx2/advanced/transports/) separates ASGI transport from application lifespan and recommends LifespanManager. Preserve state-aware startup/teardown and app exception behavior.
- [Starlette TestClient guidance](https://starlette.dev/testclient/) requires a context manager when synchronous tests claim lifespan execution. Existing raw request-only tests are not reclassified as lifespan evidence.
- [HTTPX2 timeout guidance](https://pydantic.dev/docs/httpx2/advanced/timeouts/) describes network inactivity budgets, which do not establish a whole in-process ASGI deadline or native worker interruption.
- [pytest warning policy](https://docs.pytest.org/en/latest/how-to/capture-warnings.html) supports narrowly failing a known deprecation instead of suppressing it.
- New [HTTPcore2](https://pypi.org/project/httpcore2/) 2.13.1 and [truststore](https://pypi.org/project/truststore/) 0.10.4 wheel hashes were compared with publisher metadata; no optional HTTP/2, proxy or CLI extras were added.

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Suppress the fallback warning | Tiny diff | Retains deprecated client selection | Reject |
| Replace every HTTPX dependency, including SDK/probe paths | One client name | SDK requires HTTPX; would change production graphs and transport scope | Defer until upstream compatibility and a separate design |
| Migrate API tests with an additive constrained QA graph | Current TestClient, explicit classes and lifecycle checks | Both libraries remain for distinct owners | Adopt |

Final stack: retained Python/FastAPI/Starlette/AnyIO service; HTTPX2 for API tests; state-aware managed ASGI fixture; targeted warning gate; complete reviewed QA wheel graph; independent existing SDK/probe/runtime locks; strict advisory/inventory and resource-contract validation.

## Outcome

Native Linux CPython 3.12 resolution with reviewed pip 26.2.1 produced **70 QA
wheels**. All 67 baseline artifact records are identical; only HTTPX2 2.13.1,
HTTPcore2 2.13.1 and truststore 0.10.4 were added. The complete graph installed
with required hashes in a separate image derived from the immutable old QA image,
then passed exact inventory and pip checks. All twelve other profile artifact
graphs are unchanged; five native runtime images passed renewed input/inventory
checks, including emulated ARM. Windows graph inputs and artifacts are unchanged.

The old QA image failed API collection with the specific Starlette warning, as
expected. The new image passed **67 focused cases**, including six new response
class, state, environment-isolation, exception and cleanup contracts. Full QA
passed **953 tests / seven existing skips**: two opt-in model tests and five
Windows-only setup cases. No legacy-client warning was emitted. Coverage remains
**95.22% lines / 90.12% branches**, above unchanged **89.42% / 79.37%** floors.
An existing default-image-size test in the migrated suite previously accepted any
exception from broken fakes; it now asserts successful embedding, canonical size
and output dimensions.

The isolated reviewed 29-package auditor passed exact inventory/pip checks; strict
OSV audited all 70 installed QA packages with zero known vulnerabilities or skips.
Configured native Gitleaks 8.30.1 source and 85-commit history scans passed with
zero findings. Two public source-digest fields initially matched the generic
API-key rule; separate filename/digest fields resolved the matches without any
scanner-policy or allowlist change. Ruff passed all thirteen changed/new Python files, and scoped Pyright passed the
new helper and its regression module. Copyright and skill validation passed.
The [validation record](validation/httpx2-test-clients-2026-10-03.json) contains
immutable image identities, graph deltas, commands, coverage and secret-scan
outcomes. Runtime Python and workflow source are unchanged, so this iteration does
not claim a new production-code or workflow security scan.

In-process ASGI results do not establish network/TLS/proxy behavior or GPU
performance. Ambient TLS/proxy isolation is tested on the authored async helper;
Starlette's synchronous TestClient exposes its own constructor policy. Hosted
Windows execution and real socket-upload capacity remain separate evidence gates.
