# Reliability and security recommendation stack

Assessment date: 2026-10-03. The current quota-identity/skill iteration starts from `65597eaa3e1e925abcecb02b3a1b6f22727cb455` and is delivered directly on `master`. Python is retained by user decision. Recommendations combine local code evidence, actual build/runtime measurements and official sources discovered through web/GitHub MCP, linked in the separate design records.

## Completed recommendations and tradeoffs

| Recommendation | Pros | Cons or limits | Outcome |
|---|---|---|---|
| Stable public-probe and verified embedding quota identities | Closes rotated/empty-header bypasses; removes raw credentials from storage/logs | NAT clients share probe limits; a shared service credential shares embedding quota | Implemented; [identity design](public-probe-rate-limits.md), [skill extension](project-quota-identity-skill.md) |
| Track inference owners beyond HTTP deadlines; preserve server signals and drain work | Preserves accounting during cancellation; cleanup cannot race workers | Running Python threads cannot be forcibly killed; supervisor deadlines must align | Implemented; [execution design](inference-execution.md) |
| Shared FIFO admission for batch-window members and explicit batches | One waiting budget; prompt cancellation and live-client dispatch | Separate byte/pixel limits still matter | Implemented; [admission design](bounded-batch-admission.md) |
| Streamed HTTP and modular inline/image/batch budgets | Rejects excess before queue retention or RGB/resize expansion; predictable closure | 413 behavior; budgets are not exact RSS quotas | Implemented; [input design](input-memory-budgets.md) |
| Approved numeric-address connections and validated redirects/TLS | Closes reproduced authority, redirect and DNS-to-connection bypasses | Per-hop timeouts do not bound total DNS/download lifetime | Implemented; [remote design](remote-image-destinations.md) |
| Complete-lifetime HTTP ingress and explicit launcher/Compose budgets | Rejects excess before body receive; probes retain access; workers cannot multiply implicitly | Per-worker budgets; slow clients/native memory need sizing | Implemented; [ingress design](aggregate-ingress.md) |
| Exact backend profiles, base digests and isolated vendor selection | Detects substitution; keeps CUDA libraries out of CPU builds; modern/legacy NVIDIA options | Apt is separately refreshed; driver/hardware gates matter | Implemented; [backend design](backend-build-recommendation.md) |
| Immutable publisher models, verified assets and explicit PIL preprocessing | Detects drift/corruption before deserialization; stable dimensions and math | Hashing and reviewed model updates; base publisher supplies restricted PyTorch weights | Implemented; [model contracts](model-artifact-contracts.md) |
| Versioned, process-locked atomic XML/BIN publication and explicit precision | Rejects stale/partial/corrupt IR; first use/reload share the saved representation | FP32 disk/export memory; checksums do not authenticate a malicious local writer | Implemented with four-process contention and actual production reload checks |
| Production-weight probes and separate CPU/OpenVINO CI | Real single/batch/API parity, cache reuse and peak RSS evidence | Large downloads; accelerator capacity and hosted CI remain separate | Prior production contracts passed; current maximum-batch checks supplement them |
| 8 GiB/no-swap OpenVINO containment | Accommodates measured cold export and both resident models | Larger host budget; observations do not establish every workload's capacity | Current guarded cold/multi-model probe peaked at 6.31 GiB process RSS; CPU/CUDA retain 4 GiB defaults |
| Modular bounded capacity probes and manual calibration CI | Process/cgroup evidence for batches 1/8/32, both models, cold/warm loads and detached owners | Shared page-cache charging and host load affect results; accelerator hardware remains separate | Implemented; [capacity design and measurements](capacity-calibration.md) |
| Independent finite embedding deadline, initially 45 seconds | Adds measured headroom without lengthening remote hops; preserves explicit legacy budgets | Longer ingress occupancy; client/proxy alignment and slower hosts still matter | Implemented after baseline 15-second and trial 30-second CPU timeouts; [deadline design](embedding-deadlines.md) |
| Process-wide cold initialization guard with warm-cache fast path | Resolves reproduced concurrent import failure across aliases/owners; serializes export peaks | Cold callers wait; stuck native initialization still needs containment | Native cold overlap replay and event-controlled regressions passed; [initialization design](model-initialization.md) |
| Strict OSV audits of installed environments | Covers vendor Torch and packaged tooling without advisory suppression | Current public metadata is needed; not proof of complete security | CPU/OpenVINO/QA audits passed with no skips |
| Complete OpenVINO environment copy and duplicate-inventory guard | Removes reproduced stale NumPy metadata; aligns imports and audits | Extra runtime build layer; native inventory checks remain necessary | Implemented; [runtime environment design](openvino-runtime-environment.md) |
| Complete target-specific wheel hashes and exact environment inventories | Prevents transitive/artifact drift; no mixed-index installation or source builds | Native per-target generation, artifact availability and reviewed refresh cost | Implemented; [dependency design](dependency-locks.md) |
| Reviewed weekly refresh and prompt advisory response | Keeps frozen contracts auditable while allowing security updates | Maintainer review; apt and host drivers are separate contracts | Implemented; [update policy](runtime-update-policy.md) |
| Reviewed CI action, native tool and builder contracts | Freezes reviewed source/bytes; removes PR-write credentials and unchecked tool caches | Owned OSV workflow and deliberate refresh; hosted platform remains a trust boundary | Implemented; [execution](ci-execution-contracts.md), [native tools](ci-native-artifacts.md), [cleanup](release-cleanup.md) |
| Private atomic secret setup and explicit new-key disclosure | Owner-only permissions before writes; complete-file publication; default console confidentiality | Recoverable local credential, trusted checkout and operator crash cleanup remain necessary | Implemented with native Linux/Windows checks; [setup design](secret-setup.md) |
| Cooperative total upload deadline and supervised remote-fetch processes | Bounds stalled uploads, DNS and trickled responses without abandoning owners | Per-fetch startup/RSS, temporary storage and OS cleanup latency | Implemented; [design, alternatives and native outcomes](request-lifetimes.md) |
| Total HTTP/1 response deadline with terminal buffer drain and abort-before-cancellation | Prevents slow readers and stalled producers retaining ingress; preserves successful responses | Pinned Uvicorn/CPython adapter; explicit asyncio; kernel receipt and background tasks are separate | Implemented; [design and native outcomes](response-send-lifetimes.md) |
| Shared elapsed remote batch fetch deadline | Bounds cumulative sequential remote work; preserves ordered partial success and current-byte cache identity | Later remote inputs may fail; OS cleanup latency and native inference remain separate | Implemented; [design and outcomes](remote-batch-budget.md) |
| Repository resource-contract AI skill | Focused repeatable ownership and validation routing, versioned with source | Manual scenario evaluation; not an independent model-discovery benchmark | Implemented; [skill design](project-resource-skill.md) |
| Local Trivy PR 27 and FastAPI PR 46 adoption | Reviewed action/floor updates without original PR merges | Hosted action orchestration is separate; FastAPI wheel graphs remain unchanged | [PR 27](pr-27-local-validation.md), [PR 46](pr-46-local-validation.md) |
| Selected PRs implemented locally through verified immutable refs | Requested maintenance updates without upstream merges | Hosted reporting/cache behavior remains CI work | [PR 28](pr-28-local-validation.md), [42](pr-42-local-validation.md), [40](pr-40-local-validation.md), [33](pr-33-local-validation.md), [49](pr-49-local-validation.md), [51](pr-51-local-validation.md), [41](pr-41-local-validation.md), [45](pr-45-local-validation.md), [PR 50](pr-50-local-validation.md), [PR 54](pr-54-local-validation.md), random [PR 34](pr-34-local-validation.md), and now [PR 52](pr-52-local-validation.md) |
| Task/storage/CUDA capacity extension and focused calibration skill | Real mixed inputs, child/owner settlement and accelerator allocation evidence | Shared host and sampled peaks remain workload-specific; socket/proxy buffering is separate | Implemented; [capacity](deployment-capacity.md), [skill](project-capacity-skill.md), random [NumPy PR 48](pr-48-local-validation.md) |
| Recurring Windows ACL/process/socket matrix and native-validation skill | Current framework graph, mandatory execution and complete OSV inventory gates | Separate target locks; hosted Server 2025/latest patches remain unobserved locally | Implemented; [Windows outcomes](windows-native-validation.md), [skill](project-native-validation-skill.md), random [renewed PR 46 floor](pr-46-floor-followup.md) |
| HTTPX2 API test clients and dependency-migration skill | Current Starlette client, explicit lifecycle/state and constrained QA artifacts | SDK/probe clients remain separate; ASGI tests do not prove socket/TLS behavior | Implemented; [migration](httpx2-test-clients.md), [skill](project-dependency-migration-skill.md), random [PR 53](pr-53-local-validation.md) |
| Real socket uploads and focused calibration skill reference | Server-observed body pressure, excess rejection, disconnect settlement and recovered real vectors | Shared client overhead; direct HTTP/1, padded PNGs and host-specific observations | Implemented on CPU/OpenVINO; [socket outcome](socket-upload-capacity.md), [skill](project-socket-capacity-skill.md); [no suitable unapplied PR](open-pr-availability.md) |
| API-key rejection before body receipt and resource-skill reference | Removes observed unauthenticated 16 MiB receipt; mandatory prefixed admin policy; native recovery and stable schema | Successful requests retain two shared-policy checks; HTTP/1 errors close unread connections; outer limits can reject first | Implemented; [auth outcome](early-api-key-authentication.md), [skill](project-early-auth-skill.md); no suitable unapplied PR |
| Retain modular Python and measure before language changes | Builds on native inference and tested contracts | Python footprint remains; Rust benefit is unmeasured here | User confirmed; [language decision](language-platform-decision.md) |

## Next items

| Priority | Recommendation | Pros | Cons or decisions needed |
|---|---|---|---|
| Next 1 | Measure actual reverse-proxy request buffering and target workload headroom | Covers another body/temp-storage owner before concurrency tuning | Requires the operator configuration, TLS/HTTP version, trusted peers, NAT traffic and hardware evidence |
| Next 2 | Observe hosted Windows jobs; extend target-hardware capacity and restore CUDA ARM after upstream metadata repair | Confirms actual hosted platforms and broader deployment headroom | Runner/hardware cost; cuSPARSELt ARM metadata currently fails strict pip check |

The [quota identity fix](public-probe-rate-limits.md) closes rotating-header and empty-Bearer bypasses while retaining public access and separate counters. Next measure actual reverse-proxy buffering, effective-address trust and operator workload headroom before changing limits or owner counts. Use the [capacity-calibration skill](project-capacity-skill.md) for sizing and the [resource-contract skill](project-resource-skill.md) if measurements reveal lifetime changes. The [Windows matrix](windows-native-validation.md) still requires observed hosted execution before claiming Server 2025/latest-patch results. Keep target-hardware gates and the upstream CUDA ARM metadata gate.

## Final recommendation stack

Keep Python/FastAPI/AnyIO with small authenticated-router, credential, quota-identity, ingress, admission, queue, execution, input, remote transport and model/artifact services. Keep public health/readiness and documentation behavior, explicit mandatory admin protection and constant-time byte comparison. Public probes use effective addresses; protected quotas use one verified nonsecret service principal or address fallback. Preserve separate counters, application-owned storage and server-owned proxy trust. Authentication remains within matched routing, before body parsing and inside existing ingress/response ownership. Native tensor libraries perform inference. A later language experiment must show measured benefit and equivalent model/API/backend contracts. Authored JavaScript uses ESM. The external CommonJS Gitleaks action accepted in PR 41 is now replaced by verified native scanning through small Python helpers.

Keep a per-invocation 30-second shared remote fetch-phase budget, beginning at the first remote input and clipped by each 15-second image budget. Preserve ordered partial success, current-byte cache identity, inline results and independent model execution. After shared expiry, reap the current worker before returning and refuse later remote work without spawning. Retain the reviewed FastAPI 0.142.2 graphs with PR 46's renewed 0.142.2 minimum. Use HTTPX2 for API tests with explicit lifespan state and a targeted fallback gate, while keeping the SDK and bundled probes on separately reviewed HTTPX graphs. Preserve complete native wheel/hash installation and strict advisory audits. Keep focused instruction-only project skills discoverable under `.agents/skills`, including owner-first dependency migration, with change-scoped native validation and no new external-action authority.

Generate shared service credentials with explicit 32-byte entropy. Publish private complete `.env` files atomically on an owned/trusted checkout, preserve existing config, and disclose new keys only with an explicit display option. Keep nonsecret defaults readable by the container. Use managed secret rotation only as a separately designed deployment migration.

Keep CPU-only Torch by default, exact cu130/cu126 profiles for appropriate NVIDIA hardware, and matching OpenVINO bindings/base for Intel deployments. Use immutable approved model revisions/digests, restricted local weight loading, explicit PIL preprocessing and no remote model Python. Maintain the process-owned IR contract, complete manifest, finite locks, atomic publication, FP32 serialization and accuracy execution policy on application-owned local volumes.

Retain one worker, eight complete HTTP ingress owners, server concurrency 64 and backlog 128. Keep 16 MiB bodies, 10 MiB compressed images, 32 MiB batch bytes, 16-million source/resize pixels and 32-million aggregate pixels. Contain CPU/CUDA at the tunable 4 GiB/no-swap default and OpenVINO at 8 GiB/no swap. Serialize cold initialization within each process and keep cached inference direct. Set a positive finite embedding deadline, initially 45 seconds, with legacy override compatibility and a separate 15-second remote-hop default. Align client/proxy and supervisor budgets. Preserve ownership after HTTP timeout and deterministic image closure; GPU VRAM and every workload's headroom remain separate measurements.

Use a total 30-second upload budget that stops at complete body/disconnect/response start and preserves ingress through cooperative cleanup and JSON 408 completion. Give every enabled remote fetch a separate total 15-second parent-owned budget, including interpreter startup, DNS, TLS, all hops/addresses and decoded reads. Keep the socket/hop budget independent. Kill/reap before the inference owner releases capacity; keep bounded non-executable messages and private auto-deleted image storage. Measured three-byte fetches add roughly 264 ms and 28-31 MiB child RSS here; this is accepted for termination of blocked synchronous DNS/network work, with deployment calibration required.

Use a separate total 30-second HTTP/1 response-send budget through the shipped asyncio launcher. Include terminal user-space and TLS ciphertext buffer drain; abort and confirm connection loss before cancellation releases ingress. Keep httptools/h11 parser selection automatic. Retest the narrow Uvicorn/CPython adapter on upgrades, use asyncio for TLS, and keep edge/HTTP2/WebSocket/background-work contracts explicit. Adopt the immutable Trivy v0.36.0 wrapper with the verified native binary selected by absolute path, disabled setup/cache and existing report/failure gates.

Keep the reviewed NumPy 2.5.3 graphs with the adopted PR 48 minimum. Report cgroup tasks and temporary filesystem availability independently of process RSS, and CUDA live/reserved allocator peaks independently of device-wide availability. The manual probe uses generous 1024-task/128-MiB tmpfs experiment bounds; do not copy them into production without cold/warm target measurements and headroom.

## Quota identity delivery outcome

Full locked QA passes **1,051 tests / seven existing skips**, including 29 new quota
cases and eight native parser/proxy checks. Coverage is **95.32% lines / 90.27%
branches**, above unchanged floors; both new service modules are fully covered.
Rotated public credentials now exhaust one address identity, and empty Bearer
requests are charged in both public and dev routes. Verified-key traffic, public
access, endpoint scopes, expiry recovery and early auth remain intact.

OpenAPI is byte-identical. All thirteen profiles and the exact 70-wheel QA
inventory pass with unchanged lock bytes. CodeQL Python covers 85 files with no
new findings and two unchanged setup-publication context results. Fresh verified
Gitleaks source and 88-commit baseline history scans find no secrets. Independent
source review found no concrete surviving bypass/regression; execution used the
locked image because local venvs are stale. The [identity design](public-probe-rate-limits.md)
and [skill design](project-quota-identity-skill.md) retain official October research,
tradeoffs, validation and honest native/deployment limits. No suitable unapplied
PR is available; delivery uses master without a release.

## Early authentication delivery outcome

Full locked QA passes **1,022 tests / seven existing skips**, including 59 new
policy/socket cases, with **95.31% line / 90.30% branch** coverage above unchanged
floors. Native header-only httptools/h11 and TLS checks observe no body bytes or
interim 100; correct requests recover. Real CPU/OpenVINO socket replays preserve
vector parity, queued ownership, disconnect cleanup and child reaping while now
requiring zero-byte authentication refusal. Canonical OpenAPI is byte-identical.
The final full suite runs after model/scanner work completes; initial shared-host
startup timing failures and the unchanged isolated follow-up are archived.

CodeQL Python has no new findings and two unchanged setup-publication context
results. Verified Gitleaks source and 87-commit baseline history scans find no
secrets. All thirteen locks remain valid and unchanged; CPU/OpenVINO/QA inventories
and `pip check` pass. The [auth design/outcome](early-api-key-authentication.md),
[skill extension](project-early-auth-skill.md) and [archive](validation/early-api-key-authentication-2026-10-03.json)
retain research, pros/cons, source hashes, cases and limitations. The [separate
public probe quota proposal](public-probe-rate-limits.md) records the newly reproduced
next item. PR availability is still empty of suitable unapplied changes; master
contains this work without a release or version bump.

## Real socket delivery outcome

Both native CPU/OpenVINO profiles pass held-upload, header-only excess/overflow,
disconnect recovery, ordered real vectors, mixed queueing and child-reaping gates.
All eight ingress slots receive 16 MiB minus one byte before completion. Sampled
socket-phase RSS reaches 2.94 GiB CPU / 4.55 GiB OpenVINO; these combined
client/server/model observations leave actual proxy and target-hardware gates.
Full QA passes **963 tests / seven existing skips**, with unchanged **95.22% lines /
90.12% branches** and all thirteen locks unchanged. CodeQL has no new findings;
verified Gitleaks source/history scans find no secrets. Separate [design and
outcome](socket-upload-capacity.md), [skill extension](project-socket-capacity-skill.md)
and [archive](validation/socket-upload-capacity-2026-10-03.json) retain evidence.
The [PR availability review](open-pr-availability.md) records no suitable unapplied
PR, following the user's instruction to continue without an adoption.

## HTTPX2 migration outcome

Full QA passed **953 tests / seven existing optional/platform skips**, with no legacy-client warning. Coverage remains **95.22% lines / 90.12% branches** above unchanged floors. Six focused client contracts and the old-image fallback rejection passed; the migrated suite now asserts a successful default-size embedding instead of accepting an arbitrary exception. Native resolution/install added only HTTPX2 2.13.1, HTTPcore2 2.13.1 and truststore 0.10.4 to the complete **70-wheel QA graph**. Its strict installed-package OSV audit reports no known vulnerabilities/skips, using an exact reviewed 29-package auditor.

All 67 baseline QA records and twelve other profile artifact graphs are unchanged. Six native input/inventory checks passed; Transformers remains 5.18.0 after locally adopting PR 53's 5.17.0 declared floor. The separate [migration](httpx2-test-clients.md), [PR](pr-53-local-validation.md), [skill](project-dependency-migration-skill.md) and [validation](validation/httpx2-test-clients-2026-10-03.json) documents retain research, pros/cons, outcomes and limits. Production transport/TLS/GPU behavior and hosted OS execution are not established by this test-client change.

## Native Windows delivery outcome

Two fresh native Windows 11 x64 environments pass **102 cases each**, with sixteen explicitly permitted setup skips and eighteen uvloop deselections. Both current 33-wheel framework graphs pass exact inventory/pip checks and complete strict OSV audits with zero known vulnerabilities/skips. Full locked Linux QA passes **947 tests**, seven existing skips and one existing HTTPX2 warning; coverage remains **95.22% / 90.12%**, above unchanged floors. Native gate/target/audit policy controls pass nineteen cases.

The [Windows design](windows-native-validation.md), [native-validation skill](project-native-validation-skill.md), random [renewed PR 46](pr-46-floor-followup.md) and [archive](validation/windows-native-2026-10-03.json) separate local evidence from hosted Server 2025/latest patches and model/device contracts. All thirteen locks validate; nine earlier Linux graphs retain their 446 artifacts. Delivery remains on master without original PR merge, branch, release, tag or version bump.

## Response-send delivery outcome

Full locked Linux QA passes **888 tests**, with seven existing skips and **56 added cases**. Coverage is **95.15% lines / 89.91% branches**, above unchanged floors. Native real httptools/h11 sockets cover asyncio/uvloop, TLS, slow readers, terminal writes, trickled/gapped bodies, disconnects, FastAPI request-resource cleanup during Starlette streaming and capacity recovery. Native Windows passes **31 cases** with eighteen expected uvloop skips and its separate older host framework graph. Actual two-worker launcher probes pass in the existing locked CPU/OpenVINO/CUDA images. Exact platform checks are recorded in the [response design](response-send-lifetimes.md) and [validation archive](validation/response-send-lifetimes-2026-10-03.json).

Random [PR 27](pr-27-local-validation.md) is adopted locally without merging its upstream PR. Actual fetched composite Bash steps execute verified Trivy 0.69.3: deliberate secret, vulnerability and configuration fixtures produce valid reports and failing gates; the checkout-equivalent repository has zero selected findings. Shell-escaped inputs and cleanup on failure pass. Native CodeQL Actions reports zero alerts; Python retains the two contextual setup-helper alerts without suppression. Unreleased is updated; no branch, release, tag or version bump is created.

Keep remote URLs disabled unless needed and prefer explicit allowlists. Use approved public numeric destinations, verified original-host TLS identity, at most three validated redirects and four bounded address attempts. Preserve authority/DNS policy, response closure, byte limits, sanitized errors and isolation from ambient transport credentials.

Validate small offline fixtures in routine pytest, real production models in the separate change-scoped/weekly CPU/OpenVINO workflow, capacity through the manual calibration workflow, installed packages and isolated tooling with strict OSV, and device behavior on appropriate hardware. Native workflows perform verified networked prefetch followed by offline containers. Install complete target-specific binary-wheel/hash locks through isolated no-index pip, validate input contracts and exact inventories, and follow the reviewed weekly/urgent update policy. Keep supported digest-pinned bases; Python locks do not freeze apt or host drivers. Pin action source commits and scanner/builder image digests, verify reviewed native tool hashes before execution, deny credentials by default and preserve complete-report gates and conservative tag-only retention. Review publisher provenance and hosted integration separately.

## Input lifetime delivery outcome

Full locked Linux QA passes **832 tests**, with seven platform/optional skips; native Windows process/network validation passes **41 cases**. The 74 new cases cover uploads, cleanup, protocol, native DNS/HTTP/TLS, detached owners and configuration. Coverage is **94.65% lines / 89.34% branches**, above unchanged floors. Existing CPU/OpenVINO/CUDA image probes retain native Torch in the parent and pass approved controlled HTTP, real process termination and unmodified private-destination refusals. Their 0.7-second DNS budgets settle around 0.702-0.703 seconds, and every child is reaped.

Random **PR 52** raises pytest's minimum to the already locked **9.1.1** release. All 67 QA wheel records/hashes and lock bytes remain unchanged; all nine input contracts and the exact QA inventory pass. Native CodeQL retains only two contextual setup alerts without suppression; Ruff, scoped Pyright, copyright, coverage, secret and whitespace checks pass. The separate [lifetime](request-lifetimes.md), [PR 52](pr-52-local-validation.md) and [native archive](validation/request-lifetimes-2026-10-03.json) documents contain design, official October 2026 research, pros/cons, results and limits. Delivery stays on master with Unreleased changes.

## Secret setup delivery outcome

Private same-directory staging establishes POSIX `0600` or a protected current-user Windows DACL before writing. Exclusive publication and a setup lock prevent cooperating creation/rotation races; explicit rotation replaces complete `.env` data while preserving existing defaults. Default CLI/launcher output omits the key; `--show-key` is an explicit disclosure of a newly generated key. Nonsecret POSIX config remains `0644` for non-root container access. Existing keys are neither read nor silently repaired, and crash leftovers remain private/gitignored until operator inspection.

Full locked QA passes **758 tests**, with seven platform/optional skips; native Windows setup passes **32 tests** with sixteen POSIX/unavailable-symlink skips. The 48 new setup cases and a CI policy regression cover real publication contention, cross-process lock refusal, pre-write ACL/mode checks, rotations, failures, unsafe links and actual launcher fixtures. A different Linux UID cannot read the key but can read config. Application coverage stays **94.31% / 88.71%**, and the unchanged 67-package QA inventory validates PR 34's renewed input contract.

Actual CodeQL security-extended analysis retains two contextual alerts: explicit `--show-key` output and world-readable nonsecret Docker config defaults. Neither is suppressed. Its previous plaintext-storage alert is absent, but the key remains intentionally recoverable plaintext in a private file. The separate [setup](secret-setup.md), [historical digest classification](secret-scan-digests.md), random [PR 34](pr-34-local-validation.md) and [validation archive](validation/secret-setup-2026-10-03.json) records explain official October 2026 research, tradeoffs, precise evidence and limits. No upstream PR merge, branch, release, tag or version bump is created.

## CI delivery outcome

Repository-controlled action refs and nested OSV/builder images now have reviewed immutable identities. Four native tool downloads are verified before publication; checkout credentials are not persisted, Gitleaks loses PR-write access, copyright receives contents-read, OSV fails missing/no-lockfile scans, and Docker Hub cleanup keeps bearer credentials in memory while validating the complete retention inventory.

Full QA passes **714 tests**, including **89** new CI artifact/credential/retention/report-policy cases, with two optional production cases skipped. Coverage remains **94.31% / 88.71%**. Six unchanged native environments pass the renewed Requests input contract. Gitleaks clean/leak/error controls and 78-commit history checks, Trivy selected filesystem/configuration scans, OSV differential/full/no-lockfile gates and verified tool/builder version executions pass. Actual CodeQL Actions analysis has zero findings; Python reports two existing setup-helper alerts, which inform Next 1 above.

The separate [execution](ci-execution-contracts.md), [native artifact](ci-native-artifacts.md), [cleanup](release-cleanup.md), random [PR 54](pr-54-local-validation.md) and [validation](validation/ci-contracts-2026-10-03.json) records describe official October 2026 research, alternatives, pros/cons and outcomes. Hosted cache/upload/SARIF and release credentials/privileged setup remain integration gates. No external deletion, upstream PR merge, release, tag or branch is created.

## Dependency delivery outcome

Nine complete wheel/hash profiles now cover every supported backend/architecture, QA, isolated audit tooling and bootstrap. Fresh builds and all five non-root native smokes pass, including actual CUDA 13 execution on the RTX 5070 Ti, ARM emulation and OpenVINO conversion/reload. Full locked QA passes **625 tests**, with two optional production cases skipped; **64** new boundary cases and real pip hash/graph/target rejection controls pass. Coverage remains **94.31% lines / 88.71% branches** above unchanged floors.

Strict OSV audits pass with no known vulnerabilities or skips across **56 CPU amd64**, **56 CPU arm64**, **75 CUDA**, **75 legacy CUDA**, **58 OpenVINO**, **67 QA**, and **29 packages in each audit-tool architecture**. CPU/OpenVINO baseline versions remain unchanged, both production model/API contracts pass, and representative 224-pixel batches 1/8/32 with both models resident pass within existing memory containment. Physical ARM, legacy NVIDIA, Intel GPU and hosted CI remain separate gates. Ruff, scoped Pyright, workflow lint, copyright, coverage ratchet, secret and whitespace checks pass.

The separate [dependency](dependency-locks.md), [runtime/base policy](runtime-update-policy.md) and random [PR 50](pr-50-local-validation.md) documents contain design, alternatives, pros/cons and outcomes, with a [native validation archive](validation/dependency-locks-2026-10-02.json). Uvicorn's locally adopted minimum is 0.54.0. Delivery stays on `master` with Unreleased changes and no new branch, original PR merge, release, tag or version bump. Those next CI items are now implemented in the separate CI records above.

## Previous capacity delivery

The final QA run passed **561 tests**, with **two optional production-weight tests skipped** in routine execution. Coverage is **94.31% lines / 88.71% branches**, above unchanged **89.42% / 79.37%** floors. Native capacity workloads pass single-reference and batch 1/8/32 parity with both catalog models resident on CPU/OpenVINO CPU. Guarded cold-overlap requests and detached native-owner experiments pass on both profiles; final protected maximum batches return 200 with the 45-second default. Current runtimes are Torch **2.14.1+cpu**, Transformers **5.18.0**, Pydantic **2.13.5** / pydantic-core **2.46.5**, and OpenVINO **2026.4.0**. The [capacity record](capacity-calibration.md) and [measurement archive](validation/capacity-2026-10-02.json) contain configurations, tolerances, immutable image identities, counters and outcomes.

The current guarded OpenVINO cold/multi-model experiment peaked at **6.31 GiB process RSS / 4.83 GiB cgroup memory**, retaining the 8 GiB containment recommendation. Current CPU probes peaked at approximately **2.56 GiB process RSS**, retaining 4 GiB. Shared page-cache charging can make warm cgroup readings lower than process RSS. Timings vary with host load; the final successful requests are not paired speedup measurements. A CPU 30-second trial still timed out and settled correctly, informing the 45-second budget. CPU/OpenVINO shipped-script smokes passed non-root/cache, native projection, dependency, authentication/startup and shutdown checks.

Strict installed-package audits covered **56 CPU**, **58 OpenVINO** and **67 QA** packages with zero known vulnerabilities or skipped packages. Ruff, scoped Pyright with the current Transformers API, workflow lint, copyright, coverage ratchet and whitespace checks passed. The random [PR 45 record](pr-45-local-validation.md) documents its one-line Pydantic floor, official stable-release evidence and local/native validation. GitHub MCP reconfirmed it remains open/unmerged. The manual capacity workflow retains flushed JSONL and logs with read-only repository permissions and pinned actions; hosted execution remains untested locally.

Earlier design records retain their own evidence: ingress's 456-test/controlled-retention results, backend's five-profile/RTX 5070 Ti checks, model contracts' 493-test/published-IR results and remote/input/other PR validations. CPU/OpenVINO were rebuilt here; accelerator capacity, every concurrent HTTP-input combination and a comparative Rust benchmark remain separate work. These improvements cover specific resource/artifact boundaries and package hygiene without constituting a full repository security audit.

The capacity iteration was delivered directly to local/remote `master` without a separate branch, upstream PR merge, release, tag or version bump. Its next recommendation is now implemented in the separate [dependency](dependency-locks.md), [update-policy](runtime-update-policy.md) and [PR 50](pr-50-local-validation.md) records. They contain this iteration's outcomes and the native profile archive.
