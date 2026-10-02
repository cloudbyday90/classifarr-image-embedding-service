# Reliability and security recommendation stack

Assessment date: 2026-10-02. The current production-model iteration starts from `484c7ed3515e97fe74899d932cb89b4346b21174` and is delivered directly on `master`. Python is retained by user decision. Recommendations combine local code evidence, actual build/runtime measurements and official sources discovered through web/GitHub MCP, linked in the separate design records.

## Completed recommendations and tradeoffs

| Recommendation | Pros | Cons or limits | Outcome |
|---|---|---|---|
| Track inference owners beyond HTTP deadlines; preserve server signals and drain work | Preserves accounting during cancellation; cleanup cannot race workers | Running Python threads cannot be forcibly killed; supervisor deadlines must align | Implemented; [execution design](inference-execution.md) |
| Shared FIFO admission for batch-window members and explicit batches | One waiting budget; prompt cancellation and live-client dispatch | Separate byte/pixel limits still matter | Implemented; [admission design](bounded-batch-admission.md) |
| Streamed HTTP and modular inline/image/batch budgets | Rejects excess before queue retention or RGB/resize expansion; predictable closure | 413 behavior; budgets are not exact RSS quotas | Implemented; [input design](input-memory-budgets.md) |
| Approved numeric-address connections and validated redirects/TLS | Closes reproduced authority, redirect and DNS-to-connection bypasses | Per-hop timeouts do not bound total DNS/download lifetime | Implemented; [remote design](remote-image-destinations.md) |
| Complete-lifetime HTTP ingress and explicit launcher/Compose budgets | Rejects excess before body receive; probes retain access; workers cannot multiply implicitly | Per-worker budgets; slow clients/native memory need sizing | Implemented; [ingress design](aggregate-ingress.md) |
| Exact backend profiles, base digests and isolated vendor indexes | Detects substitution; keeps CUDA libraries out of CPU builds; modern/legacy NVIDIA options | Transitive/apt dependencies remain ranged; driver/hardware gates matter | Implemented; [backend design](backend-build-recommendation.md) |
| Immutable publisher models, verified assets and explicit PIL preprocessing | Detects drift/corruption before deserialization; stable dimensions and math | Hashing and reviewed model updates; base publisher supplies restricted PyTorch weights | Implemented; [model contracts](model-artifact-contracts.md) |
| Versioned, process-locked atomic XML/BIN publication and explicit precision | Rejects stale/partial/corrupt IR; first use/reload share the saved representation | FP32 disk/export memory; checksums do not authenticate a malicious local writer | Implemented with four-process contention and actual production reload checks |
| Production-weight probes and separate CPU/OpenVINO CI | Real single/batch/API parity, cache reuse and peak RSS evidence | Large downloads; accelerator/max-batch capacity and hosted CI remain separate | Four local production probes passed; small fixtures stay offline |
| 8 GiB/no-swap OpenVINO default after measured cold-export peak | Accommodates the observed 4.40 GiB large-model cold path while retaining containment | Larger host budget; maximum capacity is not established | Implemented; CPU/CUDA retain 4 GiB defaults; explicit override wins |
| Strict OSV audits of installed environments | Covers vendor Torch and packaged tooling without advisory suppression | Current public metadata is needed; not proof of complete security | CPU/OpenVINO/QA audits passed with no skips |
| Complete OpenVINO environment copy and duplicate-inventory guard | Removes reproduced stale NumPy metadata; aligns imports and audits | Extra runtime build layer; native inventory checks remain necessary | Implemented; [runtime environment design](openvino-runtime-environment.md) |
| Selected PRs implemented locally through verified immutable refs | Requested maintenance updates without upstream merges | Hosted reporting/cache behavior remains CI work | [PR 28](pr-28-local-validation.md), [42](pr-42-local-validation.md), [40](pr-40-local-validation.md), [33](pr-33-local-validation.md), [49](pr-49-local-validation.md), [51](pr-51-local-validation.md), and now random [PR 41](pr-41-local-validation.md) |
| Retain modular Python and measure before language changes | Builds on native inference and tested contracts | Python footprint remains; Rust benefit is unmeasured here | User confirmed; [language decision](language-platform-decision.md) |

## Next items

| Priority | Recommendation | Pros | Cons or decisions needed |
|---|---|---|---|
| Next 1 | Calibrate maximum-batch, multi-model and simultaneous cold/warm capacity with pinned fixtures | Sizes container, ingress and worker budgets from actual owner peaks; can guide export isolation | RSS/cgroup/latency instrumentation and accelerator VRAM/hardware gates; 4.40 GiB cold export is only a small-batch observation |
| Next 2 | Complete hash-locked dependency profiles and runtime/base update policy | Reproducible transitive selection; reviewed security refresh | Per-platform/vendor wheel locks and apt strategy; this iteration resolved Transformers 5.18 while the prior CPU image used 5.17 |
| Next 3 | Pin remaining CI/reusable refs and review permissions/native artifact verification | Freezes more executed dependencies and narrows credentials | Release/nested refs, reporting/cache behavior and the Gitleaks native downloader need separate review |
| Next 4 | Bound total upload and DNS/download lifetime; evaluate isolation from measured need | Releases finite slots retained by slow uploads/resolution/trickled bodies | Cancellation, proxy/server budgets, resolver/process boundaries and accounting need design |
| Next 5 | Restore CUDA ARM after upstream metadata repair; add selected Intel/legacy NVIDIA hardware gates | Demonstrates supported device behavior | cuSPARSELt ARM metadata currently fails strict pip check; suitable hardware/runner cost |

The next capacity task should measure one worker with both catalog models resident, maximum allowed batches, overlapping model loads and detached owners, then add measured headroom. Record cgroup memory events, process RSS, queue wait and stage times. Do not raise workers/admission from the small current probes. Decide whether a separate exporter process is justified by cold-path retention before adding an artifact service.

## Final recommendation stack

Keep Python/FastAPI/AnyIO with small ingress, admission, queue, execution, input, remote transport and model/artifact services. Native tensor libraries perform inference. A later language experiment must show measured benefit and equivalent model/API/backend contracts. Authored JavaScript uses ESM; the user explicitly accepted the external CommonJS Gitleaks action in PR 41.

Keep CPU-only Torch by default, exact cu130/cu126 profiles for appropriate NVIDIA hardware, and matching OpenVINO bindings/base for Intel deployments. Use immutable approved model revisions/digests, restricted local weight loading, explicit PIL preprocessing and no remote model Python. Maintain the process-owned IR contract, complete manifest, finite locks, atomic publication, FP32 serialization and accuracy execution policy on application-owned local volumes.

Retain one worker, eight complete HTTP ingress owners, server concurrency 64 and backlog 128. Keep 16 MiB bodies, 10 MiB compressed images, 32 MiB batch bytes, 16-million source/resize pixels and 32-million aggregate pixels. Contain CPU/CUDA at the initial tunable 4 GiB/no-swap default and OpenVINO at 8 GiB/no swap, based on the measured large-model cold path. These ceilings are not maximum capacity guarantees and do not cap GPU VRAM. Preserve ownership after HTTP timeout and deterministic image closure.

Keep remote URLs disabled unless needed and prefer explicit allowlists. Use approved public numeric destinations, verified original-host TLS identity, at most three validated redirects and four bounded address attempts. Preserve authority/DNS policy, response closure, byte limits, sanitized errors and isolation from ambient transport credentials.

Validate small offline fixtures in routine pytest, real production models in the separate change-scoped/weekly CPU/OpenVINO workflow, installed packages with strict OSV, and device behavior on appropriate hardware. The production workflow performs verified networked prefetch followed by offline containers; it neither publishes images nor changes the existing tag-only release gate. Hash locks and measured capacity are the next operational safeguards.

## Delivery outcome

The final QA run passed **493 tests**, with **two optional production-weight tests skipped** in routine execution. Coverage is **94.17% lines / 88.55% branches**, above unchanged **89.42% / 79.37%** floors. All four actual production CPU/OpenVINO probes passed single/batch/native-vector and authenticated-route contracts; both OpenVINO models passed published-IR first/reload parity. Current runtimes are Torch **2.14.1+cpu**, Transformers **5.18.0** and OpenVINO **2026.4.0**. The model record contains tolerances, warmup/reload observations and peak RSS.

The large OpenVINO model reached **4.40 GiB service peak RSS** before extra reference/reload validation and **7.83 GiB full-probe peak**. Its cold path prompted the OpenVINO default increase; representative maximum capacity remains Next 1. Native cache volumes replaced initial Windows-share probes blocked in file-share I/O. CPU/OpenVINO shipped-script smokes passed non-root/cache, native projection, dependency, authentication/startup and shutdown checks.

Strict installed-package audits covered **56 CPU**, **58 OpenVINO** and **67 QA** packages with zero known vulnerabilities or skipped packages. Ruff, scoped Pyright with the current Transformers API, workflow lint, copyright, coverage ratchet and whitespace checks passed. The [PR 41 record](pr-41-local-validation.md) covers the Node 24 wrapper, published native checksum, clean/leak/error controls and redacted SARIF. GitHub MCP reconfirmed it remains open/unmerged; hosted cache/reporting was not tested locally.

Earlier design records retain their own evidence: ingress's 456-test/controlled-retention results, backend's five-profile/RTX 5070 Ti checks, and remote/input/other PR historical validations. All five profiles were not rebuilt here; production accelerator/max-batch results and a comparative Rust benchmark are not claimed. These improvements cover specific resource/artifact boundaries and package hygiene without constituting a full repository security audit.

README, implementation-plan references and the Unreleased changelog describe the final behavior. Separate model and PR documents record design, alternatives, recommendations and outcomes. Delivery is directly to local/remote `master` without a separate branch, upstream PR merge, release, tag or version bump.
