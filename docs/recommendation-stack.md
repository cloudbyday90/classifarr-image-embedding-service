# Reliability and security recommendation stack

Assessment date: 2026-10-02. Recommendations combine local review, actual build/runtime probes, and official sources discovered through web/GitHub MCP and linked in each design record. The current iteration starts from `97f2d9578fca851abac08472477c3458bc9ae6b5` on `master`.

## Completed recommendations and tradeoffs

| Recommendation | Pros | Cons or limits | Outcome |
|---|---|---|---|
| Track inference owners beyond HTTP deadlines | Preserves concurrency and telemetry during timeouts, cancellation, and retries | Running Python threads cannot be forcibly killed | Implemented; [execution design](inference-execution.md) |
| Preserve Uvicorn signals and drain work in lifespan | Shutdown reaches drain logic; cleanup cannot race active inference | Server, application, and container deadlines need coordinated settings | Implemented with execution ownership |
| Shared FIFO admission for batch-window members and explicit batches | One waiting budget, prompt cancellation, live payload preparation | Counts need the separate byte/pixel budgets below; collection reserves capacity | Implemented; [admission design](bounded-batch-admission.md) |
| Native streamed HTTP limit plus modular inline/image/batch budgets | Rejects excess before JSON completion, queue retention, or RGB/resize expansion; deterministic image closure | Oversized requests now receive 413; input ceilings are not exact RSS or aggregate ingress quotas | Implemented; [input budget design](input-memory-budgets.md) |
| Exact CPU/CUDA/OpenVINO profiles, matching digests, isolated vendor indexes | Detects missing images and backend substitution; keeps CUDA libraries out of CPU builds | Version/digest maintenance; transitive/apt dependencies remain ranged | Implemented; [backend design](backend-build-recommendation.md) |
| Modern cu130 default plus explicit amd64 cu126 legacy profile | Supports Blackwell and retains an older GPU option without duplicating Dockerfiles | Driver and GPU architecture selection must match the profile | Implemented and covered by backend checks |
| Non-root offline build/startup probes on PRs | Actual tiny CLIP projection, OpenVINO export/reload, auth, cache, and shutdown evidence before publishing | More CI work; production weights and Intel GPU validation remain separate | Implemented with five native CI jobs |
| Patch dependency floors and strictly audit installed packages through OSV | Covers vendor Torch local versions and packaged setuptools; no advisory suppression | Audits require current public metadata and do not prove complete security | Implemented; test environment and all five runtime audits passed without skips |
| Selected open PRs 28, 42, 40, and now randomly selected 33 implemented locally | Updated pytest, ESM checkout, OSV scanner, and consistent pytest-cov reporting | Original PRs remain unmerged; dependency ranges still need lock design | [PR 28](pr-28-local-validation.md), [PR 42](pr-42-local-validation.md), [PR 40](pr-40-local-validation.md), [PR 33](pr-33-local-validation.md) |

## Next items

| Priority | Recommendation | Pros | Cons or decisions needed |
|---|---|---|---|
| Next 1 | Validate remote destinations across redirects and DNS-to-connection changes | Keeps allowlisted/public-address policy attached to actual network destinations | Choose bounded manual redirects or refusal, connection-time checks, proxy policy, and IPv4/IPv6 tests; remote URLs stay disabled by default |
| Next 2 | Bound concurrent body parsing and size deployment memory across workers | Complements per-request limits with aggregate ingress protection | Server/proxy controls, queued-body retention, model/tensor sizing, and container quotas need workload measurements |
| Next 3 | Add production-model/preprocessor fixtures and cached-IR version contracts | Detects large-model and stale-cache regressions beyond tiny CLIP checks | Assets, model revision management, test runtime, and backend-specific environments |
| Next 4 | Complete hash-locked dependency profiles and base/runtime update policy | Reproducible transitive package selection and explicit update review | Per-platform locks, vendor wheels, apt strategy, and security refresh cadence |
| Next 5 | Pin remaining CI actions/reusable workflows and validate permissions | Freezes other executed CI dependencies and reduces credential exposure | Nested container refs and update automation need review too |
| Next 6 | Restore CUDA ARM after valid upstream wheel metadata, then add selective Intel GPU and legacy NVIDIA hardware gates | Demonstrates device-specific behavior on suitable hardware | Current cuSPARSELt ARM metadata fails pip check; hardware access and CI cost; Jetson needs distinct validation |

The next remote-fetch item comes from inspecting `_validate_remote_url` and `_fetch_image_bytes`: the first URL is checked, then `requests.get` is called without disabling its [documented automatic redirects](https://requests.readthedocs.io/en/latest/user/quickstart/). Current DNS validation also precedes the HTTP library's connection. Review the transport boundary and validate deterministic tests before assigning severity or claiming an exploit. This iteration does not enable or alter remote fetching.

## Final recommendation stack

Keep Python/FastAPI/AnyIO and the application's scoped admission, queue, execution, and batch services. Maintain small routes and modules with explicit ownership. Use Starlette's supported body middleware and the small `input_limits`/`image_input` modules for resource policy instead of expanding the inference class with parsing/accounting logic. No platform rewrite or new CommonJS code is needed; any future JavaScript should use ES Modules.

Use CPU-only Torch for the default image, cu130 Torch/NVIDIA CUDA 13 for modern NVIDIA GPUs, the documented cu126 profile for older amd64 NVIDIA hardware, and matching OpenVINO bindings/base for Intel deployments. Keep exact native-backend profiles separate from shared requirements and resolve Torch from a single official index before constrained PyPI installation. Keep non-root runtime execution and writable caches scoped to the application.

Validate with event-controlled worker tests, actual tiny native-model checks, network-disabled service smokes, and strict installed-package audits. Publish only from release tags after unit/backend gates. Immutable base/action image references require reviewed updates; default Docker bases receive weekly Dependabot proposals, while the legacy base override and direct scanner image require explicit digest review.

Keep the selected 16 MiB HTTP, 10 MiB compressed-image, 32 MiB batch-byte, 16-million source/resize-pixel, and 32-million aggregate-pixel defaults. Apply inline checks before admission and remote/direct-call checks in the worker. Keep Pillow's global safety policy intact and close detached images after native work. These choices add predictable input boundaries while preserving accepted-image math; they require explicit tuning for large batches and deployment memory limits.

Next implement a small remote-fetch service that owns redirect policy, destination validation, streaming ceilings, and response closure. Preserve authentication, disable remote access by default, avoid recording payloads/secrets, and test refusal/cancellation without real network access to protected destinations. Follow with aggregate ingress limits and production-model/cache contracts.

## Delivery outcome

The current input-budget iteration preserves the previously merged execution/admission/backend work. The complete suite passed 315 tests on Python 3.12 with the selected pytest-cov 7.1.0 plugin. Coverage is 94.74% lines and 88.48% branches, above the unchanged 89.42%/79.37% floor. Input accounting and the executor have 100% line and branch coverage. Strict OSV audits checked the actual CPU runtime's 56 packages and test environment's 63 packages with zero known vulnerabilities and zero skips; no advisory suppression was used.

The newly rebuilt CPU image passed its shipped-script smoke, including a real native CLIP projection, non-root/cache checks, authentication, startup, and shutdown. A live Uvicorn probe confirmed declared/chunked 413 refusals and subsequent authenticated requests; the Starlette 1.6.0 minimum passed 28 input/integration tests. Ruff, scoped Pyright, workflow lint, compilation, copyright, coverage ratchet, and whitespace checks passed. PR 33's four reporting/fail-under fixture cases passed; GitHub MCP reconfirmed it is open/unmerged.

The prior [backend iteration](backend-build-recommendation.md) separately records its five-image smokes/audits and RTX 5070 Ti check; those native profiles were not all rebuilt in this input-budget iteration. CUDA ARM's unsupported vendor SBSA metadata remains a deferred support limit with pip check kept strict.

Separate input-budget and PR 33 documents record design, alternatives, implementation, and outcomes; README/config and the Unreleased changelog describe the new settings and behavior. Delivery targets `fix/input-memory-budgets`. No release, tag, upstream PR merge, or merge to the default branch is part of this iteration. These checks improve specific security/resource properties and package hygiene; they are not a full repository security audit.
