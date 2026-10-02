# Reliability and security recommendation stack

Assessment date: 2026-10-01. Recommendations combine local review, actual build/runtime probes, and official sources linked in each design record.

## Completed recommendations and tradeoffs

| Recommendation | Pros | Cons or limits | Outcome |
|---|---|---|---|
| Track inference owners beyond HTTP deadlines | Preserves concurrency and telemetry during timeouts, cancellation, and retries | Running Python threads cannot be forcibly killed | Implemented; [execution design](inference-execution.md) |
| Preserve Uvicorn signals and drain work in lifespan | Shutdown reaches drain logic; cleanup cannot race active inference | Server, application, and container deadlines need coordinated settings | Implemented with execution ownership |
| Shared FIFO admission for batch-window members and explicit batches | One waiting budget, prompt cancellation, live payload preparation | Admission counts do not bound decoded bytes; collection reserves capacity | Implemented; [admission design](bounded-batch-admission.md) |
| Exact CPU/CUDA/OpenVINO profiles, matching digests, isolated vendor indexes | Detects missing images and backend substitution; keeps CUDA libraries out of CPU builds | Version/digest maintenance; transitive/apt dependencies remain ranged | Implemented; [backend design](backend-build-recommendation.md) |
| Modern cu130 default plus explicit amd64 cu126 legacy profile | Supports Blackwell and retains an older GPU option without duplicating Dockerfiles | Driver and GPU architecture selection must match the profile | Implemented and covered by backend checks |
| Non-root offline build/startup probes on PRs | Actual tiny CLIP projection, OpenVINO export/reload, auth, cache, and shutdown evidence before publishing | More CI work; production weights and Intel GPU validation remain separate | Implemented with five native CI jobs |
| Patch dependency floors and strictly audit installed packages through OSV | Covers vendor Torch local versions and packaged setuptools; no advisory suppression | Audits require current public metadata and do not prove complete security | Implemented; test environment and all five runtime audits passed without skips |
| Selected open PRs 28, 42, and 40 implemented locally | Updated pytest, ESM checkout, and OSV scanner execution with immutable references where supported | Original PRs remain unmerged; other workflow references require review | [PR 28](pr-28-local-validation.md), [PR 42](pr-42-local-validation.md), [PR 40](pr-40-local-validation.md) |

## Next items

| Priority | Recommendation | Pros | Cons or decisions needed |
|---|---|---|---|
| Next 1 | Bound HTTP bodies, decoded pixels, and aggregate batch memory before retention | Prevents expansion and payload retention beyond the admission count | Choose documented byte/pixel budgets, chunked-body handling, and batch rejection semantics |
| Next 2 | Add production-model/preprocessor fixtures and cached-IR version contracts | Detects large-model and stale-cache regressions beyond the new tiny CLIP checks | Assets, model revision management, test runtime, and backend-specific environments |
| Next 3 | Complete hash-locked dependency profiles and base/runtime update policy | Reproducible transitive package selection and explicit update review | Per-platform locks, vendor wheels, apt strategy, and security refresh cadence |
| Next 4 | Pin remaining CI actions/reusable workflows and validate permissions | Freezes other executed CI dependencies and reduces credential exposure | Nested container refs and update automation need review too |
| Next 5 | Restore CUDA ARM after valid upstream wheel metadata, then add selective Intel GPU and legacy NVIDIA hardware gates | Demonstrates device-specific behavior on suitable hardware | Current cuSPARSELt ARM metadata fails pip check; hardware access and CI cost; Jetson needs distinct validation |

## Final recommendation stack

Keep Python/FastAPI/AnyIO and the application's scoped admission, queue, execution, and batch services. Maintain small routes and modules with explicit ownership. No platform rewrite or new CommonJS code is needed; any future JavaScript should use ES Modules.

Use CPU-only Torch for the default image, cu130 Torch/NVIDIA CUDA 13 for modern NVIDIA GPUs, the documented cu126 profile for older amd64 NVIDIA hardware, and matching OpenVINO bindings/base for Intel deployments. Keep exact native-backend profiles separate from shared requirements and resolve Torch from a single official index before constrained PyPI installation. Keep non-root runtime execution and writable caches scoped to the application.

Validate with event-controlled worker tests, actual tiny native-model checks, network-disabled service smokes, and strict installed-package audits. Publish only from release tags after unit/backend gates. Immutable base/action image references require reviewed updates; default Docker bases receive weekly Dependabot proposals, while the legacy base override and direct scanner image require explicit digest review.

Next implement the input-memory budget: enforce an HTTP body ceiling before JSON parsing, bound decoded image pixels before conversion/loading, and charge aggregate batch bytes before queuing or retaining payloads. Count admission alone cannot bound memory. Preserve authentication and remote-image validation, avoid recording payloads/secrets, and cover cancellation/rejection without leaking reservations.

## Delivery outcome

The backend iteration preserves the previously committed execution/admission work. The full suite passed 252 tests on the new Python 3.12 CPU environment. Coverage is 92.97% lines and 85.48% branches, above the unchanged 89.42%/79.37% floor. The executor remains at 100% line and branch coverage. A strict OSV audit covered all 79 packages with zero findings and zero skips; package advisory suppression was not used.

All five rebuilt images passed their shipped-script smokes; modern CUDA inference and service device selection passed on an RTX 5070 Ti. CUDA ARM failed because its vendor wheel declares an unsupported internal SBSA tag; keep pip check strict and defer this target. Ruff, Pyright, workflow lint, compilation, copyright, coverage ratchet, and whitespace checks passed. The direct OSV 2.3.8 image returned the expected clean/vulnerable fixture exits; GitHub MCP reconfirmed PR 40 is open/unmerged.

Documentation, README, and the Unreleased changelog describe the design, alternatives, implementation, compatibility limits, and outcomes. Delivery targets `fix/backend-build-contracts`. No release or merge is part of this iteration. These checks improve security properties and package hygiene; they are not a full repository security audit.
