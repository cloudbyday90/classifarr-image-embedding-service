# Contract and validation map

Service modules below are under `src/image_embedder`; dependency helpers are under `scripts`. Other paths are relative to the Classifarr repository root. Read the relevant row, not every design record. Confirm filenames and current behavior with `rg` before invoking tools.

| Changed boundary | Owner / primary modules | Read / focused validation |
|---|---|---|
| Upload and ingress | `ingress.py`, `ingress_middleware.py`, `request_body_deadline.py`, `main.py` | `docs/aggregate-ingress.md`, `docs/request-lifetimes.md`; `tests/test_ingress.py`, `tests/test_request_lifetimes.py` |
| HTTP response transport | `response_send_deadline.py`, `response_http_protocol.py`, `server.py` | `docs/response-send-lifetimes.md`; `tests/test_response_send_deadline.py`, `tests/test_response_http_protocol.py`; real parser/TLS/slow-reader and worker-spawn probes |
| Admission and detached inference | `admission.py`, `execution.py`, `queue.py`, `batch.py` | `docs/inference-execution.md`, `docs/bounded-batch-admission.md`; `tests/test_execution.py`, `tests/test_execution_batch_window.py`, `tests/test_batch_admission.py` |
| Remote fetch | `remote_process.py` owns child; `remote_worker.py`, `remote_protocol.py`, `remote_fetch.py`, `remote_url.py` enforce policy | `docs/remote-image-destinations.md`, `docs/request-lifetimes.md`; `tests/test_remote_process.py`, `tests/test_remote_process_network.py`; native blocked DNS, TLS and public-destination refusal |
| Shared batch preparation | Per-invocation `remote_budget.py`, `embedder.py`; existing byte/pixel budgets in `input_limits.py` | `docs/remote-batch-budget.md`; `tests/test_remote_batch_budget.py`, `tests/test_remote_batch_process.py`, `tests/test_remote_batch_api.py`, existing embedder/batch tests |
| Input/cache/vector contracts | `image_input.py`, `input_limits.py`, `embedder.py`, `models.py`, `routes/` | `docs/input-memory-budgets.md`, `IMPLEMENTATION_PLAN.md`; discover relevant tests with `rg --files tests`; validate content keys and vector shape/finite/normalization invariants |
| Model initialization and artifacts | `model_initialization.py`, `model_catalog.py`, `model_loading.py`, `openvino_models.py` and IR helpers | `docs/model-artifact-contracts.md`, `docs/model-initialization.md`; deterministic fixtures first, then affected offline production model/backend/device checks |
| Dependency intent / inventory | `scripts/dependency_profiles.py`, `dependency_locks.py`, `lock_dependencies.py`, `install_dependencies.py` | `docs/dependency-locks.md`, `docs/runtime-update-policy.md`; all input contracts, affected exact inventories/`pip check`, unchanged lock comparison for floor-only adoption |

## Repeatable commands

Use the repository's locked QA image after verifying its identity and availability. The image tag is a local convenience, not immutable proof. Full QA uses `python -m pytest` with this repository's `pytest.ini`; use `-o addopts=''` only for focused checks. Store coverage output outside Git, then apply `scripts/check_coverage_ratchet.py` to that exact XML. Do not lower the floors.

Portable offline input checks: `python scripts/lock_dependencies.py --check`. Native exact-inventory check: `python scripts/install_dependencies.py --backend qa --root <checkout> --verify-only` inside the corresponding locked Linux environment. Review all affected profiles; an amd64 QA pass alone does not validate ARM or accelerator execution.

Run `python scripts/check_copyright.py`, scoped Ruff/Pyright and `git diff --check`. Follow `docs/ci-native-artifacts.md` and `.github/ci-tools.json` for verified native scanners; use `.gitleaks.toml` unchanged. Scan checkout-equivalent source with its relative paths so exact historical allowlists keep their intended scope. Do not scan ignored secrets or silently broaden allowlists.

## Evidence boundaries

Timeouts are duration policies, not hard real-time guarantees. OS launch/reaping and scheduling add latency. Returning from an HTTP waiter is independent of synchronous completion. A local transport-buffer drain is independent of peer receipt. A mocked model proves routing/math contracts, not GPU throughput or model artifact compatibility.

For floor-only dependencies, compare normalized input text, every `.txt` wheel lock and all artifact records against the reviewed Git baseline. Renew only input hashes that changed after proving the same locked distributions satisfy the requirement. For runtime graph changes, use native lock generation and the affected backend validation instead.
