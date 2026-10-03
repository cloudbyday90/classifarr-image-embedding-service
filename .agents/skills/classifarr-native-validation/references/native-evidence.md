# Native evidence map

All command paths are relative to the repository root. Use fresh venvs and keep result paths outside Git.

| Boundary | Implementation | Native suites and observations |
|---|---|---|
| Private setup and rotation | `scripts/secret_windows.py`, `secret_publication.py`, `secret_setup.py` | `tests/test_secret_setup.py`: owner-only creation/rotation, junction refusal, pre-write fail-closed ACL/handle failures, concurrent complete publication, launcher secrecy |
| Remote worker | `src/image_embedder/remote_process.py`, `remote_worker.py`, `remote_fetch.py` | `test_remote_process.py`, `test_remote_process_network.py`, `test_remote_native_transport.py`: blocked DNS kill/reap, retained detached ownership, real TLS/HTTP/trickle and production private-destination refusal |
| Shared remote batch | `src/image_embedder/remote_budget.py`, `embedder.py` | `test_remote_batch_process.py`: 32-item shared budget, earlier deadline, success then trickle, owner retention and settlement |
| Response lifetime | `src/image_embedder/response_http_protocol.py`, `response_send_deadline.py` | `test_response_http_protocol.py`: httptools/h11, TLS, terminal/stream writes, disconnect, cleanup and capacity recovery on native asyncio |
| Target and execution gate | `scripts/dependency_profiles.py`, `windows_contract_policy.py` | `test_windows_dependency_profiles.py`, `test_windows_contract_policy.py`; complete native run remains required |

In a fresh conventional Windows x64 Python 3.13 or 3.14 venv:

```text
python scripts/install_dependencies.py --backend bootstrap
python scripts/install_dependencies.py --backend windows-contracts
python scripts/validate_windows_contracts.py --output <temporary-path>/windows-contract-results.json
```

The invocation's `python` must be the venv executable. Install commands verify the graph; the fixed native runner verifies it again and clears routine coverage addopts without needing pytest-cov. Regenerate locks only deliberately with reviewed pip: `scripts/lock_dependencies.py --backend bootstrap` and `--backend windows-contracts` inside each target. `--check` verifies every committed profile offline. Existing Linux backend checks remain Linux CPython 3.12.

Read [the Windows design](../../../../docs/windows-native-validation.md) for researched tradeoffs and results. The workflow is `.github/workflows/windows-contracts.yml`. Its upload contains metadata only, no console capture, `.env`, TLS keys or traceback. Native reports must show no gate errors, successful mandatory-case counts, only permitted setup skips, and explicit unsupported uvloop deselections. Hosted image fields can be null locally; do not invent them.

For wider resource design use the sibling resource-contract skill when it is available. For model/task/storage/device sizing use the capacity-calibration skill. Neither replaces native OS evidence.
