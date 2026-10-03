# Measurement contracts

All implementation paths below are relative to the repository root.

| Boundary | Implementation | Evidence required |
|---|---|---|
| Memory/controller membership | `scripts/capacity_metrics.py` | Own v1/v2 membership/mount root, current vs kernel lifetime peak, limit events; do not infer shared page charging from RSS |
| Tasks and anonymous output | `scripts/capacity_resources.py` | Independently mounted v1 PID controller or v2, parent threads, task peak/limit events; whole-filesystem bytes/inodes and baseline |
| Accelerator | `scripts/capacity_cuda.py` | GPU available and loaded model device, synchronized phase boundaries, live/reserved allocator peaks; device-wide free memory labelled separately |
| Concurrent input and cancellation | `scripts/capacity_mixed.py`, `scripts/capacity_remote_fixture.py` | Actual protected ASGI routes and fresh remote children, ordered vectors, retained owner/live waiter, native outcome after cancellation, all children reaped |
| Native upload pressure | `scripts/capacity_socket.py` and its scoped body/server/receive/upload/mixed helpers | Server bytes held before EOF, header-only excess rejection, disconnect settlement, recovered vectors, actual children reaped; label direct HTTP/1 and shared client overhead |
| Workload/CLI | `scripts/capacity_workload.py`, `scripts/capacity_probe.py` | Both resident models, cache disabled, bounded 1/internal/API batches, near-ceiling fixtures, offline assets; no CPU fallback labelled CUDA |
| Delivery | `.github/workflows/capacity-calibration.yml` | Manual CPU/OpenVINO experiments; NVIDIA hardware is an explicit local/operator requirement |

Prefer one changed dimension per comparison. Use `--backend cpu`, `openvino` or `cuda` with its matching image; CUDA also requires Docker `--gpus all`. Select serial, api, overlap, detached, mixed or socket scenarios. For native capacity, use `--image-edge 974 --repeats 1` under current default pixel ceilings. These are fixture choices, not production requirements. Check CLI limits when operator settings differ.

The mixed scenario uses two padded remote PNG bodies and remaining inline images. A live maximum inline batch queues behind the remote/native owner. A second mixed caller cancels after child dispatch. ASGI measurements do not establish proxy/socket ingress buffering, and padded PNGs do not represent every codec or byte/pixel combination.

Focused checks: `python -m pytest tests/test_capacity_metrics.py tests/test_capacity_resources.py tests/test_capacity_cuda.py tests/test_capacity_workload.py tests/test_capacity_mixed.py tests/test_capacity_socket.py -o addopts=''`. Routine checks do not fetch model weights. Use the complete locked QA suite when code behavior changes, and actual model/device runs for affected backend measurements. Validate modified workflow syntax and skill metadata; do not substitute string matching for skill evaluation.

Examples to evaluate: calibrate CUDA with no GPU (refuse a GPU claim), low sampled tmpfs use but transient anonymous output (report a lower bound), canceled mixed caller (validate its native outcome), shared GPU busy with another application (do not attribute device use to this service), NumPy minimum-only review (do not trigger capacity work), or documentation-only correction (no model rerun).

Current design/results: [deployment capacity](../../../../docs/deployment-capacity.md), [original capacity](../../../../docs/capacity-calibration.md), [resource lifetimes](../../../../docs/request-lifetimes.md). Read only the record relevant to the deployment question.
