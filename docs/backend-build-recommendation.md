# Backend build and dependency validation

Assessment date: 2026-10-01. Status: implemented and locally validated for the five supported build targets; CUDA ARM is deferred with evidence below.

## Problem and research

The old CUDA base, `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04`, returned “not found” during registry inspection. Installing cu124 Torch and then `torch>=2.10` could also replace the chosen wheel. CPU builds used the default PyPI Torch package, which now selects CUDA on Linux. OpenVINO duplicated shared dependencies, mixed indexes, and attempted to create GID 1001 even though Intel's base already owns it; a real build reproduced that failure.

Official URLs were discovered through web search, document links, and GitHub MCP. Registry manifests and actual wheel availability were checked separately. [PyTorch's release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) distinguishes CUDA 13 support for Blackwell from CUDA 12.6 support for older GPUs. The host's RTX 5070 Ti therefore needs cu130. Keep a separate cu126 profile for Maxwell/Pascal/Volta on amd64 rather than selecting a wheel solely by toolkit version.

An installed-environment audit reported [Torch's advisory with a 2.13.0 fix](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), [Transformers' reviewed path-traversal advisory](https://github.com/github/advisory-database/blob/main/advisories/github-reviewed/2026/08/GHSA-xrqw-3rrv-vx5w/GHSA-xrqw-3rrv-vx5w.json), and Pillow issues addressed in its [12.3.0 release notes](https://github.com/python-pillow/Pillow/blob/main/docs/releasenotes/12.3.0.rst). Raise shared floors to Torch >=2.13, Transformers >=5.10.1 (the first available stable 5.10 wheel), and Pillow >=12.3.0. These are package maintenance decisions, not claims that every advisory is exploitable through this API. No advisory is suppressed.

Torch's vendor index also supplied setuptools 78.1.0. Raise its minimum to 83.0.0, whose [official changelog](https://github.com/pypa/setuptools/blob/v83.0.0/NEWS.rst) documents the advisory fix. The default PyPI audit skipped Torch's vendor-local version; switch to the [documented OSV service and strict mode](https://github.com/pypa/pip-audit/blob/main/README.md?plain=1). A complete installed-environment audit must include Torch, not silently omit it. A separate vulnerable `torch==2.11.0+cpu` fixture returned exit 1 and `PYSEC-2025-194` with fix 2.13.0, confirming the vendor-local version is audited. The [official advisory record](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-194.yaml) identifies its GHSA alias.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Replace only the missing base tag | Small change | Leaves wheel replacement, CPU bloat, and other build failures undetected | Reject |
| Use one unqualified Torch dependency for all images | Fewer files | Backend can change with PyPI defaults; GPU libraries enter CPU images | Reject |
| Exact backend profiles, digest pins, isolated indexes, offline smoke jobs | Detects dependency and runtime drift before publishing; supports explicit hardware choices | More CI work and reviewed version maintenance | Adopt |
| Real production-model GPU inference on every PR | Strong hardware/model evidence | Needs GPU runners, model weights, longer jobs | Add selectively after build checks |

Keep one shared `requirements.txt`, small Torch profile files, and an OpenVINO file that includes the shared requirements. Install Torch using exactly one official index, then constrain its version during the PyPI installation. [pip documents that index locations have no priority and warns about dependency confusion with extra indexes](https://pip.pypa.io/en/latest/cli/pip_install/). Remove the OpenVINO extra-index option; do not combine PyPI and vendor indexes in one resolver invocation.

[Docker recommends digest pins and explicit update processes](https://docs.docker.com/build/building/best-practices/). Pin verified manifests and add weekly Docker Dependabot checks for default Dockerfile bases. The legacy base override in CI and README needs manual digest review. These pins stabilize backend choices; apt and transitive Python packages remain ranged, so full hash-locked dependencies are a follow-up. Validate the installed environment with `pip check` and audit it instead of resolving an unrelated default Torch build for auditing.

## Selected contracts

| Profile | Torch | Runtime base | Build/smoke architectures |
|---|---|---|---|
| CPU | 2.14.1+cpu | Ubuntu 24.04 | amd64, arm64 |
| CUDA default | 2.14.1+cu130 | NVIDIA CUDA 13.0.3 base / Ubuntu 24.04 | amd64 |
| CUDA legacy | 2.14.1+cu126 | NVIDIA CUDA 12.6.3 base / Ubuntu 24.04 | amd64 |
| OpenVINO | 2.14.1+cpu for export; OpenVINO 2026.4.0 | Intel OpenVINO Ubuntu 24 runtime / 2026.4.0 | amd64 |

The [official NVIDIA image description](https://hub.docker.com/r/nvidia/cuda) distinguishes the minimal base from larger runtime/cuDNN images. Torch's installed wheel dependencies supply the CUDA libraries and cuDNN; avoid a second cuDNN installation in the base. The host supplies its driver through NVIDIA Container Toolkit. [CUDA 13.0.3 notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html) give driver branch 580 as the minor-compatibility floor and Linux 580.126.20 as the corresponding toolkit driver. Prefer that version or newer; Windows/WSL requires a compatible separately installed driver. [CUDA 12.6.3 notes](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html) list Linux 560.35.05 and Windows 561.17 as corresponding versions; older minor-compatible drivers have restrictions.

CUDA 13 supports Turing and newer on amd64, including Blackwell; cu126 retains older amd64 architectures but excludes Blackwell. Although PyTorch lists ARM server architectures and publishes the selected cu130 wheel, a real ARM build fails `pip check` on `nvidia-cusparselt-cu13==0.8.1`. Inspection of the wheel downloaded from the official Torch index found filename `nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl`, but internal `WHEEL` metadata declares `Tag: py3-none-manylinux2014_sbsa`. Keep dependency validation strict and exclude CUDA ARM from the supported matrix until a matching Torch dependency provides valid platform metadata and passes native build/smoke checks. No vendor metadata is rewritten or failure suppressed. Jetson compatibility needs separate validation.

Match the [OpenVINO 2026.4.0 Python package](https://pypi.org/project/openvino/2026.4.0/) to the verified Intel base. The newer 2026.4.1 package appeared on October 1, but the corresponding container tag was absent during inspection; do not silently mix versions. This Intel image is amd64 only. [Intel's system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html) provide device guidance, not additional architectures for this manifest. Reuse UID/GID 1001 and retain video/render memberships.

Verified digests:

- Ubuntu 24.04: `sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3` (multi-platform index).
- CUDA 13.0.3 base: `sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1` (multi-platform index).
- CUDA 12.6.3 base: `sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e` (multi-platform index).
- OpenVINO Ubuntu 24 runtime 2026.4.0: `sha256:b326dcae9b91fa25f794a391d9953e65a65096597237e6de431116b184a01718` (amd64 manifest).

## Design and validation

One CUDA Dockerfile accepts explicit base, Torch profile, and index build arguments. Defaults select the modern profile; CI and README pass all three legacy values together. Constraints turn an incompatible shared requirement into a build failure instead of replacing Torch. The smoke rejects a mismatched runtime `CUDA_VERSION` as well as an incorrect Torch flavor. Separate Torch installation layers improve reuse; multi-stage runtime images retain the non-root identity and existing API entrypoint.

Five CI jobs build and smoke without publishing: CPU on both architectures, plus modern CUDA, legacy CUDA, and OpenVINO on amd64. [GitHub documents native ARM runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). Jobs grant only repository read access, use no registry credentials, drop capabilities, forbid privilege escalation, and disable networking for inference/startup checks. Tag-only publishing waits for unit and backend jobs.

Each job also audits the actual image's runtime site-packages with strict OSV auditing. Audit tools are installed only in a temporary executable tmpfs; the image root filesystem is read-only, capabilities are dropped, and privilege escalation is forbidden. Public package/advisory access is enabled for auditing. The temporary tool environment does not change the shipped image. Use pip-audit >=2.10.1 for its current OSV response handling; any vulnerability or skipped package fails the check.

CUDA's installed venv alone measured approximately 6.6 GiB for the legacy profile, so multi-stage copies and layer storage can exceed the runner's guaranteed free space. On disposable GitHub-hosted CUDA jobs, reclaim the unused Android SDK, .NET SDK, and hosted tool cache before building. The step requires `RUNNER_ENVIRONMENT=github-hosted`, uses explicit absolute paths, and leaves Docker/checkout available. The [official runner inventory](https://github.com/actions/runner-images/blob/main/images/ubuntu/Ubuntu2404-Readme.md) documents these preinstalled tools. Hosted storage and native runner execution still need observation in CI; local tests use the existing Docker daemon and emulated ARM.

`scripts/backend_probe.py` checks installed native libraries, Torch CUDA/HIP metadata, generated-image preprocessing, and a tiny randomly initialized CLIP projection. OpenVINO converts the same model, saves/reloads IR in an owned cache subdirectory, compiles on CPU, and compares its first output against Torch. CUDA runs and compares an actual GPU projection when a GPU is available; `--require-gpu` forbids a CPU-only result.

`scripts/smoke_backend.py` checks dependency consistency and non-root cache access, then starts Uvicorn with warmup disabled and an ephemeral API key. It verifies health, readiness false without production weights, protected-endpoint refusal and authenticated access, and graceful shutdown. Only loopback is used; networking is disabled externally. No production weights or remote model code are downloaded. The scripts are Python; no CommonJS code is introduced.

## Outcome and remaining work

The complete suite passed 252 tests on Python 3.12 in the new CPU dependency environment, after installing the patched setuptools floor. Coverage reached 92.97% lines and 85.48% branches, above the unchanged committed floor. Strict OSV auditing covered all 79 installed test-environment packages with zero findings and zero skips. Audits of actual runtime images also passed without findings or skips: CPU amd64/arm64 (56 packages each), modern CUDA amd64 (75), legacy CUDA amd64 (75), and OpenVINO amd64 (58).

All five rebuilt images passed their shipped-script smokes. CPU ARM ran under local emulation. Modern CUDA's tiny projection executed and matched CPU on an RTX 5070 Ti with driver 616.92, and the HTTP service selected CUDA. Legacy CUDA passed its offline CPU projection/startup without a GPU. OpenVINO's actual conversion/reload check passed on 2026.4.0. All validated targets retain non-root cache access, authentication checks, and graceful shutdown. A deliberately mismatched CUDA 12.6 base environment with the modern Torch wheel failed as expected. The Intel identity collision is repaired. Ruff, Pyright, actionlint (without external ShellCheck/Pyflakes), copyright checks, Python compilation, coverage ratchet, and whitespace checks passed.

This iteration verifies small model and container contracts, not production-model quality, native ARM execution, Intel GPU support, or legacy NVIDIA hardware inference. GitHub-hosted jobs have not been exercised locally. CUDA ARM's metadata failure remains an explicit support limit. The subsequent [input budgets](input-memory-budgets.md), [remote destination boundary](remote-image-destinations.md), and [aggregate ingress/deployment controls](aggregate-ingress.md) have separate outcomes. The [production model/artifact iteration](model-artifact-contracts.md) now records pinned CPU/OpenVINO production parity, versioned IR and native peak measurements, plus new CPU/OpenVINO rebuilds. Next calibrate maximum-batch/multi-model capacity, complete dependency locks and remaining CI pins; restore CUDA ARM after an upstream-compatible fix. No release is created.
