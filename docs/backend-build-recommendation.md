# Backend build and dependency validation

Assessment date: 2026-10-01. Status: next recommended task; no backend implementation change in this commit.

## Evidence and official sources

`Dockerfile.cuda` specifies `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04`. A fresh read-only registry check, `docker buildx imagetools inspect nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04`, returned "not found" on 2026-10-01. It installs Torch from the cu124 wheel index before installing the general `torch>=2.10` runtime requirement. The base-image and wheel/dependency contracts therefore require explicit validation before selecting replacements.

The [official NVIDIA image repository](https://hub.docker.com/r/nvidia/cuda/) was discovered through web search. Published newer tags alone do not establish compatibility with this service, its Torch wheels, or user drivers. Research the supported Torch/CUDA/driver matrix and verify image manifests rather than substituting the newest tag by assumption.

## Alternatives and recommendation

| Option | Benefit | Cost or risk | Recommendation |
|---|---|---|---|
| Change only the missing image tag | Small patch | Does not resolve wheel selection, runtime compatibility, architecture support, or reproducibility | Insufficient |
| Separate validated CPU, CUDA, and OpenVINO dependency profiles with build/import smoke checks | Detects invalid bases, dependency conflicts, and backend import regressions before release | More CI time and explicit version maintenance | Adopt next |
| Run real GPU/model inference for every PR | Strongest hardware correctness evidence | Requires GPU runners, model assets, time, and backend-specific coverage | Add selectively after builds are reliable |

Use small backend-specific dependency files and a shared validation script where behavior overlaps. Validate base manifests for each advertised architecture, resolve the chosen dependencies from their intended indexes, run `pip check`, import backend modules, and exercise service startup without a model download. Pin validated image digests and document driver/runtime requirements. Keep actual model inference as a separate hardware-dependent gate.

## Acceptance and current outcome

Before selecting a concrete backend replacement, discover official version and compatibility sources through search/MCP and record exact verified image and package identifiers. Builds should cover CPU, CUDA, and OpenVINO on supported platforms and execute on PRs without publishing containers. Preserve tag-only release publishing. Test CPU import/startup, CUDA availability handling, OpenVINO cache/export paths, and dependency consistency separately.

This commit repairs admission and implements the selected checkout PR locally. It reproduces the missing CUDA manifest and records this plan, but does not claim repaired backend builds or tested GPU compatibility.
