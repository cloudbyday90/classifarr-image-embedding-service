# Python and Rust platform decision

Assessment date: 2026-10-02. Outcome: retain Python for this service; consider a measured, bounded Rust component before any platform rewrite.

## Evidence and official sources

URLs were discovered through web search and followed from the projects' own documentation. [PyTorch's frontend guidance](https://docs.pytorch.org/cppdocs/frontend) explains that expensive Python tensor operations already call native C++ and recommends Python when deployment requirements permit it. Changing the HTTP language alone therefore does not establish faster model inference.

[Rust's ownership documentation](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) describes compile-time memory-safety guarantees without garbage collection. [Hugging Face Candle](https://github.com/huggingface/candle) provides Rust inference with CPU/CUDA backends and lightweight-binary goals; its [CLIP example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/clip) demonstrates a relevant model family. [Intel's OpenVINO Rust bindings](https://github.com/intel/openvino-rs) offer high-level bindings, backed by an unsafe native C API and shared libraries. These are credible options to investigate, rather than proof of parity with this service's production models, CUDA profiles or OpenVINO export/cache path.

Local code already separates admission, inference ownership, lifecycle, input accounting and remote transport. Native projection/container checks exist for the current Python stack; earlier backend records include CUDA/OpenVINO validation. No comparative Rust latency, throughput, memory or embedding-quality benchmark has been run. The recommendation below is an engineering judgment based on those contracts and migration cost, not a measured language-performance result.

## Options and tradeoffs

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Python/FastAPI with native inference | Keeps Hugging Face preprocessing, Torch/OpenVINO integrations and tested API/queue/cache behavior; fastest route to current security fixes | Interpreter/startup footprint; Python orchestration and preprocessing can limit some workloads; deployment memory still needs sizing | Adopt now |
| Rust for one measured component | Memory-safe application code and explicit ownership; can isolate CPU-bound orchestration or reduce ingress overhead | FFI or IPC, additional packaging and failure contracts; native codecs/tensor libraries remain security boundaries | Prototype only when measurements justify it |
| Full Rust inference/API rewrite | Could reduce Python dependencies and produce a smaller deployment with a suitable runtime | Must re-establish preprocessing/output parity, model compatibility, backend support, cancellation, batching, limits, cache and operational behavior | Defer until a bounded prototype proves a material benefit |

Rust does not automatically fix SSRF, unsafe redirects, oversized inputs or authentication policy. Implement those boundaries explicitly in either language. Safe Rust also cannot provide memory-safety guarantees for every external native library behind FFI. The current remote-destination fix belongs at a shared transport boundary and is implemented as small Python service modules.

## Gate for a future Rust prototype

1. Profile fixed hardware, pinned production model revisions and representative image sizes. Measure stage times for download, decoding/preprocessing, inference, queue wait and serialization, plus p50/p95/p99 latency, sustained throughput, RSS/VRAM and startup time.
2. Select a component whose measured cost matters. Agree on an improvement target before building; record maintenance and packaging costs alongside performance.
3. Establish production image/preprocessor fixtures and numerical tolerances for each backend. Verify projection dimensions, normalization, cache identity, resize/crop semantics and batching against those fixtures. Keep current policy and error/shutdown regressions.
4. Compare the prototype on the same hardware, versions and workload. Retain Python inference if the prototype does not demonstrate sufficient end-to-end benefit or cannot match required CPU/CUDA/OpenVINO behavior.

## Design outcome and next work

The user has confirmed that the platform remains Python; no Rust rewrite or new runtime service is introduced. Authored JavaScript/TypeScript must use ES Modules; external CommonJS packaging was explicitly accepted for [PR 41](pr-41-local-validation.md). Continue extracting cohesive services instead of enlarging the inference class. The subsequent [aggregate ingress/deployment protection](aggregate-ingress.md), [production-model/artifact contracts](model-artifact-contracts.md), [CPU/OpenVINO capacity calibration](capacity-calibration.md), and [hash-locked dependency profiles](dependency-locks.md) with their [update policy](runtime-update-policy.md) are implemented and locally validated. Next complete remaining CI references and native tool verification. The pinned fixtures also support a future measured language evaluation. See the [recommendation stack](recommendation-stack.md) and [remote destination design](remote-image-destinations.md).
