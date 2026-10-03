# Production model, preprocessing and generated IR contracts

Assessment date: 2026-10-02. Baseline: `484c7ed3515e97fe74899d932cb89b4346b21174`. Status: implemented; final production validation recorded below. Python and direct master delivery are user decisions.

## Evidence and official research

The baseline loader fetched model/processor assets without a revision. OpenVINO reused `OV_MODEL_CACHE/<alias>/model.xml` based only on XML existence, without checking the matching BIN, source revision, export/runtime versions or integrity. It compiled the original converted graph on first export but saved weights with OpenVINO's default FP16 compression, so a later reload could use different precision. Existing small native probes did not establish production-model preprocessing or cache contracts.

Official sources were discovered through web search, followed links and GitHub MCP. [Hub download guidance](https://huggingface.co/docs/huggingface_hub/en/guides/download) documents immutable full revisions and SDK-derived URLs. [Transformers loading APIs](https://huggingface.co/docs/transformers/main_classes/model) support revision selection, restricted weight loading and explicit format choice. The [CLIP processor reference](https://huggingface.co/docs/transformers/model_doc/clip) documents the PIL implementation. [OpenVINO's 2026 IR guide](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-to-ir.html) describes XML/BIN serialization, reload and default FP16 compression. [Filelock guidance](https://py-filelock.readthedocs.io/en/latest/how-to.html) describes process locks with finite acquisition timeouts.

GitHub MCP also verified the explicit [PIL processor source at the declared Transformers 5.10.1 floor](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/clip/image_processing_pil_clip.py). The class exists at that minimum; full native runtime validation here uses installed 5.18.0. Hugging Face Hub and filelock become explicit direct dependencies at their tested installed floors, 1.33.0 and 3.32.3, rather than relying solely on transitive installation.

The official [large-model commit](https://huggingface.co/openai/clip-vit-large-patch14/commit/32bd64288804d66eefd0ccbe215aa642df71cc41) includes safetensors; the official [base-model commit](https://huggingface.co/openai/clip-vit-base-patch16/commit/57c216476eefef5ab752ec549e440a49ae4ae5f3) supplies PyTorch weights. Do not silently substitute an unmerged third-party safetensors conversion. Use restricted `weights_only=True` with the patched Torch profile for the base model, pin the publisher revision and verify source asset digests before loading.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Tests alone, mutable models and alias-only IR | Smallest change | Future asset drift and stale/partial IR remain possible | Reject |
| Pinned verified model assets, explicit PIL preprocessing and versioned atomic IR | Stable identity; mismatches fail/rebuild before reuse; small cohesive Python services | Asset review and hashing cost; FP32 IR needs more disk; first export needs memory | Adopt |
| External artifact registry and isolated exporter | Can separate export failures and centralize signed artifacts | New protocol, trust/signing, deployment and recovery ownership | Defer until workload evidence warrants it |

API model names, dimensions, image sizes, successful responses and input/resource ownership stay stable. `model_catalog.py` holds frozen model specs in a read-only catalog, `model_loading.py` verifies publisher assets and restricts local deserialization, and `artifact_integrity.py` streams hashes. `ir_cache.py` owns publication and `openvino_models.py` owns conversion/compilation. The inference class delegates loading to these services. Only the explicit PIL image processor is loaded; tokenizers and remote model Python are unnecessary.

| Alias | Publisher repository | Immutable revision | Projection | Weight format |
|---|---|---|---|---|
| ViT-B-16 | openai/clip-vit-base-patch16 | `57c216476eefef5ab752ec549e440a49ae4ae5f3` | 512 | Verified PyTorch, restricted loading |
| ViT-L-14 | openai/clip-vit-large-patch14 | `32bd64288804d66eefd0ccbe215aa642df71cc41` | 768 | Verified safetensors |

Both use a 224-pixel crop. Exact publisher configuration and preprocessing bytes, digests, asset sizes and SDK-derived URLs are archived in `tests/fixtures/models/sources.json` and its sibling fixtures. Large binary weights are excluded from the repository. Source hashes are checked before deserialization, including each new model owner; warm in-process inference reuses the loaded model.

## Cache design and rollout

IR keys include source identity/revision/digests, preprocessing policy, shape/projection dimensions, complete installed Torch/Transformers/OpenVINO versions, FP32 serialization and accuracy execution policy. An exact manifest and streamed size/hash records for XML/BIN must match before reuse. Generated files must be nonempty regular files; linked entries/files and oversized manifests are rejected. Non-regular manifests are refused before opening, and JSON nesting overflow triggers rebuild. Export occurs in a private staging directory; the complete directory is published atomically under a separate persistent per-key process lock with a 300-second acquisition timeout. NaN, infinite and negative timeouts are refused. Locks are never unlinked while waiters may own their inode. First load and reload both compile the published representation.

[OpenVINO precision guidance](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html) distinguishes stored weights from inference precision and recommends the portable execution-mode hint. The loader requests `ACCURACY`, alongside uncompressed FP32 serialization, to avoid relying on a backend's performance-oriented precision defaults. This can cost disk, startup memory and throughput; accelerator behavior still needs hardware validation.

Entries live under `OV_MODEL_CACHE/<contract-sha256>/`, defaulting to `/app/.cache/ov_ir`. Legacy alias-only entries are ignored and left intact. A new contract creates a separate entry. A corrupt entry is displaced only after its complete replacement is ready. Failed export cannot publish half a pair; process death can leave unused staging directories, which are never accepted as entries. Monitor disk use and perform cache maintenance only while owners are stopped. The manifest schema and preprocessing policy must be bumped when their semantics change.

This cache is trusted local storage. Checksums detect corruption and mismatches; a writer who can replace both artifacts and manifests is outside this integrity boundary. Use application-owned local volumes and supported locking filesystems. Locks bound competing exporters, not stuck native conversion. Existing inference ownership, HTTP deadlines, non-root execution and deployment containment remain complementary.

## Validation and outcome

The final rebuilt Python 3.12 QA environment passed **493 tests**, with **two explicitly opt-in production-weight cases skipped** in routine execution. Line/branch coverage is **94.17% / 88.55%**, above the unchanged **89.42% / 79.37%** floors. Offline fixtures independently reproduce PIL bicubic resize, center crop, rescaling and CLIP normalization for both model aliases and two aspect ratios. Source failures are asserted to stop before deserialization. Generated-cache regressions exercise seven identity axes, damaged/missing/linked/oversized entries, failed export, thread contention and twelve requests from four independent spawned processes publishing one complete pair.

Rebuilt CPU and OpenVINO images passed their shipped offline tiny-native/service smokes, dependency consistency, non-root/cache, authentication and graceful shutdown checks. Strict installed-package OSV audits found zero known vulnerabilities and zero skipped packages across **56 CPU**, **58 OpenVINO** and **67 QA** packages. Ruff, scoped Pyright against the actual Transformers 5.18 API, workflow lint, copyright and coverage ratchet checks passed. Workflow lint disables external ShellCheck/Pyflakes integrations; hosted Actions jobs have not been run locally. No advisory suppression or coverage-floor reduction is used.

Final image inspection also found an inherited NumPy metadata record in the Intel base environment. The [separate runtime-environment design](openvino-runtime-environment.md) documents removal before complete environment copy, the new native duplicate-inventory guard, reproduced negative control and cleaned-image verification. The cleaned image passed a fresh strict audit, shipped smoke and large-model offline cache/parity probe. The last application suite used the audited QA image with the final cache-validation module mounted read-only; final runtime images include that module and pass the recorded native checks.

All four separate production probes used the actual shipped loaders, verified pinned weights and externally disabled networking. Torch is **2.14.1+cpu**, Transformers **5.18.0**, OpenVINO **2026.4.0** on the OpenVINO CPU plugin. Both generated images pass raw single, mixed-normalization batch and protected real `/embed-image` route checks. Service/native Torch parity uses `rtol=atol=1e-4`; published-IR first/reload parity uses `1e-6`. CPU was contained at 4 GiB/no swap and OpenVINO at 8 GiB/no swap. Export/reload tests run on a native Docker cache volume; the initial Windows file-share runs were stopped after kernel I/O waits made their timings unrepresentative.

| Backend/model | Cold loader/export seconds | Same-process fresh-owner IR reload seconds | Service peak RSS before validation oracle | Full probe peak RSS |
|---|---:|---:|---:|---:|
| CPU / ViT-B-16 | 33.81 | N/A | 728.49 MiB | 741.93 MiB |
| CPU / ViT-L-14 | 47.23 | N/A | 1585.73 MiB | 1599.85 MiB |
| OpenVINO CPU / ViT-B-16 | 56.99 | 4.83 | 1593.05 MiB | 2608.64 MiB |
| OpenVINO CPU / ViT-L-14 | 51.81 | 8.57 | 4502.95 MiB | 8021.98 MiB |

RSS comes from Linux `ru_maxrss` for the probe process. The first peak includes model loading/export and the service's single/two-image batch, before loading the extra Torch oracle and second OpenVINO owner. The second peak includes that additional validation work and API check. Cold timings were measured with competing validation/build activity and are observations, not comparable throughput benchmarks. GPU VRAM, simultaneous maximum batches, multiple resident production models and latency/throughput distributions remain unmeasured. Full publisher checkpoints include unused text fields; vision-only loading reports these expected extra keys. Native tracing also emits upstream deprecation/shape warnings; tested single and two-image batch outputs pass the recorded contracts.

A separate fresh container repeated the large OpenVINO probe against the existing native-volume entry. It passed all parity/API checks without re-export: initial cached load was 9.27 seconds, another owner's reload 3.35 seconds, service peak 2783.28 MiB and full validation peak 6355.39 MiB. This confirms reuse across process/container lifetimes as well as the in-process fresh-owner checks.

The large-model cold-export service peak is **4.40 GiB**, exceeding the former 4 GiB default. The OpenVINO Compose override now uses a tunable **8 GiB/no-swap default**; explicit `IMAGE_EMBEDDER_MEMORY_LIMIT` still takes priority. CPU/CUDA retain their initial 4 GiB policy. Raising the bounded OpenVINO default accommodates the measured cold path, costs a larger host budget and does not establish maximum production capacity. An isolated exporter or reviewed lower-precision deployment remains a future alternative; the current change keeps numerical policy explicit. The full large-model validation probe peaked at **7.83 GiB**, close to its 8 GiB allowance, so future runtime changes must rerun and monitor this check.

Compose rendering confirmed default CPU/CUDA memory and combined memory/swap at 4,294,967,296 bytes, OpenVINO at 8,589,934,592 bytes, and an explicit 6 GiB OpenVINO override at 6,442,450,944 bytes for both fields. An actual 8 GiB probe container's cgroup files confirmed equal memory/combined limits, hence no additional swap. The native Gitleaks 8.24.3 changed-file scan found zero secret findings with redacted reporting; this is scoped hygiene validation rather than a repository security audit.

Change-scoped PR/default-branch checks, a weekly schedule and manual dispatch build CPU/OpenVINO images. A networked prefetch verifies exact publisher assets; fresh offline containers compare both production models. Permissions are read-only and local validation logs are retained by a pinned artifact action. This separate workflow does not publish images or change the existing tag-only release gate. Routine pytest never fetches production weights; `RUN_PRODUCTION_MODEL_TESTS=1` enables the two optional CPU cases only with a populated offline cache. The [PR 41 record](pr-41-local-validation.md) documents its own action design and local controls. The changelog stays under Unreleased; no release or upstream PR merge is created.

## Next work

The subsequent [capacity calibration](capacity-calibration.md) exercises both resident models, maximum batches, cold initialization and detached owners on CPU/OpenVINO CPU, with separate deadline and initialization fixes. Its measurements supplement this document's historical small-batch evidence. Keep one worker and existing admission budgets. Next complete hash-locked dependency profiles and their update policy, followed by remaining CI refs and total upload/download deadlines. Accelerator VRAM and compatibility beyond tested devices remain separate hardware gates.
