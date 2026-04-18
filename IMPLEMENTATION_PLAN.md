# Embedding Math And Invariant Hardening Plan

## Goal

Fix the correctness issues found in the embedding pipeline and close the test gaps that currently allow mathematically invalid or misleading responses to pass.

## Priority Summary

### P0: Cache correctness

1. Align cache-key precedence with actual input precedence.
   - Current problem: cache keys prefer `image_url`, while embedding prefers `image_base64` when both are present.
   - Files:
      - `src/image_embedder/embedder.py`
      - `src/image_embedder/models.py`
      - `src/image_embedder/routes/embed.py`
      - `src/image_embedder/routes/batch.py`
   - Research-backed direction:
      - Pydantic's latest validator guidance recommends model-level validation for cross-field constraints. This is the right place to enforce "exactly one of `image_url` or `image_base64`" at the request-model layer instead of relying on route-level branching.
      - Best practice is to validate the selector before any cache interaction so there is only one canonical input path per request.
   - Recommended implementation:
      - Add `@model_validator(mode="after")` to `EmbedImageRequest`.
      - Add the same invariant to `EmbedBatchItem`.
      - Enforce: one and only one of `image_url` / `image_base64` must be present.
      - Remove ambiguous dual-input behavior entirely rather than preserving it.
      - Introduce a single helper that derives the cache identity from the effective input actually embedded.
   - Design notes:
      - Canonicalization should happen once, near request parsing, not separately in routes and embedder methods.
      - The cache-key helper should consume a canonical "image source" object, not loose optional fields.
      - This should eliminate precedence drift between `embed()`, `embed_batch()`, and route pre-validation.
   - Concrete tasks:
      - Add request-model validation for XOR semantics.
      - Update endpoint tests to expect HTTP 422 for dual-input payloads.
      - Refactor `EmbeddingLRUCache.make_key()` so it is driven by canonicalized input, not raw option precedence.
   - Acceptance criteria:
      - A request with both fields is rejected before reaching the embedder.
      - Cache identity is derived from the same source that `_load_image()` consumes.
      - Single and batch endpoints behave identically.
   - Preferred outcome:
      - No request path in the system can produce two different interpretations of the same payload.

2. Stop caching remote images by URL string alone.
   - Current problem: a stable URL can serve different bytes over time, producing stale embeddings.
   - Files:
      - `src/image_embedder/embedder.py`
   - Research-backed direction:
      - RFC 9111 treats URI reuse alone as insufficient when representation freshness matters; caches revalidate stored responses using validators such as `ETag` and `Last-Modified`.
      - For this service, correctness matters more than URL-level cache hit rate. The current URL-string key is not a safe representation key.
   - Decision:
      - P0 will keep remote caching enabled, but the cache key will be derived from fetched image bytes instead of the URL string.
      - We are explicitly not implementing full validator-aware HTTP revalidation in P0.
   - Why this decision:
      - It fixes the real bug directly: the cache must represent the actual bytes embedded, not the locator used to fetch them.
      - It preserves cache value for repeated remote-image content instead of disabling remote caching entirely.
      - It keeps P0 scoped to embedding-cache correctness instead of turning it into a full HTTP-cache feature.
   - Recommended implementation:
      - For remote inputs:
        - fetch the response bytes first
        - compute `sha256(fetched_bytes)`
        - build the embedding cache key from that digest plus `model | image_size | normalize`
      - For base64 inputs:
        - decode the payload first
        - compute `sha256(decoded_bytes)`
        - build the cache key from bytes, not the raw base64 string
      - Unify both paths under one rule:
        - cache identity = actual image bytes + embed parameters
   - Refactor shape:
      - Introduce a helper that resolves the effective image bytes before PIL decoding.
      - Split the current flow into:
        1. resolve bytes
        2. compute content-derived cache key
        3. check cache
        4. decode uncached bytes into PIL image
        5. run inference
      - Apply the same structure to `embed_batch()` so each item is keyed from its actual bytes before inference.
   - Design notes:
      - Byte-hash keys are the simplest correctness-preserving approach because they tie the embedding to the actual representation, not the identifier.
      - This design still requires a network fetch for remote URLs before the cache can be checked; that is acceptable for P0 because the bug is about correctness, not bandwidth optimization.
      - The cache will save model inference work for identical remote content, but it will not act as an HTTP freshness cache.
   - Concrete tasks:
      - Replace selector-string cache-key generation with byte-derived cache-key generation.
      - Refactor `_fetch_image_bytes()` and `_decode_base64()` callers so the bytes are available before cache lookup.
      - Refactor `embed()` to check the cache after bytes are resolved but before PIL decoding and inference.
      - Refactor `embed_batch()` to resolve bytes per item, compute per-item keys, and skip PIL decode / inference for byte-cache hits.
   - Acceptance criteria:
      - Changing content at the same URL cannot return a stale embedding from cache.
      - Remote-cache correctness does not depend on URL immutability assumptions.
   - Preferred outcome:
      - The cache represents image content, not URL identity.
   - Explicit non-decisions for P0:
      - Do not implement `ETag` / `Last-Modified` conditional revalidation yet.
      - Do not store remote validator metadata in the embedding cache yet.
      - Do not attempt to avoid network fetches for remote URLs yet.
      - Do not switch to decoded-pixel canonicalization or image-normalization hashing yet; raw fetched bytes are sufficient for this fix.
   - Deferred phase 2 work:
      - Persist remote response metadata:
        - final URL
        - `ETag`
        - `Last-Modified`
        - content length
        - content type
      - Revalidate remote entries with conditional requests.
      - Handle 304-based embedding reuse without refetching the full payload.

3. Avoid returning mutable cached objects by reference.
   - Current problem: callers can mutate cached embedding lists and corrupt future hits.
   - Files:
      - `src/image_embedder/embedder.py`
   - Research-backed direction:
      - Python's current `copy` documentation is explicit that assignment creates bindings, not copies. Returning the same cached object instance is therefore a shared-mutable-state hazard.
      - Deep copies are not automatically the best default; Python also documents that `deepcopy()` can copy too much. For this cache, immutability is the cleaner strategy.
   - Decision:
      - P0.3 will use immutable internal cache entries instead of copy-on-read as the primary defense.
      - The cached embedding vector will be stored as an immutable tuple of floats.
      - The cached record will be represented as a typed immutable object rather than a raw mutable structure.
   - Why this decision:
      - Python's glossary and container docs treat immutability as a first-class property with thread-safety and stable-hash benefits.
      - Avoiding mutation by construction is better than trying to remember to copy correctly on every cache access.
      - `deepcopy()` is explicitly not the preferred default because the standard library docs warn that it can copy too much.
   - Recommended implementation:
      - Introduce an internal cached-result type, preferably a frozen dataclass with `slots=True`, containing:
        - `embedding: tuple[float, ...]`
        - `dims: int`
        - `provider: str`
        - `model: str`
        - `image_size: int`
      - Store instances of that immutable type inside `EmbeddingLRUCache`.
      - Convert cached entries into fresh outward-facing response values when returning from `embed()` and `embed_batch()`.
      - Keep the external contract unchanged for now:
        - `embedding` returned to callers remains a `list[float]`
        - result tuple shape remains unchanged unless and until a larger refactor is approved
   - Design notes:
      - A frozen dataclass is preferred over a plain tuple for readability, field naming, and future invariant checks.
      - `slots=True` is desirable because this is a hot-path internal structure and Python's dataclass docs support slot generation for lower-overhead instances.
      - A `namedtuple` would also be valid, but the dataclass form is easier to extend with helper methods and explicit typing.
      - The cached `embedding` itself must be immutable even if the record wrapper is frozen; otherwise the wrapper only creates shallow immutability.
      - This change also makes future invariant checks simpler because `dims` and vector length cannot drift due to downstream mutation.
   - Concrete tasks:
      - Define an internal frozen cache-entry type.
      - Add helpers:
        - convert raw inference output to immutable cache entry
        - convert cache entry to outward-facing return tuple with a fresh `list`
      - Refactor `EmbeddingLRUCache.get()` / `put()` call sites to operate on cache entries, not mutable outward-return values.
      - Refactor `embed()` and `embed_batch()` to return fresh response objects from immutable cache entries.
      - Add tests that mutate one returned embedding and assert later cache hits are unchanged.
   - Acceptance criteria:
      - Mutating a returned embedding cannot affect future responses.
      - Cache hits and misses produce equivalent response objects but do not share mutable state.
   - Explicit non-decisions for P0:
      - Do not use `deepcopy()` on every cache get/put as the primary design.
      - Do not leave mutable `list[float]` values stored inside the cache.
      - Do not expose the internal cache-entry type as a public API contract.
   - Fallback option if the refactor proves too invasive:
      - Use shallow-copy-on-read for the outer structure and `list()` for the embedding as an interim safety patch.
      - This fallback should only be used if the immutable-entry refactor blocks P0 progress.

#### P0 rollout plan

1. Implement XOR validation for `image_url` / `image_base64`.
2. Refactor cache-key derivation around resolved image bytes.
3. Replace URL-string-based remote cache keys with byte-derived keys.
4. Convert cache storage to immutable entries.
5. Add regression tests before moving to P1.

#### P0 online references

- Pydantic validators: model-level validation for cross-field constraints
  - https://pydantic.dev/docs/validation/latest/concepts/validators/
- Python `copy` module: assignment does not copy; deep copy tradeoffs
  - https://docs.python.org/3/library/copy.html
- Python `dataclasses`: frozen instances and `slots=True`
  - https://docs.python.org/3/library/dataclasses.html
- Python `collections.namedtuple`: lightweight immutable fielded tuples
  - https://docs.python.org/3/library/collections.html
- Python glossary: immutable objects and thread-safety
  - https://docs.python.org/3/glossary.html
- Python `hashlib`: current standard-library hashing APIs
  - https://docs.python.org/3/library/hashlib.html
- RFC 9111 HTTP caching: validator-based revalidation with `ETag` and `Last-Modified`
  - https://www.ietf.org/rfc/rfc9111.pdf

### P1: Output invariants

1. Enforce embedding width invariants.
   - Current problem: `dims` is derived from `len(embedding)` and never checked against `ModelSpec.dims`.
   - Files:
      - `src/image_embedder/embedder.py`
      - `src/image_embedder/routes/embed.py`
      - `src/image_embedder/routes/batch.py`
   - Research-backed direction:
      - FastAPI's response-model docs state that `response_model` is used for validation and filtering of output data, but the application still needs to provide semantically correct values. A structurally valid response is not enough if its metadata is wrong.
      - Hugging Face's CLIP docs are explicit that `CLIPVisionModelWithProjection` exposes `outputs.image_embeds`, and CLIP vision configs have a fixed model `image_size` and projection dimensionality per checkpoint.
   - Recommended implementation:
      - Validate every successful embedding result inside the embedder before it is returned:
        - `len(embedding) == dims`
        - `dims == spec.dims`
        - the embedding is one-dimensional
        - every value is finite
      - Treat width mismatches as internal errors, not recoverable partial success.
      - Keep the invariant check in the embedder, then optionally assert again at the route boundary for defense in depth.
   - Design notes:
      - The embedder is the narrowest correctness choke point; both single and batch flows pass through it.
      - Route-level checks are still useful because the current route contract trusts embedder-returned metadata.
      - A helper such as `_validate_embedding_result(spec, embedding, dims)` should be reused across torch and OpenVINO paths.
   - Concrete tasks:
      - Add a shared embedding-result validator in `embedder.py`.
      - Apply it in `embed()` and `embed_batch()` before caching and before returning.
      - Add failure-path tests for nested arrays, wrong width, wrong `dims`, and non-finite values.
   - Acceptance criteria:
      - Impossible metadata such as `model=ViT-L-14` with `dims=3` is rejected.
      - Any successful embedding result has a one-dimensional vector whose width equals `spec.dims`.

2. Canonicalize response metadata.
   - Current problem: routes trust embedder-returned `model` and `image_size` rather than the resolved request values.
   - Files:
      - `src/image_embedder/routes/embed.py`
      - `src/image_embedder/routes/batch.py`
   - Research-backed direction:
      - FastAPI response models validate output shape and type, but not domain-specific semantics like "this embedding actually belongs to the requested model".
      - Best practice here is to canonicalize metadata at the application boundary from the resolved request context, not from a lower-level tuple that could drift.
   - Recommended implementation:
      - Resolve `spec` and `target_size` once per request path.
      - Return canonical response metadata from that resolved context.
      - Treat embedder-returned `model` and `image_size` as assertions to check internally, not as authoritative response fields.
   - Design notes:
      - The top-level batch response already partially does this; the per-item response should follow the same rule.
      - This change makes the API contract deterministic even if internals regress.
   - Concrete tasks:
      - Update `routes/embed.py` to populate response `model` and `image_size` from resolved values.
      - Update `routes/batch.py` to do the same for per-item success results.
      - Add tests where the embedder intentionally returns mismatched metadata and assert that the route fails or overwrites it.
   - Acceptance criteria:
      - Responses cannot claim a different model or image size than the one actually resolved for the request.
      - Batch top-level and per-item metadata follow the same source of truth.

### P1: OpenVINO hardening

1. Stop assuming `result[0]` is always `image_embeds`.
   - Current problem: OpenVINO output selection is positional and unvalidated.
   - Files:
      - `src/image_embedder/embedder.py`
   - Research-backed direction:
      - Hugging Face documents `CLIPVisionModelWithProjection` as producing `outputs.image_embeds`.
      - OpenVINO's current docs support selecting outputs by index or tensor name, and conversion docs support restricting conversion to only necessary outputs.
      - OpenVINO also warns that tensor names are not guaranteed to exist, so relying on names alone is weaker than selecting the right output at conversion time and validating shape afterward.
   - Recommended implementation:
      - Preferred: during `openvino.convert_model(...)`, specify the required output so the exported IR contains only the projection embedding output.
      - At inference time, retrieve the sole output explicitly and validate shape before use.
      - If single-output export is not practical, inspect available outputs, prefer the output that matches the expected shape, and fail if there is ambiguity.
   - Design notes:
      - Conversion-time output selection is the cleanest fix because it shrinks the IR surface and removes positional ambiguity.
      - Runtime shape validation is still required because a wrongly exported model can otherwise look superficially valid.
      - The bad failure mode here is subtle: a 2-D tensor can pass through normalization with a Frobenius norm and produce misleading `dims`.
   - Concrete tasks:
      - Prototype `openvino.convert_model(..., output=...)` for the CLIP projection output.
      - Validate what output names are actually preserved for this model family.
      - Add explicit post-inference shape checks:
        - single: `(spec.dims,)`
        - batch: `(N, spec.dims)`
   - Acceptance criteria:
      - Wrong output ordering cannot silently produce nested embeddings or bogus `dims`.
      - OpenVINO inference either returns the projection embedding shape or fails loudly.

2. Harden OpenVINO normalization.
   - Current problem: raw `np.linalg.norm()` with only `norm > 0` is weaker than the torch path and does not explicitly reject non-finite data.
   - Files:
      - `src/image_embedder/embedder.py`
   - Research-backed direction:
      - PyTorch's current `torch.nn.functional.normalize` docs define normalization as dividing by `max(||v||_p, eps)`, with default `eps=1e-12`.
      - Matching that behavior on the NumPy/OpenVINO path is the correct way to keep backend semantics aligned.
   - Recommended implementation:
      - Introduce a NumPy normalization helper that mirrors PyTorch's `normalize(..., p=2, dim=-1, eps=1e-12)` semantics for one vector or a batch of vectors.
      - Reject non-finite input vectors before normalization and reject non-finite output vectors after normalization.
      - Normalize along the last dimension only.
   - Design notes:
      - Backend parity matters more than micro-optimizing the NumPy code path.
      - A dedicated helper also makes test expectations clearer and avoids duplicating norm logic in single and batch paths.
   - Concrete tasks:
      - Add `_normalize_embedding_np()` in `embedder.py`.
      - Use it in both OpenVINO single and batch flows.
      - Add tests covering zero vectors, tiny norms, `NaN`, and `Inf`.
   - Acceptance criteria:
      - Zero or non-finite vectors cannot produce unstable normalized output.
      - OpenVINO normalization semantics match the torch path for valid vectors.

### P1: `image_size` semantics

1. Decide whether variable `image_size` is actually supported.
   - Current problem: the service reports arbitrary `image_size`, but the CLIP processor path appears to keep default crop semantics.
   - Files:
      - `src/image_embedder/embedder.py`
      - `src/image_embedder/models.py`
      - `README.md`
   - Research-backed direction:
      - Hugging Face's CLIP vision config docs describe model `image_size` as the input resolution of the checkpoint.
      - Hugging Face image-processor docs show that center cropping is a distinct preprocessing step and that crop behavior is tied to `crop_size`, not just `size`.
      - That means "set shortest edge only" is not sufficient evidence that a new requested resolution is actually what reaches the model.
   - Recommended implementation options:
      - Preferred P1 choice: reject non-default `image_size` for current CLIP models and document that these checkpoints are fixed-resolution in this service.
      - Alternative: if variable sizes are intentionally supported, explicitly configure both resize and crop behavior and prove it with tests.
   - Design notes:
      - Rejecting unsupported sizes is the safer path because it aligns the public API with model reality immediately.
      - Supporting arbitrary sizes should be treated as a feature, not as an incidental byproduct of passing `size={"shortest_edge": ...}`.
   - Concrete tasks:
      - Decide whether to freeze `image_size` to `spec.image_size` in P1.
      - If freezing:
        - add request validation that only `spec.image_size` is allowed
        - update docs and tests
      - If supporting:
        - set both resize and crop behavior explicitly
        - add tests that inspect processor calls and resulting tensor shapes
   - Acceptance criteria:
      - Returned `image_size`, cache key behavior, and actual model preprocessing all mean the same thing.
      - The service no longer advertises a size that it does not truly honor.

#### P1 rollout plan

1. Add embedder-level result validation helpers.
2. Canonicalize route response metadata from resolved request context.
3. Harden OpenVINO output selection and shape checks.
4. Align OpenVINO normalization with PyTorch semantics.
5. Resolve and document `image_size` behavior.

#### P1 online references

- FastAPI response models: validation, conversion, and filtering of outputs
  - https://fastapi.tiangolo.com/tutorial/response-model/
- Hugging Face CLIP docs: `CLIPVisionModelWithProjection` and `outputs.image_embeds`
  - https://huggingface.co/docs/transformers/en/model_doc/clip
- Hugging Face image processor docs: center crop and preprocessing stages
  - https://huggingface.co/docs/transformers/main/main_classes/image_processor
- OpenVINO compiled model outputs API
  - https://docs.openvino.ai/nightly/api/ie_python_api/_autosummary/openvino.CompiledModel.html
- OpenVINO model conversion parameters, including output selection
  - https://docs.openvino.ai/2026/openvino-workflow/model-preparation/conversion-parameters.html
- OpenVINO model representation and output access caveats
  - https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/model-representation.html
- PyTorch normalization semantics
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

## Test Plan

### Traceability matrix

#### P0.1: XOR input validation

1. Add request-model validation tests for `EmbedImageRequest`.
   - Accepts exactly one of `image_url` / `image_base64`.
   - Rejects both missing.
   - Rejects both present.

2. Add request-model validation tests for `EmbedBatchItem`.
   - Accepts exactly one source per item.
   - Rejects dual-input items with HTTP 422 at the API boundary.

3. Update route tests.
   - `POST /embed-image` returns 422 for dual-input payloads.
   - `POST /embed-batch` returns 422 when any item uses both fields.

#### P0.2: Remote cache correctness

1. Add cache-key tests for remote inputs.
   - If remote caching is disabled, repeated URL requests never hit the embedding cache.
   - If remote byte-hash caching is implemented, same bytes at different URLs can reuse the cache, and changed bytes at the same URL cannot.

2. Add remote fetch regression tests.
   - Simulate one URL returning two different payloads across requests.
   - Assert that the second request does not return the first embedding from cache.

3. If validator-based remote caching is added later:
   - Add tests for `ETag` / `Last-Modified` revalidation behavior.
   - Add tests for fallback when validators are absent.

#### P0.3: Cache immutability

1. Add cache hit mutation tests.
   - Fetch one embedding.
   - Mutate the returned list.
   - Fetch the same input again.
   - Assert the second result is unchanged and still internally consistent.

2. Add batch cache mutation tests.
   - Mutating one per-item batch result must not corrupt future hits for that item.

### P1.1: Embedding width and finiteness invariants

1. Add embedder-level invariant tests.
   - Reject nested arrays.
   - Reject `dims != len(embedding)`.
   - Reject `dims != spec.dims`.
   - Reject `NaN` and `Inf`.

2. Add route-level invariant tests.
   - Fake embedder returns malformed tuples.
   - Route fails instead of serializing impossible metadata.

### P1.2: Canonical response metadata

1. Add single-route metadata tests.
   - Fake embedder returns the wrong `model`.
   - Fake embedder returns the wrong `image_size`.
   - Route either overwrites with canonical resolved values or fails loudly, depending on final implementation choice.

2. Add batch-route metadata tests.
   - Top-level and per-item metadata use the same source of truth.
   - Per-item mismatches cannot leak into successful responses.

### P1.3: OpenVINO output selection and shape validation

1. Add OpenVINO output-shape tests.
   - Single inference returning `(dims,)` succeeds.
   - Batch inference returning `(N, dims)` succeeds.
   - Returning `(N, patches, dims)` or any nested result fails.

2. Add OpenVINO output-identity tests.
   - If conversion-time output selection is used, assert the exported model exposes only the intended output.
   - If runtime selection is used, assert ambiguous output sets fail loudly.

### P1.4: OpenVINO normalization parity

1. Add deterministic normalization tests.
   - Nonzero vectors produce unit L2 norm within tolerance.
   - Zero vectors do not produce `NaN`.
   - Tiny norms use epsilon-clamped behavior.
   - `NaN` / `Inf` inputs are rejected.

2. Add parity tests versus torch semantics.
   - For the same sample vectors, NumPy/OpenVINO normalization matches the torch path closely enough for production use.

### P1.5: `image_size` semantics

1. If non-default sizes are rejected:
   - Add request validation tests that only the checkpoint-default size is accepted.
   - Add route tests that non-default size requests fail clearly.

2. If non-default sizes are supported:
   - Assert both resize and crop settings are passed explicitly.
   - Assert final tensor shape matches the requested effective model input size.

### Existing test cleanup

1. Replace mathematically impossible fake responses.
   - Current fake embedder returns 3-value vectors for models that claim 512/768 dims.
   - Files:
     - `tests/fakes.py`
     - `tests/test_api.py`
     - `tests/test_batch_endpoint.py`
     - `tests/test_integration.py`
   - Cleanup goal:
     - Fakes should preserve `len(embedding) == dims`.
     - Model-specific fakes should preserve `dims == spec.dims`.

2. Strengthen normalization tests.
   - Assert actual L2 unit norm, not just that normalization was called.
   - Cover zero-vector and non-finite-vector behavior.
   - Files:
     - `tests/test_embed_flow_mocked.py`
     - `tests/test_embedder_advanced.py`

3. Add one explicit regression test per identified defect.
   - Cache dual-input poisoning.
   - URL-string stale remote cache.
   - Mutable cache result corruption.
   - Wrong `dims` acceptance.
   - Wrong `model` / `image_size` acceptance.
   - OpenVINO wrong-output acceptance.
   - OpenVINO normalization instability.
   - Misleading `image_size` behavior.

## Suggested Execution Order

1. Fix cache-key correctness and immutability.
2. Enforce response invariants in the embedder and routes.
3. Harden OpenVINO output selection and normalization.
4. Resolve `image_size` semantics and update docs.
5. Tighten tests and replace invalid fakes.

## Validation Commands

```powershell
.\scripts\dev-shell.ps1
pytest tests\test_embedder.py tests\test_embedder_advanced.py tests\test_embed_flow_mocked.py tests\test_batch_endpoint.py -q
python -m pytest -q
```
