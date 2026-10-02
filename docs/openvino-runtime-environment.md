# OpenVINO runtime environment publication and package inventory

Assessment date: 2026-10-02. Discovered during the [production artifact iteration](model-artifact-contracts.md), based on `484c7ed3515e97fe74899d932cb89b4346b21174`. The platform stays Python and delivery stays directly on master.

## Evidence and official research

The pinned Intel runtime base already contains files under `/opt/venv`. Copying the builder environment into that directory left NumPy 1.26.4 metadata beside the installed NumPy 2.5.3 metadata. Python imported 2.5.3, but inventory enumeration returned both versions and ordering could change which version an inventory consumer selected. A final comparison against the strict audit snapshot exposed this ambiguity. `pip check` and ordinary native inference had passed, so they did not detect the stale distribution record.

The [official Dockerfile reference](https://docs.docker.com/reference/dockerfile), discovered through web search, documents directory merge behavior for COPY. Replacing matching files does not discard unrelated old files. The local image inspection independently confirmed the resulting two NumPy distribution records; this is a runtime packaging defect, not a reported vulnerability or evidence of malicious code.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Overlay the builder environment | Smallest Dockerfile | Retains stale metadata and other old files; ambiguous audits | Reject |
| Clear the owned environment path, then copy the complete builder environment | One controlled package inventory; preserves builder paths and executable shebangs | Additional runtime build layer | Adopt |
| Move both builder and runtime to a distinct environment path | Explicit separation from vendor defaults | Changes all environment paths and rebuilds dependency layers | Defer; unnecessary for this correction |

The OpenVINO runtime stage removes the fixed `/opt/venv` path inside the disposable image before copying the complete builder environment. This does not perform any host workspace deletion. The source builder's exact Torch/OpenVINO profiles and constrained shared installation remain authoritative.

The small `validate_package_inventory()` helper in `scripts/backend_probe.py` canonicalizes distribution names and rejects duplicates before native model checks. Every shipped backend smoke calls it. It reports package names as CLI diagnostics; no product API schema changes. A stale distribution can now fail backend CI even when import selection happens to work.

## Validation and outcome

The negative control ran the new helper against the image containing both NumPy records and failed with `Duplicate installed package metadata: numpy`. The CPU image passed the same guard. The cleaned OpenVINO image contains **58 unique distributions**, exactly matching the strict audit snapshot's package versions; the actual NumPy import and its sole metadata record both report **2.5.3**. A fresh strict OSV audit on the cleaned image found zero known vulnerabilities and zero skipped packages.

Both final shipped-script smokes passed dependency consistency, the inventory guard, native inference, non-root/cache checks, authenticated service startup and shutdown. OpenVINO also passed native export/reload. The cleaned image's offline large-production-model probe passed single, batch, authenticated API and reload parity against the existing versioned IR. Its service peak was 2787.25 MiB and full validation peak 6351.04 MiB; timing observations remain workload-dependent. The accompanying application suite passed 493 tests with two opt-in cases skipped; its design record contains coverage and the cold-path measurements. Package inventories are hygiene evidence, not a complete security audit. No release, tag or upstream PR merge is created.

## Next recommendation

Include duplicate-inventory checks and exact installed-version checks in the planned hash-locked backend update policy. First complete the [capacity measurements](recommendation-stack.md) selected from the observed 4.40 GiB OpenVINO cold-export peak. Preserve reviewed source revisions, artifact contracts and strict auditing during dependency/base updates.
