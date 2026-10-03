# PR 46: local FastAPI minimum adoption

## Selection and design

GitHub MCP listed 11 open PRs on **2026-10-03**. Eligible, unadopted proposals were **48 (NumPy floor)** and **46 (FastAPI floor)**. A single `Math.random()` draw of **0.6211134341237502** selected pool index 1, [PR 46](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/46). Exclusions: 53 and 35 are superseded by locked Transformers/Torch versions; 44 proposes the end-of-life Ubuntu 25.10 base; 42, 41, 40, 34, 33 and 27 were already adopted or superseded by documented native tooling. The original PR was open and unmerged when fetched.

Reviewed head: `e75d93e3394a9e87f8eb022127c8a5b01136904b`; upstream base: `d72eec64388bd4b5712a31fb1105378c895cc0da`. The fetched [head requirements](https://github.com/cloudbyday90/classifarr-image-embedding-service/blob/e75d93e3394a9e87f8eb022127c8a5b01136904b/requirements.txt) change only `fastapi>=0.135,<1` to `fastapi>=0.142.1,<1`. Implement that intent directly on master, rather than merging or applying the old branch wholesale.

## Research and recommendation

Official [FastAPI release notes](https://fastapi.tiangolo.com/release-notes/) identify 0.142.1 (September 29) as a repeated-router-wrapping fix and 0.142.2 (September 30) as a startup fix for failed automatic OpenTelemetry configuration. Every affected locked profile already selects **0.142.2**. Adopt the PR's 0.142.1 minimum; keep exact 0.142.2 wheel versions, URLs, hashes and complete dependency graphs. Renew only affected input attestations after checking that the reviewed locked versions satisfy the changed requirement.

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| Retain 0.135 floor | No metadata change | Advertises older untested versions despite newer reviewed locks | Reject |
| Adopt 0.142.1 floor with current locks | Implements PR; aligns supported input intent; no runtime graph drift | Minimum is below the exact deployment version | Adopt |
| Refresh all wheels now | Could pick other newer packages | Unrelated graph/backend risk and unnecessary downloads | Defer to update-policy review |

Validate all nine lock contracts, unchanged wheel lock bytes/artifact records, exact installed QA inventory and `pip check`, API and lifespan behavior in full offline QA. Official [async testing guidance](https://fastapi.tiangolo.com/advanced/async-tests/) supports HTTPX ASGI transport with explicit lifespan management; native subprocess tests separately exercise OS ownership.

## Outcome

The sole normalized dependency-input delta is the FastAPI floor. **Six** affected application/QA manifests renew only the `requirements.txt` input hash. Every `.txt` lock, all **446 artifact records across nine profiles**, other input attestations and the three unrelated bootstrap/audit manifests remain unchanged. All nine input contracts pass. Exact installed inventories and `pip check` pass in native QA, CPU amd64/arm64, OpenVINO, CUDA and legacy CUDA images; ARM is emulated and accelerator inventories do not claim device execution.

Full locked QA passes **914 tests**, with seven existing skips and **95.22% / 90.12%** line/branch coverage. Native CPU/OpenVINO/CUDA fetch-owner probes all report FastAPI **0.142.2**. No wheel resolution, runtime graph refresh, model download or device benchmark was needed. The [archive](validation/remote-batch-budget-2026-10-03.json) records image identities, unchanged graph/input comparisons and exact log/source hashes.

GitHub MCP reconfirms the selected PR is **open and unmerged** at the same immutable head before delivery. This work creates no release, tag, version bump or separate branch and does not merge the original PR. Related: [dependency contracts](dependency-locks.md), [batch budget](remote-batch-budget.md), [project skill](project-resource-skill.md).
