# PR 48: local NumPy minimum review

Assessment date: 2026-10-03. [Open PR 48](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/48) was selected uniformly from eligible open PRs `[46, 48]` using draw `0.5890290584185915`. Reviewed head: `c17fe91c0ff1716dd2d927a0952d0d278143b37b`. Apply its NumPy minimum locally on master; leave the original PR unmerged.

## Research, alternatives and design

GitHub MCP enumerated eleven open PRs and fetched current diffs. PR 46 now proposes a new FastAPI 0.142.2 minimum and was eligible again. PRs 53/35 propose superseded framework/backend floors; 42/40/34/33/27 were already adopted; 41's action was replaced by verified native scanning; 44 proposes an expired interim Ubuntu release. Selection did not favor a particular dependency.

The publisher's [NumPy 2.5.3 release](https://github.com/numpy/numpy/releases/tag/v2.5.3), discovered from the fetched PR and opened, documents patch fixes and Python 3.12–3.15 support. All current production/QA locks already use 2.5.3. A minimum-only change aligns unlocked installation with the reviewed wheel; it neither refreshes transitive dependencies nor claims arbitrary future versions are validated.

| Choice | Pros | Cons | Recommendation |
|---|---|---|---|
| Retain `numpy>=2.4` | Wider unlocked compatibility | Permits versions older than the reviewed graph | Replace |
| Adopt `numpy>=2.5.3` and preserve complete locked graphs | Small reproducible change; current numerical behavior retained | New minimum excludes older Python/NumPy combinations | Adopt for supported Python 3.12 profiles |
| Refresh all graphs | Newer packages if available | Broadens regression scope without need | Defer to a dedicated update |

Re-attest affected lock input hashes after checking every locked NumPy satisfies the new minimum. Preserve every artifact/version/URL/hash and rendered lock. Test numerical/model/API behavior locally. All nine lock manifests pass their contracts: 446 wheel records and all rendered lock hashes remain unchanged; only six affected requirements-input attestations changed. Exact installed inventory and `pip check` pass for QA, CPU amd64/arm64, OpenVINO CPU, modern CUDA and legacy CUDA. Actual CPU/OpenVINO/NVIDIA maximum-batch and mixed-input parity checks pass. Complete QA, quality and security outcomes are recorded in the [capacity archive](validation/deployment-capacity-2026-10-03.json). The original PR is adopted locally without a merge operation, branch, tag, version bump or release.
