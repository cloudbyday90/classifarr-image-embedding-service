# Local implementation of open PR 40

Decision date: 2026-10-01. Status: implemented and locally validated.

## Discovery and scope

GitHub MCP found 16 open PRs. Excluding already implemented PRs 28 and 42, a uniform random selection from the remaining 14 chose [PR 40](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/40): update the direct Google OSV scanner action from 2.3.5 to 2.3.8. Separate MCP calls confirmed open/unmerged state and retrieved its one-file patch.

Inspected PR head: `a28cf48f141fa165599d0b9d93341c212664af48`; base: `a764ad47d9ba29d5bb674ed07c04134283811292`. Local work starts from `c6e0f40589acf7e0cb6989ddf96129a122d3e1ab`, preserving prior admission changes. The PR is implemented locally, not merged. The separate reusable PR/release workflow references remain outside this PR's scope.

## Official research and design

MCP fetched Google's [2.3.8 action manifest at its verified commit](https://github.com/google/osv-scanner-action/blob/9a498708959aeaef5ef730655706c5a1df1edbc2/osv-scanner-action/action.yml). The commit and version tag match `9a498708959aeaef5ef730655706c5a1df1edbc2`. This manifest delegates to the mutable image tag `ghcr.io/google/osv-scanner-action:v2.3.8`; a wrapper SHA alone would not freeze the scanner code. Registry inspection/pull verified image digest `sha256:48406c58197201fe55e56615ad9d414f85063da320e204d0b0ed460fb3908dba`.

[Google's action documentation](https://google.github.io/osv-scanner/github-action/) and the manifest recommend reusable workflows for general integration. This Dependabot job specifically avoids SARIF upload with read-only permissions, so preserve that behavior. [GitHub documents passing container-action arguments to the image entrypoint](https://docs.github.com/en/actions/reference/workflows-and-actions/metadata-syntax?apiVersion=2022-11-28). Execute the verified image directly using `docker://...@sha256:...`, with `args: --recursive ./`, retaining the actual upstream entrypoint and scan arguments.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Copy the PR's mutable 2.3.8 action tag | Exact textual match; automatic tag updates | Both wrapper and delegated image can move | Reject for execution reproducibility |
| Pin the wrapper commit | Freezes the small manifest | Still executes a mutable container tag | Insufficient |
| Pin the actual 2.3.8 image digest | Freezes executed scanner and wrapper; preserves existing scan behavior | Future image upgrades need digest verification and review | Adopt |
| Replace this job with Google's reusable workflow | Aligns with upstream's preferred general integration | Changes the Dependabot permission/SARIF behavior beyond this PR | Separate follow-up |

No registry credentials, API write permissions, CommonJS modules, or runtime service changes are introduced by the PR implementation. Scanner/image update automation and the two remaining reusable workflow upgrades require separate review.

## Local validation and outcome

The actual pinned image reports OSV scanner 2.3.8, SCALIBR 0.4.5, scanner commit `408fcd6f8707999a29e7ba45e15809764cf24f67`. Inspection confirmed its Bash entrypoint forwards scan arguments and propagates vulnerability failure.

Run the image against read-only temporary requirements fixtures using public vulnerability metadata, dropped capabilities, and no privilege escalation:

- Clean fixture `numpy==2.4.6`: exit 0, empty findings.
- Vulnerable fixture `requests==2.19.1`: exit 1, reports Requests and its vulnerable IDNA/urllib3 dependencies, including `GHSA-9hjg-9r4m-mvj7` and `PYSEC-2018-28`.

This verifies real scanner execution and failure propagation, not merely workflow syntax. An initial Requests 2.33.1 fixture resolved a vulnerable transitive IDNA version in the scanner's dependency metadata, so the clean test uses a current NumPy fixture without that ambiguity. No findings were suppressed. The tests use isolated fixture mounts, not repository credentials or application data.

Workflow lint passed with actionlint (external ShellCheck/Pyflakes disabled), and the complete service suite passed 252 tests. Full hosted-runner execution is not simulated locally. GitHub MCP reconfirmed the original PR remains open and unmerged before delivery. No review/comment, closing keyword, release, or version bump is part of delivery to `fix/backend-build-contracts`.
