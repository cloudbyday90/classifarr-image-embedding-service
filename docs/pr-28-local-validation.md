# Local implementation of open PR 28

Decision date: 2026-10-01. Status: implemented and locally validated.

## Discovery and selection

The GitHub MCP searched open pull requests in `cloudbyday90/classifarr-image-embedding-service` and returned 16 candidates, numbered 27 through 42. A uniform random selection chose [PR 28](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/28), titled "build(deps-dev): update pytest requirement from >=9.0 to >=9.0.3". Metadata retrieved separately confirmed that it was open and unmerged. Its patch was retrieved through the MCP service.

Inspected head: `236363d2d84625e1b684958f57ca2ee8736dc62d`; inspected base: `a764ad47d9ba29d5bb674ed07c04134283811292`. The diff changes one file with one insertion and one deletion. The change was copied locally rather than merging the PR branch.

## Change and rationale

Apply the exact one-line patch in `requirements-dev.txt`: raise `pytest>=9.0` to `pytest>=9.0.3`. This dependency is used by developers and CI, not the production dependency file.

The [official pytest 9.0 changelog](https://docs.pytest.org/en/9.0.x/changelog.html), discovered by web search, documents a temporary-directory security fix and several correctness fixes in 9.0.3. The local Python environment already reports pytest 9.0.3, so the repository change can be exercised directly with that version.

## Tradeoffs and decision

| Choice | Benefit | Cost |
|---|---|---|
| Raise the minimum to 9.0.3 | Excludes older affected test-tool versions and matches the selected PR | Fresh installs can still select newer versions because the requirement is a lower bound |
| Fully lock all development dependencies | Reproducible tool environments | Broader dependency policy and maintenance change than the selected PR |

Adopt the exact PR change. Consider dependency profiles and locking in the separate packaging follow-up. Keep the original PR open and unmerged; implementation here does not submit a review, comment, or merge command.

## Validation and outcome

The exact requirements patch is implemented. All 216 tests passed with pytest 9.0.3 and Python 3.14.5; the 31 new execution tests also passed separately. The coverage ratchet passed at 92.04% lines and 84.83% branches. Copyright compliance, whitespace checks with CRLF support, lint checks, and `python -m pip check` passed.

The original PR was not merged, reviewed, or commented on. Delivery targets the `fix/inference-execution-ownership` feature branch alongside the execution repair. No release tag or package-version bump is part of this work; the commit message does not contain a PR-closing keyword. This validates the selected minimum locally, rather than claiming that every future version permitted by the lower bound is tested.
