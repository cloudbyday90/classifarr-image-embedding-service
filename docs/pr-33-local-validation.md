# Local implementation of open PR 33

Decision date: 2026-10-02. Status: implemented locally and validated; upstream PR remains open and unmerged.

## Discovery and design

GitHub MCP returned 17 open PRs. Excluding previously implemented PRs 28, 40, and 42, uniform random selection from 14 candidates chose [PR 33](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/33), raising pytest-cov's minimum from 7.0 to 7.1.0. Separate MCP calls confirmed open/unmerged status and retrieved its one-file patch. Inspected head: `5054ede6625ba6fc860c3a563644f8402a5f5769`; base: `a764ad47d9ba29d5bb674ed07c04134283811292`.

Apply the dependency floor locally while preserving the already patched pytest 9.0.3 minimum. No upstream merge, review/comment, closing keyword, release, or version bump is requested.

[Official pytest-cov 7.1.0 release notes](https://pytest-cov.readthedocs.io/en/stable/changelog.html) describe consistent total coverage computation across reporting settings, including fail-under behavior. This matters to the project's branch/line coverage ratchet. The release also adjusts sqlite3 ResourceWarning handling. Subprocess coverage remains an explicit coverage.py configuration choice; this update does not enable it.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Keep minimum 7.0 | No dependency-floor change | Allows inconsistent coverage totals described upstream | Reject |
| Raise minimum to 7.1.0 | Small local implementation; current coverage reporting fix | Future releases still resolve unless locked | Adopt |
| Pin all development dependencies | Repeatable toolchain | Broader dependency-lock design and update maintenance | Separate follow-up |

## Validation and outcome

The test container installed pytest-cov **7.1.0** with coverage **7.16.2**, and dependency consistency passed. The complete service suite passed **315 tests**. Coverage is **94.74% lines / 88.48% branches**; the committed ratchet remains **89.42% / 79.37%**. A strict OSV audit of the actual test environment checked 63 installed packages with zero known vulnerabilities and zero skips.

A branch fixture covered 3 of 4 executable lines and 1 of 2 branches. JSON-only reporting and combined JSON/XML/terminal reporting both produced exactly **66.66666666666667%** total coverage. Each configuration exited 0 with a 50% fail-under threshold and 1 with a 90% threshold: four expected results. This confirms the selected plugin's reporting/fail-under behavior without weakening the project's independent line/branch ratchet. Temporary fixture artifacts remain outside the repository.

GitHub MCP reconfirmed the same PR head and open/unmerged status before delivery. Only its dependency-floor change is implemented; the original PR is not merged or modified. No release, tag, version change, or closing keyword is created.
