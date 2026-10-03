# Aging pull request cleanup

Assessment: 2026-10-03. Reviewed master baseline:
`b43272ba9f380f7488f1a70e5dafa70a91754b04`.

## Design and decision

The user requested cleanup of aging PRs after the current iteration found no
suitable unapplied proposal. Review the actual diffs against master, then close
completed, superseded or unsuitable proposals. Age alone does not establish
that a change is obsolete. GitHub MCP discovered the repository URL and its
pull collection; current states and immutable heads were rechecked before each
closure. The [availability assessment](open-pr-availability.md) retains the
heads, official-source support and exclusion reasons.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Keep all proposals open | Preserves a visible backlog | Repeats already completed reviews; obsolete diffs obscure useful work | Reject for these reviewed proposals |
| Merge old branches | Records original PR commits as merged | Can restore obsolete tooling, mutable references or unsupported dependencies | Reject |
| Close with a repository decision record | Clears the queue and preserves review evidence; reopening remains possible | Closed state alone does not explain local adoption | Adopt |

Closure changes only PR state. Original titles, descriptions, commits and
branches remain intact; no new comments or Dependabot ignore commands were
posted. Current immutable actions, native scanners, supported base images and
backend-specific dependency locks remain the recommendation stack. New updates
still require review and affected native validation.

## Outcome

GitHub MCP confirmed each PR as **closed and unmerged** on 2026-10-03:

| PR | Closure reason | Closed at (UTC) |
|---|---|---|
| [#44](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/44) | Ubuntu 25.10 is end of life; retain supported 24.04 | 21:02:13 |
| [#42](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/42) | Checkout v7 already adopted with immutable v7.0.1 | 21:02:14 |
| [#41](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/41) | Gitleaks action already adopted, then replaced by verified native scanning | 21:02:15 |
| [#40](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/40) | OSV update already adopted; strict native entrypoints and installed-environment audits supersede the wrapper | 21:02:16 |
| [#35](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/35) | Invalid ordered local-version label and shared CPU selection superseded by exact backend locks | 21:02:17 |
| [#34](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/34) | HTTPX 0.28.1 floor already adopted | 21:02:18 |
| [#33](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/33) | pytest-cov 7.1.0 floor already adopted | 21:02:19 |
| [#27](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/27) | Trivy action v0.36.0 already adopted with immutable reference | 21:02:20 |

A fresh fetch of the repository-provided open pull collection returns **zero
open PRs**. No original PR was merged. Check the current list in the next
iteration because new proposals can arrive; reopen only when a proposal has
renewed applicable work. This cleanup requires no application behavior change
or new runtime tests. The [representative calibration](representative-capacity.md)
records this iteration's code, native experiments and full QA results.
