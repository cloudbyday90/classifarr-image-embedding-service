# Open PR availability

Assessment: 2026-10-03. GitHub MCP returned eight open PRs; their fetched diffs and
immutable heads were compared with the local master baseline
`ded0a63843c728abfaed29e8a8e6c8a05e80ebb8`.

## Design and decision

Only suitable unapplied work belongs in a new local PR adoption. The user clarified:
"if none exists, then just move on with a note thre is none to choose".
No random PR adoption is included in this iteration. No original PR was merged,
closed, commented on or otherwise changed.

| Open PR | Reviewed head | Reason excluded |
|---|---|---|
| [#44](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/44) | `e66ea84b748833448eb7dd8efbf91eb78f551239` | Ubuntu 25.10 reached end of life on July 9, 2026; retain the supported 24.04 base. |
| [#42](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/42) | `c174b4d4ce6c8a54e3d5583998aa0fa08414c3ee` | Checkout v7 is already adopted with the reviewed v7.0.1 immutable reference. |
| [#41](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/41) | `66ba03671b3deea9e808de88ff6ce7fb038c4681` | The action was adopted earlier and subsequently replaced by verified native Gitleaks scanning. |
| [#40](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/40) | `a28cf48f141fa165599d0b9d93341c212664af48` | The proposed action has been replaced by strict native installed-environment audits. |
| [#35](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/35) | `e01c22ffad5997bb99a4bc6eecfb944068d6ee31` | The ordered range includes a forbidden local-version label and would force CPU selection into shared requirements; current generic floor and exact backend locks supersede it. |
| [#34](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/34) | `b4d4ccc31d1a24bdfc3a628ba56b91ba2e2546b2` | The HTTPX 0.28.1 SDK/probe floor is already implemented. |
| [#33](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/33) | `5054ede6625ba6fc860c3a563644f8402a5f5769` | The pytest-cov 7.1.0 floor is already implemented. |
| [#27](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/27) | `e640f0807906bd12705b311b101c80b753ee41e0` | Trivy action v0.36.0 is already implemented with an immutable reference. |

[Ubuntu's official EOL announcement](https://lists.ubuntu.com/archives/ubuntu-announce/2026-July/000325.html)
confirms the July 9 cutoff. The
[Python version-specifier specification](https://packaging.python.org/en/latest/specifications/version-specifiers/)
excludes local-version labels from ordered comparisons such as `>=`.
These URLs were discovered and fetched through web services rather than constructed.

## Outcome and recommendation

There is no suitable unapplied open PR to choose. Continuing the recommended
early-authentication work avoids replaying already delivered changes or weakening the
platform. Recheck the current open list in the next iteration; PR state and heads
can change. The [authentication design and outcome](early-api-key-authentication.md) records
this iteration's actual implementation and validation. The eight heads were
refetched through GitHub MCP and remain unchanged from the socket iteration.
