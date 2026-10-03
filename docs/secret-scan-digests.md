# Historical validation digest classification

Assessment date: 2026-10-03. This is a narrow companion to [private secret setup](secret-setup.md), with its own design and native scanner outcome.

## Evidence and design

The final full-history scan detected two `generic-api-key` matches in baseline commit `d72eec64388bd4b5712a31fb1105378c895cc0da`, both in `docs/validation/ci-contracts-2026-10-03.json`. Their values are public SHA-256 digests for `secret-changes.log` and `secret-history-final.log`. Recomputing SHA-256 from those preserved native logs proves both values are artifact metadata, not credentials. The previous pre-commit history scan did not yet include that new record.

Retain Gitleaks 8.30.1's complete default rule set. Extend only `generic-api-key` with an AND condition combining the exact historical document path and either exact reviewed digest. Pass the repository config explicitly in CI. Do not ignore a whole commit, directory, file, rule or arbitrary hash-shaped string. Represent new validation hashes as separate `path` and `sha256` fields to avoid ambiguous credential-like dictionary keys. Do not rewrite published history.

The official [Gitleaks configuration documentation](https://github.com/gitleaks/gitleaks/blob/master/README.md), discovered through web MCP, describes extending default rules and rule-local AND allowlists. Native controls verify the actual pinned binary rather than relying only on documentation.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Leave verified artifact hashes unresolved | No policy change | Fails every full-history scan and obscures actionable findings | Reject |
| Exclude the document/commit or disable generic detection | Small configuration | Hides unrelated credentials | Reject |
| Exact rule, path and digest conjunction | Keeps historical provenance and other detection | Two reviewed exceptional public values remain policy metadata | Implement |
| Rewrite published history | Removes old pattern | Disrupts collaborators and remote history | Reject |

## Validation and outcome

Seven controls execute the verified native 8.30.1 binary against temporary Git histories. The two reviewed values at the exact path each return 0 with zero findings. The same value at another path, an unapproved value, and an extended approved value each return 2 with a generic finding. A synthetic GitHub-shaped token at the approved path returns 2 from the retained `github-pat` rule. Malformed configuration returns 1. All results are redacted; no real credential is used. The temporary orchestration helper is ESM and is not part of runtime code.

Full fetched history replay with the scoped configuration passes with zero findings across 80 commits. A committed policy regression guards default extension, rule locality, exact path/value matching, AND semantics and explicit CI configuration. The [setup validation archive](validation/secret-setup-2026-10-03.json) retains the reviewed public digests, native outcomes and evidence hashes. Current setup/CodeQL design remains in its own document; these reviewed false-positive exceptions do not suppress CodeQL's explicit-display or nonsecret-config alerts.
