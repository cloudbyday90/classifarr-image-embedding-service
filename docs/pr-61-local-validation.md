# PR 61 local Pygments floor adoption

Assessment: 2026-10-16. Baseline: `8dad842ee51106aa3bea17eb98ae138b2959fe98`.

## Selection and official research

GitHub MCP returned six open PRs: #55, #56, #59, #60, #61 and #62.
The initial `secrets.choice([55,56,59,60,61,62])` draw selected
[PR #62](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/62).
Its fetched patch replaces Ubuntu 24.04 builders with 26.04 while leaving CUDA
and OpenVINO runtime bases on 24.04. The
[official Ubuntu package listing](https://packages.ubuntu.com/en/resolute/python3),
discovered through web MCP, identifies its default Python as 3.14. The repository
explicitly requires Linux CPython 3.12 and its current binary wheels target that
interpreter. Static contract review therefore rejects this patch as a standalone
base edit. No Ubuntu 26.04 build or compatibility success is claimed.

The second draw was restricted to the two fetched, compatible floor-only
proposals, `secrets.choice([56,61])`, and selected
[PR #61](https://github.com/cloudbyday90/classifarr-image-embedding-service/pull/61),
head `fb4e1f228e8ab3baa8b8cec8ad61aba5376797b6`. This is a suitability-filtered
selection, not an unbiased choice across all six. The other newer runtime/major
migrations remain separate work; they were not installed in this iteration.

The [official Pygments changelog](https://pygments.org/docs/changelog/), located
and fetched through web MCP, records 2.21.0 released August 17, 2026 and 2.22.0
unreleased. QA already contains the exact 2.21.0 publisher wheel, SHA-256
`2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9`.

## Decision and tradeoffs

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Align the development floor with locked 2.21.0 | Reproduces PR intent; prevents an older unconstrained developer install; no QA artifact churn | Requires renewing the QA input attestation | Adopt |
| Refresh unrelated wheels | Wider update | Extra compatibility scope without evidence of need | Defer |
| Apply Ubuntu 26.04 base substitutions alone | New supported base family | Interpreter/ABI mismatch and mixed builder/runtime versions | Plan separately |

`requirements-dev.txt` raises only the Pygments floor. Its QA manifest renews
only that input hash; the QA text lock and full artifacts are unchanged.
The two Windows text/JSON locks separately add the four dependencies needed for
the [matched-route native tests](post-upload-disconnects.md). Those additions are
part of the disconnect validation design, not effects of PR #61.

## Outcome

The QA graph retains all 70 artifact records and its exact wheel-lock text.
Only its development-input hash changes. The other eight Linux profiles retain
their artifacts and lock text, as do both Windows bootstrap profiles. The two
native Windows contract graphs preserve all 33 earlier artifacts and add exactly
the four reviewed middleware dependencies; resolver-selected bytes are hashed
before adoption. Both target interpreters resolve natively with reviewed pip
26.2.1 and constraints preserving existing versions.

All 13 input attestations pass. Native inventory/pip checks, strict complete OSV
audits, **1,203 passing Linux tests** and the two fresh Windows mandatory gates
pass; [the disconnect outcome](post-upload-disconnects.md) gives counts and
limits. No new Pygments wheel is installed by the floor change, so it is not
described as a newly applied runtime security patch. The
[validation archive](validation/disconnect-2026-10-16.json) separates this
floor-only adoption from the Windows test-graph additions.

The original PR is not merged. Delivery is directly on master with Unreleased
changes and no release, tag or version bump.
