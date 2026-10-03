# Recurring native Windows boundary validation

Assessment: 2026-10-03. Baseline: `3b9573ba9ef156fa45c898e611ecc04a4d64bf09`. Implementation and local validation stay on master.

## Problem and design

Linux QA cannot establish Windows DACL publication, reparse-point refusal, native child termination or Proactor socket cleanup. Earlier one-off Windows runs also used an older framework graph. Add recurring Windows Server 2025 x64 validation on conventional CPython 3.13 and 3.14, with weekly, manual and relevant push/PR triggers. Production remains Linux CPython 3.12.

Reuse the binary-wheel lock resolver/installer, adding explicit OS/interpreter profile identities instead of bypassing its target checks. The small test graph includes the current FastAPI/Starlette/Uvicorn transport and image-input dependencies. It excludes inference runtimes; the existing native fixtures intentionally never load production models. Each target has a complete graph, reviewed bootstrap, SHA-256 artifacts, input attestation, isolated/no-index installation, pip check and exact inventory verification.

A dedicated runner fixes the selected setup, remote supervisor/network/batch and response transport suites. It deselects unsupported uvloop, permits only explicit POSIX and unavailable symlink setup skips, and fails missing critical cases or any other skip. In particular, junction/DACL checks must execute. Preserve actual child/socket fixtures and production private-destination refusals. Publish only bounded outcome metadata; console/temporary secret files and generated TLS keys are not uploaded.

A separate Linux job audits both attested Windows graphs through the isolated reviewed auditor. It exports every exact version/hash, disables cross-platform resolution and requires a complete, unskipped, vulnerability-free JSON inventory. OSV auditing checks advisory metadata; artifact hashes are verified by the native installer, not by OSV.

Use full action commits, contents-read only, ephemeral hosted runners, no checkout credentials, dependency cache or secrets. Python minor versions are explicit, latest stable patches are selected, and prerelease/free-threaded interpreters are excluded. Hosted orchestration is a distinct follow-up observation after push; local Windows 11 is not Windows Server 2025 evidence.

## Official guidance and tradeoffs

Sources were discovered through web search and GitHub MCP on 2026-10-03.

- [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) lists Windows 2025 x64 and fresh VMs. Explicit labels avoid silently switching OS families, but the image still receives updates.
- [GitHub secure use](https://docs.github.com/en/actions/reference/security/secure-use?ref=blog.gitguardian.com) recommends minimum token permissions and full commit pins. These reduce authority and mutable action drift; hosted integration still needs observation.
- [setup-python guidance](https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md) explains minor selection and check-latest. Latest patch selection improves patch uptake but must be recorded because interpreter bytes are not frozen by wheel locks.
- [Python Windows asyncio support](https://docs.python.org/3.14/library/asyncio-platforms.html) distinguishes Proactor subprocess support and unsupported Unix facilities. Native tests are needed; Linux emulation would be cheaper but cannot prove these boundaries.
- [Python subprocess](https://docs.python.org/3/library/subprocess.html) distinguishes timeout, kill and subsequent cleanup, and notes process creation limits. Assertions establish reaping and retained ownership, not a hard real-time guarantee.
- [Microsoft protected DACLs](https://learn.microsoft.com/en-us/windows/win32/secauthz/security-descriptor-control) explains inheritance protection. Validate the actual creation/rotation ACL rather than relying on POSIX mode bits.
- [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/?highlight=no-deps) supports complete hash checking and binary-only installation. Target locks cost review effort but avoid executing source builds and accepting an unreviewed graph.
- [PyPA pip-audit](https://github.com/pypa/pip-audit/blob/main/README.md?plain=1) documents auditing complete pinned inventories without pip resolution. Audit the reviewed Windows versions directly rather than resolving Linux markers in their place.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Continue occasional host tests | Minimal CI cost | Drift and missed regressions | Replace |
| Full Windows inference environment | Broader runtime scope | Heavy unsupported backend graph and downloads | Defer until Windows inference is supported |
| Small native contract graph plus mandatory outcomes | Repeatable OS evidence without model downloads | Separate locks and explicit scope | Adopt |

Final stack: native hosted Windows matrix; complete target locks; real ACL/process/socket tests; mandatory-case and skip gate; metadata-only artifacts; existing Linux production/model/device gates.

## Outcome

Two fresh native Windows 11 x64 venvs pass **102 cases each**, with **16 permitted POSIX/symlink-privilege skips** and **18 uvloop deselections**. Actual interpreters are CPython 3.13.3 and 3.14.2, both using `asyncio.windows_events.ProactorEventLoop`; these local patches are older than the stable patches the hosted workflow will select. Each has the current 33-wheel contract graph, including FastAPI 0.142.2, Starlette 1.7.0 and Uvicorn 0.54.0. Owner-only DACL creation/rotation, junction refusal, ACL/handle failure controls, blocked DNS kill/reap, TLS/trickle fetches, 32-item shared budgets and both response parsers execute successfully. Native gate/target/audit policy checks pass **19 additional cases** on Python 3.14.

The first native attempt exposed an omitted import dependency, FileLock; it is now explicitly pinned. The next run passed all executable cases but correctly failed the gate's incorrect symlink parameter IDs. The allowlist now enumerates the eight existing symlink cases exactly; critical cases remain mandatory. Controlled servers can report connection resets after intentionally killed clients; the cleanup assertions still pass. No production timeout, resource or destination policy is relaxed.

Final locked Linux QA passes **947 tests**, seven existing platform/optional skips and one existing Starlette TestClient HTTPX2 deprecation warning. Coverage is **95.22% lines / 90.12% branches**, above unchanged **89.42% / 79.37%** floors. Strict complete OSV reports cover 33 packages per Windows graph with zero known vulnerabilities or skips. The audit tool environment itself is the existing verified 29-wheel Linux graph. All thirteen lock attestations pass; all nine earlier Linux profiles retain their 446 artifact identities/hashes and lock text.

Ruff, scoped Pyright, workflow lint, copyright, local Markdown links and skill metadata checks pass. Native CodeQL security-extended analysis reports zero Actions alerts and retains only two contextual Python setup alerts (explicit `--show-key` output and nonsecret `0644` defaults), with no suppressions. Verified Gitleaks 8.30.1 reports zero leaks in checkout-equivalent source and the existing 84-commit history. Source identities and exact native reports are preserved in the [validation archive](validation/windows-native-2026-10-03.json). Hosted Server 2025 setup, latest interpreter patches and artifact upload remain unverified locally; model/device behavior is outside this feature's scope. No branch, original PR merge, release, tag or version bump is created.
