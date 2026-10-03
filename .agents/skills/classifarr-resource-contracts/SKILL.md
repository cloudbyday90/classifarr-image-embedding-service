---
name: classifarr-resource-contracts
description: "Plan, implement and validate Classifarr image-embedding resource contracts: early API-key checks, quota identities, remote fetch and batch budgets, ingress, cancellation, admission, worker cleanup and cache preparation. Use for these service refactors or ownership regressions; not for unrelated copy edits, generic version lookups or release operations."
---

# Classifarr Resource Contracts

Keep service changes small and explicit about who owns capacity after the HTTP caller leaves. Start at the repository root containing `OPENAI.md`, `src/image_embedder` and `requirements/locks`; resolve repository paths from there, not from the skill's own directory.

## Establish the contract

Read `OPENAI.md` and the current `docs/recommendation-stack.md`, then follow only the affected rows in [the contract map](references/contracts-and-validation.md). Inspect the implementation and relevant existing tests before adopting a historical recommendation.

For authentication/body-order changes, read [early authentication contracts](references/early-authentication.md). Keep protection attached to matched routers and verify native header-only rejection, mandatory admin policy and response ownership.

For quota-key or public-probe changes, read [quota identity contracts](references/quota-identities.md). Trace every unverified header form into storage and preserve the server's forwarding trust boundary.

Record:

- The resource owner, admission point and release condition, including failure cleanup.
- Which clock starts the budget, what consumes it, and whether it limits a socket hop, fetch, batch preparation, HTTP wait or native execution.
- Ordered partial-result behavior, cached/inline input precedence and sanitized error behavior.
- What happens when the caller cancels or times out while the resource is still active.

Dispatched inference belongs to `InferenceExecutor`, not its waiter. Keep its permit and read lock until synchronous work returns. Kill/reap a supervised remote child before its owner returns. Closing a response or cancelling a Python task is not evidence that DNS, a native kernel or a child process has stopped. Never make a budget mutable global, shared embedder field or temporary Settings override; each invocation owns it.

## Choose and implement

Explain the recommended option and its concrete costs in a separate design/outcome MD for a new behavior. Discover official source URLs with search/fetch tools when researching current best practices; distinguish upstream guidance from project defaults. Preserve public API schemas and content-based cache identity. Prefer a small service module over adding another ownership mechanism to `embedder.py`.

Keep remote destination, TLS, redirect, byte/pixel, protocol and environment-isolation checks intact. Use Python for application code; authored JavaScript uses ES modules. Avoid introducing a new dependency unless its benefit justifies reviewing all affected complete locks. A dependency-floor change is not a graph refresh: first prove the reviewed wheels satisfy it, then renew input attestations and verify unchanged artifacts.

## Prove the affected boundaries

Choose checks from the reference map according to changed contracts. Use deterministic clocks for boundary policy and actual OS children/sockets for termination or transport claims. Cover success, failure, delayed completion, detached ownership and capacity recovery; for batches include cache hits, inline inputs and ordered partial failure. Keep fixtures offline and avoid model downloads in routine tests.

For native validation/scanning, copy Git-tracked files plus relevant nonignored new files to a temporary checkout-equivalent source tree. Exclude `.env`, ignored caches, wheels and private data. Mount source read-only and put logs/artifacts outside Git. Record the exact image/interpreter, source hashes, commands, outcomes and limitations; an in-process ASGI fake cannot prove native cleanup. Follow existing scanner policy rather than suppressing findings to make a gate pass.

After runtime changes, run full locked QA and the unchanged coverage ratchet. Add native backend/model/device checks when those contracts change; don't claim accelerator performance from a CPU fallback or rerun unrelated expensive checks without a reason. Validate the skill itself with the skill-creator validator when editing its metadata/references.

## Complete the record

Replace planned outcomes with observed results in each design MD. Update README/config examples for operator-visible behavior, Unreleased for high-level changes and the recommendation stack with the next unresolved item. Separate current evidence from historical results and state untested platforms honestly.

Use only the branch, commit/push, PR, release and external-message authority supplied by the current user/session. A request to adopt a PR locally does not merge its upstream PR. This skill creates no additional approval step and grants no authority for deployment, publishing, credentials or external writes.
