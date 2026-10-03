---
name: classifarr-dependency-migrations
description: Review and implement Classifarr dependency migrations with native wheel graphs, explicit client ownership and affected contract evidence. Use for dependency PR adoption or compatibility changes; not standalone version lookups, model capacity experiments or release operations.
---

# Classifarr Dependency Migrations

Identify the dependency's owner before changing its graph. The API test client,
Hugging Face SDK client, bundled capacity probes, platform contract suites and
advisory tooling can use separate environments. A compatible API does not make
objects from two client libraries interchangeable.

1. Inspect the current declared floor, installed lock and affected import paths.
   Distinguish an unapplied floor update from a package version already locked.
   Inspect current PR metadata/diff through GitHub tools when adopting a PR;
   reproduce its intent locally without merging the original unless requested.
2. Discover official publisher/framework sources for the target date. Treat API,
   typing, TLS trust, environment defaults and lifecycle behavior as separate
   compatibility questions. Record verified URLs and exact artifact identities.
3. For native lock work, read [references/native-graphs.md](references/native-graphs.md).
   Constrain existing reviewed distributions for an additive migration. Review
   every changed artifact before installation; do not refresh unrelated wheels
   merely because a resolver offers newer versions.
4. Prove the affected behavior in a complete isolated target environment. Use a
   meaningful negative control when a new gate should reject a missing dependency
   or fallback. Preserve deliberate request-only and lifespan tests. In-process
   ASGI evidence cannot establish socket/TLS behavior or worker interruption.
5. Record design, alternatives, observed outcome and limits in relevant project
   documents. Update Unreleased at the behavior level. Use the user's current
   branch/commit/push scope; this skill adds no authority to publish or release.

Use existing resource-contract, native-validation or capacity-calibration skills
only when the change affects those boundaries. Keep a test-client-only migration
out of production model, GPU and operating-system claims. Do not hide advisory
failures, loosen coverage floors or suppress new deprecations to make a graph pass.
