# Native dependency graph review

The repository's `scripts/dependency_profiles.py` is the current profile authority.
`scripts/lock_dependencies.py --check` validates input attestations and complete
wheel records without resolution. The reviewed bootstrap pins pip; resolve with
that interpreter/pip on the actual profile OS, Python version and architecture.
Cross-target dry-run reports are not native validation.

## Floor update versus graph change

For a floor-only adoption, compare all artifact records with the baseline. Renew
only affected input hashes through `dependency_locks.input_contract`; unchanged
wheel URLs, versions and hashes establish that runtime libraries did not change.
The installer `--verify-only` still has to accept the exact installed native graph.

For an additive QA migration, build constraints from the current reviewed QA
artifact versions, then run the existing generator with `--backend qa --root
<isolated-source> --constraints <reviewed-constraints> --cache-dir <cache>`. Review
the JSON delta and emitted text lock together. Confirm publisher hashes for new
wheels, approved origins and absence of unplanned removals/version changes.

Install the complete graph with `scripts/install_dependencies.py` in a fresh
environment or immutable image derived from the previously verified graph. Do not
add packages to a shared old environment and call its inventory a new lock. Use
`--verify-only` after installation and audit that installed graph with the isolated
reviewed pip-audit tool graph, strict OSV mode and no advisory exclusions.

## Evidence boundaries

- API tests: explicit response classes, lifespan state/startup/shutdown, exception
  policy, cleanup and existing service admission/cancellation tests.
- Bundled probes and SDKs: retain their client classes and dependencies until a
  separate migration demonstrates their actual interfaces and transport defaults.
- Windows contract graphs: Linux universal-wheel compatibility alone does not
  prove a native Windows graph; consult the native-validation skill if affected.
- Production graphs: exact inventory plus unchanged records is useful evidence
  for a declared floor update, but does not establish new model performance.

Run checks proportional to the changed boundary. For a broad API test-client
migration, run full QA with unchanged coverage floors. Keep source snapshots
immutable while tests/scanners use them. Store evidence outside the checkout until
its summaries are ready; copy only reviewable records into a validation document.

Read and write project text explicitly as UTF-8 on Windows. Keep machine-specific
cache/interpreter paths out of committed workflow commands and avoid copying
ignored environments, credentials or model caches into source snapshots.
