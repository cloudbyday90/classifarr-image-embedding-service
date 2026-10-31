---
name: classifarr-storage-maintenance
description: Inventory and clean stale Classifarr image-embedding validation files, containers, images, volumes and build cache while preserving models, active work and other projects. Use for disk cleanup and at the end of storage-heavy validation; not remote registry retention or host-wide Docker maintenance.
---

# Classifarr storage maintenance

Keep maintenance part of the work: inventory before large builds, settle owned
containers after validation, and review retained artifacts when an iteration
finishes. Review disk use weekly while this project is actively developed;
this is a maintenance convention, not authority to install a scheduled job.

Read `OPENAI.md` and [the storage policy](../../../docs/storage-maintenance.md).
Use the current repository/daemon inventory, not historical IDs as delete lists.
An unused resource is only a candidate: detached model caches, rollback images
and another project's stopped containers can still be valuable.

## Establish ownership and retention

- Save a small inventory and explicit candidate/keep list outside Git. Record
  current Docker context, all container references (including stopped ones),
  exact image IDs/tags, mounts, builder/cache records and resolved host paths.
- Prefer the exact project label
  `org.classifarr.image-embedding.project=classifarr-image-embedding-service`.
  Legacy resources need corroborating entrypoint/module, repository bind mount,
  archived run identity or project-specific build description. A `classifarr-`
  prefix alone also matches the separate Classifarr application.
- Keep `.env`, configuration, `.git`, tracked source/tests/docs/lockfiles, user
  edits, the working development environment, both persistent model/IR caches,
  all active resources and explicitly retained rollback/current validation images.
  Retain one validated dependency image per required backend/architecture and
  needed scanner tools; do not assume an old dependency image contains current
  application source. Preserve concise committed validation evidence.
- Completed source snapshots, analysis databases, temporary environments and
  obsolete validation images can be removed after evidence is archived and
  ownership/non-use is verified. Seven days is the normal scratch review age,
  not automatic deletion authority or proof that a resource is stale.

## Clean within the authorized scope

Reinspect immediately before each mutation. Skip changed identities, new users,
running containers, mounts in use and ownership ambiguities. Report what was
left and why; do not widen a selector to obtain a larger reclaimed number.

Remove exact stopped container IDs without `--force` or volume deletion. Remove
explicit obsolete image tags/IDs without force and with `--no-prune`; protect
any image with a retained or unrelated tag. Remove only individually verified
scratch volumes, never blanket unused volumes. Do not run host-wide
`docker system prune`, unfiltered image/volume prune, or `compose down -v`.

The default builder is shared with other projects. Select known project cache
record IDs, or a genuinely dedicated project builder, and retain its bounded
working cache. Image/container labels do not identify BuildKit records. Never
apply age/size pruning to the shared builder as a substitute for ownership.

For Windows directory deletion, use one PowerShell workflow with `-LiteralPath`.
Resolve and verify each absolute target stays strictly inside the named checkout
or approved task-temp directory. Reject roots, symlinks/junctions/reparse points,
tracked files and active process paths. Extended paths may be needed for old
CodeQL scratch trees. Never delete the Docker VHDX, reset Docker Desktop or stop
other applications to make a project cleanup appear complete.

## Close the loop

Check retained image identities, model-cache presence and other containers'
state/health afterward. Report host-file bytes, Docker filesystem free-space
change and Docker logical usage separately: shared image/cache layers overlap,
and free VM space is not proof that the Windows VHDX shrank.

Record outcomes and intentional keeps in a concise design/outcome document;
avoid another full cache dump or tool archive in Git. Restate the previously
recommended development step when cleanup is the requested interlude. This
skill adds no authority to delete unrelated files, publish, release or change
production. Follow the user's current delivery instructions.
