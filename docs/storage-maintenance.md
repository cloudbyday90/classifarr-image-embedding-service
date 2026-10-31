# Local storage maintenance

Assessment: 2026-10-31. This policy covers the Classifarr image embedding
repository and its local validation resources. The separate Classifarr
application, Harmoniarr, Duskcue, unknown Docker objects and remote registries
are outside this cleanup.

## Research and design

Official guidance was located and fetched through web MCP on October 31, 2026.
[Docker pruning guidance](https://docs.docker.com/engine/manage-resources/pruning)
distinguishes dangling images, stopped containers and unused volumes. An unused
volume can still contain valuable persistent data. This project's model caches
are kept even when no container currently mounts them.

[Docker disk accounting](https://docs.docker.com/reference/cli/docker/system/df/)
distinguishes shared and unique image sizes. Summed image sizes and build-cache
totals are not independent reclaimed-byte measurements.
[Buildx disk usage](https://docs.docker.com/reference/cli/docker/buildx/du/)
also distinguishes active records and shared image/cache storage. This daemon's
default builder serves several repositories, so a global age/size prune would
exceed the project scope.

[Buildx prune filters](https://docs.docker.com/reference/cli/docker/buildx/prune/)
support selecting individual record IDs and descriptions. Use positively owned
records here. For a future dedicated embedding builder, adopt a bounded cache
policy using [BuildKit garbage collection](https://docs.docker.com/build/cache/garbage-collection/).
A starting convention is a 20 GB cache budget, with a seven-day scratch review;
these are project retention choices, not measured capacity requirements. Do not
change the shared Docker daemon's settings on behalf of this one repository.

[Docker context guidance](https://docs.docker.com/build/concepts/context/) and
[cache optimization](https://docs.docker.com/build/cache/optimize/) recommend
excluding irrelevant files before transfer. The former explains `.dockerignore`
and recursive patterns. Our context now excludes `.env` files, nested bytecode,
alternate environments, analysis/editor directories and generated output.
Application source, scripts, inputs and complete wheel locks remain included.
No inference behavior or dependency artifact changes are needed.

[Docker labels](https://docs.docker.com/engine/manage-resources/labels/) are
metadata attached to objects. New runtime images identify this repository using
`org.classifarr.image-embedding.project=classifarr-image-embedding-service`.
Labels do not retroactively identify old resources or select BuildKit cache
records. Scratch containers/volumes should additionally record a role and unique
run ID; persistent model volumes should be explicitly retained.

## Retention and tradeoffs

| Resource | Retention | Benefit and cost |
|---|---|---|
| Tracked source, locks and concise validation evidence | Keep in Git | Reproducibility without tool/database dumps |
| `.env`, config, user edits and active `.venv` | Preserve | Maintains working setup; host venv is not the locked Linux QA environment |
| Hugging Face weights and OpenVINO IR caches | Keep both named model volumes | Avoids model downloads/export; intentionally retains several GB |
| Current dependency validation environments | One per required backend/architecture, plus needed scanner tools | Fast follow-up validation; still consumes storage and must match current locks |
| Superseded validation images | Remove explicit unused project tags after evidence is archived | Releases obsolete layers; shared layers may remain |
| Completed stopped test containers | Remove after evidence capture, without `-v` | Releases stale container ownership without touching model volumes |
| Completed source snapshots, CodeQL databases and wheel scratch | Remove once archived and unused; review scratch after seven days | Prevents accumulation; reconstruction/download costs return next time |
| Build cache on the shared builder | Exact proven project records only | Preserves other projects; leaves unclassified host-wide growth for a separate task |

Keep an explicitly needed rollback image while a deployment change is being
evaluated. Age alone is never sufficient evidence of ownership or abandonment.
Before removing an image, inspect all tags and all container references, including
stopped containers. Before removing a volume, inspect every mount reference and
its contents/type. Recheck immediately before mutation; avoid forced deletion.

For Windows files, resolve exact absolute paths and reject roots, reparse points,
tracked files and active process paths. Use `Remove-Item -LiteralPath` within one
PowerShell workflow. Old CodeQL trees can require extended paths. Preserve
unknown files instead of expanding a prefix/glob to make deletion succeed.

The [maintenance skill](../.agents/skills/classifarr-storage-maintenance/SKILL.md)
is discoverable in the repository and is referenced by `OPENAI.md`. Apply it
before storage-heavy builds and at iteration close; review disk use weekly during
active development. This creates no scheduled destructive job and no authority
over other repositories, registry retention or Docker Desktop resets.

## Cleanup outcome

Removed **15 stopped embedding-test containers**, **31 obsolete image tags
representing 30 distinct listed image IDs**, **eight unused scratch volumes**,
**11 positively identified BuildKit records** and **one empty project network**.
The scratch volumes held about 6.2 GB of allocated files. BuildKit reported only
1.143 MB reclaimed from the selected records: ten shared records released cache
metadata without freeing image layers. Do not count their logical sizes again.

Retained all six latest validated dependency images (QA, CPU amd64/arm64,
OpenVINO, CUDA and legacy CUDA), the small scanner image, both persistent
model/IR volumes, the primary `.venv`, configuration, Git and user edits.
The separate Classifarr, Harmoniarr and Duskcue containers retain their original
IDs, images, states and health classifications. The two running applications
are healthy. Their reported uptime changed during this shared-host session;
this cleanup issued no stop/restart command, and does not claim uninterrupted
service or exclusive attribution of whole-daemon disk changes.

Automatic approval review rejected both the bulk Windows directory removal and
the individually bounded `.venv-1` removal with **"blocked by policy"** and no
more specific reason. No host directories/files were deleted by those commands.
The unused `.venv-1` environment (831,367,964 logical bytes), old validation temp
remnants and generated checkout caches remain pending. They were not deleted
through a different tool or shell. The new ignore rules prevent their inclusion
in future Docker contexts even while they remain on disk.

The [compact validation record](validation/storage-maintenance-2026-10-31.json)
records successful removals, intentional keeps and measured Docker storage
changes. Docker VM free space is separate from Windows drive/VHDX allocation.
No disk-image reset, WSL shutdown, host-wide prune or remote registry deletion
was performed; unknown/shared resources remain outside this task.

| Observed daemon/VM metric | Before | After |
|---|---|---|
| Docker image accounting | 367.1 GB | 182.5 GB |
| Build-cache accounting | 316.6 GB | 312.4 GB |
| Volume accounting | 27.96 GB | 22.12 GB |
| Available Docker VM filesystem space | 339.9 GB | 530.0 GB |

The observed free-space increase is **190.1 GB** (decimal), measured with
`statvfs` on the same Docker filesystem. Image/cache figures overlap and must
not be added. This is a shared-host before/after observation, not an exclusive
per-deletion attribution or evidence that the Windows VHDX allocation shrank.

## Validation and skill evaluation

A real scratch-stage BuildKit export of the revised context contains 149 files,
752,833 bytes. It excludes the existing `.env`, both environments, nested
bytecode, editor/analysis directories, tests and generated coverage, while
preserving runtime source, scripts and complete lock inputs. An initial export
found 94 nested bytecode files; recursive ignore patterns corrected that before
the successful final check. This validates actual context behavior rather than
only matching strings in the ignore file.

All three real Dockerfiles pass native BuildKit `--check`. All thirteen lock
attestations pass. The six retained dependency images each pass exact native
inventory verification and pip check after cleanup; ARM64 uses QEMU, with no new
physical ARM/model-performance claim. Runtime Python and dependency artifacts
are unchanged, so production inference benchmarks and full API QA were not rerun
for this maintenance-only change.

The skill metadata validator, copyright and whitespace checks pass. Manual
scenario review confirms that unused model volumes, another application's
stopped container, an image with an unrelated tag, a Windows reparse/outside-root
path and unclassified shared-builder records must be preserved. The actual
cleanup checks the corresponding image/mount/container boundaries. This is a
manual skill review plus observed cleanup behavior, not an independent agent
benchmark or proof of an unattended scheduler. No destructive schedule was added.

## Resume development

After the maintenance push, GitHub reported **54 dependency alerts** on the
default branch: 6 critical, 24 high, 22 moderate and 2 low. This is an untriaged
[GitHub notification](https://github.com/cloudbyday90/classifarr-image-embedding-service/security/dependabot),
not a finding that all alerts affect the installed runtime. Exact inventory and
`pip check` do not replace advisory audits. Before the next deployment-host run,
refresh the existing strict audits and triage this notice against the actual
locked environments. No dependency upgrades or advisory fixes are claimed here.

The previous next step remains **deployment-host calibration of native chunk
size and caller/batch pressure**. Use portrait posters plus landscape and
PNG/JPEG/WebP cases, the verified models, isolated clients, and early disconnect
recovery. Keep one inference owner, current limits and chunk size eight until
the target-host evidence supports a change. See the
[recommendation stack](recommendation-stack.md) and
[capacity skill](../.agents/skills/classifarr-capacity-calibration/SKILL.md).
