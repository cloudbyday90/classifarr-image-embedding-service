# Release cleanup credential and retention design

Assessment date: 2026-10-03. Cleanup remains a future version-tag-only job dependent on successful release publication; this iteration creates no tag or release and performs no external deletions.

## Design and official research

The previous inline shell inserted credentials directly into generated code, exported a derived bearer token through job outputs and ignored HTTP failures. It also considered only the first 100 tags. Replace it with separate Python HTTP, inventory/retention and entrypoint modules.

The discovered [Docker authentication contract](https://docs.docker.com/reference/api/hub/latest/operations/AuthCreateAccessToken/) uses `/v2/auth/token` with identifier/secret JSON and returns a short-lived access token. The previous users/login route is deprecated. Use a scoped personal access token rather than an account password, with only the repository access and deletion scope needed by cleanup.

The [current tag-list API](https://docs.docker.com/reference/api/hub/latest/operations/ListRepositoryTags/) supplies paginated counts/results. The official [Docker hub-tool tag client](https://github.com/docker/hub-tool/blob/main/pkg/hub/tags.go), fetched through GitHub MCP, supplies the existing legacy tag-deletion route. Do not infer a new deletion route from the newer listing route.

Credentials arrive through step environment variables and are JSON data, never shell fragments or CLI arguments. Keep the returned bearer in memory; never print it, publish a job output or include server bodies in errors. Refuse redirects and ambient proxies. Require bounded object responses and expected successful HTTP statuses.

Validate every pagination authority, repository path and numeric page parameter before sending authentication. Bound inventory to 100 pages/10,000 tags, require stable counts and complete collection, and reject duplicate/invalid names or missing/timezone-free timestamps before deleting anything.

Keep the five most recently updated non-latest tags, always preserve `latest`, and explicitly protect the tag just published even if timestamps are skewed. Sort deterministically and delete oldest first. Unknown or absent current tags fail closed. Stop on the first failed deletion; prior successful deletions cannot be rolled back.

## Alternatives and recommendation

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Repair inline curl/jq | Less Python code | Shell quoting, pagination and token-output contracts remain harder to verify | Reject |
| Modular fixed-origin Python client | Keeps credentials out of generated code; pure retention policy; bounded reads | Own API/schema maintenance and partial-delete behavior | Adopt |
| Disable all retention | Removes cleanup credentials and deletion risk | Registry storage grows; changes the existing release contract | Revisit if automatic retention is unnecessary |

GHCR cleanup keeps its existing five-version policy through a verified immutable action. Its package-write permission is confined to the tag-only cleanup job. Add contents-read solely for checking out the cleanup implementation, with persisted Git credentials disabled.

## Local outcome

Fixture tests cover safe JSON serialization of quotes/newlines/shell-like credential strings, invalid bearer rejection, redirect refusal, sanitized HTTP errors, multi-page collection, count changes, incomplete/looping/off-origin pagination, malformed tag records, duplicate names, current-tag protection and deterministic retention.

A public read-only request to the documented listing endpoint successfully validates this repository's complete six-tag inventory; no secret, authentication or deletion is used for that check. Authentication/deletion use fixtures locally. Actual credential scopes, deletion response behavior, permissions and provider integration remain future release-job gates.
