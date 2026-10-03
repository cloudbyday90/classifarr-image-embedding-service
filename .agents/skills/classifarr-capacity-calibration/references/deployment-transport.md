# Deployment transport evidence

The current operator uses direct Docker port publishing, without a reverse proxy.
Do not add a proxy merely to satisfy a capacity experiment. Reconfirm topology
when the user changes it.

The shipped `image_embedder.server` launcher and socket-capacity server explicitly
disable forwarding for an empty `[server].forwarded_allow_ips`. The environment
override is `IMAGE_EMBEDDER_FORWARDED_ALLOW_IPS`; an empty override disables trust.
`FORWARDED_ALLOW_IPS` is deliberately ignored by these entrypoints. An alternate
ASGI launcher needs its own policy. IP networks must contain only controlled
proxy peers; publishing a Docker port does not authorize HTTP forwarding headers.

Reuse the native parser fixtures in `tests/test_forwarding_sockets.py` and the
existing `capacity_probe.py --scenario socket`. For a Docker port contract, start
a fresh bounded non-root locked image with the current source overlay, a temporary
configuration, a disposable API key, warmup disabled and a low public probe quota.
Publish only to host loopback on an ephemeral port. From outside that container,
send repeated/rotated `X-Forwarded-For` and `X-Forwarded-Proto` fields: direct mode
must exhaust one address bucket despite ambient `FORWARDED_ALLOW_IPS=*`. Verify
authentication and graceful shutdown; retain status evidence without keys.
Always remove the temporary container on failure and verify it was removed.

Record image/source identity, Docker version, bind address and client placement.
A Docker Desktop host request can appear to the service as a gateway/NAT peer.
That result does not establish how a remote LAN client is represented on Linux
or in the operator's production network. Do not label an injected forwarding
header as an observed external address.

For headroom, retain backend-specific cgroup/RSS/task/tmpfs journals from the
existing offline socket scenario. Its client and sampler share the service's
cgroup; keep that overhead in the record. A published-port quota smoke with fake
models establishes transport behavior, not model throughput or memory headroom.

If a proxy is actually adopted, measure its request buffering separately from
response buffering, finite body/temp-storage budgets, trusted peers and timeout
alignment. Streaming request bodies does not itself guarantee a total proxy
upload deadline. Read the proxy's current official guidance and test the actual
configuration; a reference topology is not production evidence.

Evaluate: direct Docker/no proxy (keep direct topology), wildcard ambient trust
(prove disabled default), explicit untrusted peer (headers ignored), controlled
trusted peer (headers honored), Docker/NAT collapsed clients (shared quota), and
dependency-only change (do not trigger this workflow).

Design and outcome: [direct deployment trust](../../../../docs/direct-deployment-trust.md).
