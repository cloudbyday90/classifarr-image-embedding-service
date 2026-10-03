# Quota identity contracts

Use this reference for quota selectors, public-probe limits or upgrades affecting
identity/storage behavior. Read `rate_limits.py`, `credentials.py`, `security.py`
and affected decorators before choosing a boundary.

Public `/health` and `/ready` always use the effective client address, including
valid credential headers. Protected embedding uses one nonsecret principal only
after the configured shared key matches. Missing, invalid, empty and unconfigured
credentials fall back to the address, including dev mode. Every identity is
nonempty and namespaced; raw keys must not enter storage or limiter logs.

Keep X-Api-Key precedence, empty-primary fallback, case-insensitive Bearer scheme,
exact suffix comparison and constant-time UTF-8 content comparison shared with
authentication. An empty Bearer suffix historically produced an empty key that
SlowAPI skipped; hashing unverified input would retain the rotation bypass.
Do not add request-state markers, a path allowlist or per-header buckets.

The server owns forwarding trust. Use `request.client` through the existing
address helper; do not parse X-Forwarded-For, Forwarded or X-Real-IP in quota code.
Test actual pinned Uvicorn with trusted/untrusted peers and repeated forwarding
headers. ASGI scopes alone do not prove native server trust. Different ports must
not produce new callers. NATs share probe quota; wildcard proxy trust requires
separate operator evidence.

Preserve separate endpoint counters and current quota settings, public probe
access, dev behavior, mandatory admin protection and zero-body auth refusal.
Verify fixed/rotating/empty credentials, valid header forms across addresses,
mounted prefixes, independent app state, expiry/recovery and secret-free storage.
Memory storage expires identities; stable keys are not a global address ceiling
or proof against distributed callers. Do not replace storage or tune capacity
without deployment evidence.

Focused checks:

```sh
python -m pytest tests/test_rate_limits.py tests/test_rate_limit_sockets.py tests/test_auth.py tests/test_auth_early.py tests/test_auth_sockets.py tests/test_ingress.py tests/test_integration.py -o addopts=''
```

Use the locked QA environment; local venvs may have older framework versions.
[Design and outcomes](../../../../docs/public-probe-rate-limits.md) separate current
evidence from proxy/hardware follow-up work.
