# Next task: public probe rate-limit identity

Assessment: 2026-10-03. This is a bounded local finding and proposal; the limiter
policy has not been changed in the early-authentication iteration.

## Observed outcome

`security.make_limiter` returns any nonempty X-Api-Key or Bearer value as the quota
identity. Public `/health` intentionally requires no key. An offline experiment
used one client address, a two-request/minute health limit and eight requests:

| Invalid credential headers | Observed statuses | Storage identities |
|---|---|---|
| Same value four times | 200, 200, 429, 429 | One |
| Four distinct values | 200, 200, 200, 200 | Four more |

Rotating an unverified header bypassed the shared public-probe quota and created
five stored identities in total. No body, model work or external traffic was used.
This demonstrates the selection problem; it is not a stress test or a quantified
memory-exhaustion claim. Early protected-route rejection does not authenticate
public health headers and therefore does not resolve this independent boundary.

## Recommendation and tradeoffs

[SlowAPI's official examples](https://github.com/laurentS/slowapi/blob/master/docs/examples.md),
discovered through web search/fetch, describe configured quota keys and endpoint
scope. Apply a stable, verified identity policy instead of accepting arbitrary
header strings as new public-probe callers.

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| Retain arbitrary header identities on health | Existing behavior | Reproduced quota bypass and extra keys | Replace |
| Key public probes by effective client address | Simple stable bucket despite rotating headers | Shared NATs; proxy trust needs explicit review | Preferred initial fix |
| Bound invalid credentials by client address, valid protected traffic by verified identity | Consistent identity contract, bounded invalid-header variation | More policy and tests; dev-mode behavior must be defined | Evaluate alongside the probe fix |

Use a small identity service shared with authentication, preserve ordinary quotas
and public probe access, and never store raw secret values in new evidence. Test
rotating X-Api-Key/Bearer values, valid-key precedence, dev mode, independent app
state, expiration and trusted/untrusted forwarding. Do not broaden proxy trust to
make a local test pass. Actual proxy buffering and target hardware remain later
deployment measurements.

Final next stack: stable public-probe quota identity, then operator proxy-buffer
capacity, hosted Windows execution and remaining accelerator/platform gates.
The [early-authentication outcome](early-api-key-authentication.md) and its evidence
archive separate the completed fix from this proposal.
