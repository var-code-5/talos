# Proposal

## Why

The recorded 50-round two-host CIFAR experiment assigned 5,000 samples to each laptop per round, although the friend's laptop trained about 1.31 times faster and spent 241 seconds at the update barrier versus 13 seconds locally. Measured-work allocation can reduce this imbalance, but full replay auditing and replicated evaluation also limit achievable speedup.

## What Changes

- Add opt-in `--allocation compute-aware`, retaining equal allocation as the default and experimental control.
- Calibrate actual CIFAR throughput with the selected backend, numerical profile, and offered threads; do not use logical CPU count or the existing linear-kernel benchmark as training speed.
- Agree on deterministic, disjoint per-round sample assignments using smoothed measurements, with bounded adjustments and explicit plan identities.
- Preserve sample-weighted aggregation, per-peer optimizer state, independent aggregation, and full audit correctness for variable assignments.
- Account for audit and evaluation costs; report when these prevent useful balancing instead of promising zero waiting.
- Extend timing evidence and add an explicit allocation-policy comparison with quality and end-to-end runtime checks.

## Capabilities

### New Capabilities

- `peer-work-allocation`: Measured, deterministic training allocation and its correctness, timing, and comparison contract.

### Modified Capabilities

None; the main spec inventory is currently empty.

## Impact

Changes will affect `mtcp/cifar/{__main__,experiment,peer,training,report}.py`, backend payload handling and artifact identity, peer status UI, CIFAR tests, and `docs/cifar-peers.md`. Updated peers must negotiate the allocation protocol; old runs remain readable. No new dependency is planned. Asynchronous/stale updates, changing membership, GPU support, audit-policy changes, and distributed evaluation are out of scope.
