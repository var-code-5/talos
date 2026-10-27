# Design

## Context

See proposal.md for observed imbalance. `training.indices` currently fixes a strided shard per peer, `Peer.execute` divides the round budget equally, and `validate_config` computes round counts from that assumption. Simply changing `count` would exhaust shards unevenly. Full audits reconstruct remote payloads from the local payload, which also assumes equal offsets. `report.compare` currently assumes identical optimization settings and has no allocation-policy comparison.

## Goals / Non-Goals

**Goals:** Reproducible plans from shared measurements, complete sample coverage, auditable unequal contributions, and measured reductions in end-to-end time when the workload permits them.

**Non-Goals:** Eliminating all barriers, promising a speedup under full replay, changing audit coverage, assigning work from nominal hardware specifications, changing peer membership, or migrating optimizer momentum between peers.

## Decisions

### 1. Versioned opt-in policy and comparable controls

Add `--allocation equal|compute-aware`, default equal. Preserve legacy equal behavior and its old indexing version when reading or replaying existing runs. Introduce a versioned global-window indexing mode for compute-aware runs and for new equal controls used in explicit allocation comparisons. Record policy, indexing version, planner constants, and offered resource settings in config. Capability negotiation rejects unsupported modes before starting any run. Keep the current equal thread-budget constraint in this change; measuring differing CPU performance under those budgets already addresses the observed problem.

### 2. Calibrate the actual kernel

Use an authenticated, bounded calibration operation on idle peers: one discarded warm-up and three measured executions of 512 samples from the selected dataset using exact initial weights, batch size, numerical profile, backend, and offered threads. Restore initial weights and empty momentum between repetitions. Record preparation/execution overhead separately from kernel time, take median microseconds per sample, and measure one evaluation at the configured test size. For full auditing initialize replay cost from the same kernel, then refine it using actual replay measurements. Zero, negative, nonfinite, timed-out or mismatched results abort compute-aware admission; never silently invent rates. Calibration is isolated from run checkpoints and reported separately and in total startup-to-finish cost. No public endpoint triggers expensive calibration.

The current advertised fixed linear-kernel throughput and CPU count are unsuitable substitutes for CIFAR throughput. Using recorded historical rates alone would miss thermal throttling and background load.

### 3. Global sample windows with deterministic assignments

Keep the existing seeded selection of `train_limit` examples. Version 2 shuffles that selected set once per epoch using a specified seed derivation shared by all peers. Divide the resulting permutation into global windows of at most `sync_samples`. A plan assigns contiguous, nonoverlapping ranges within each window in sorted peer-ID order. Counts sum exactly to the window size. This allows work to move without losing or duplicating samples. Normal counts are batch multiples, with a deterministic final remainder; tiny windows can give zero samples to some members. Round count becomes `epochs * ceil(train_limit / sync_samples)` for version 2.

Plan metadata includes run/config digest, epoch, round, global offset, member ranges, previous model digest, and measurement-set digest. Hash it canonically. Payload validation reconstructs the expected range from the agreed plan, rather than accepting arbitrary caller-supplied indices. Full audit replay uses the target peer's range and preserved optimizer state. Zero-work peers preserve momentum and participate in synchronization with weight zero. Aggregation remains weighted by count; unequal local SGD steps change the optimization trajectory and must be reported.

### 4. Optimize full phase cost, with bounded adaptation

For no auditing and negligible fixed costs, initial counts are proportional to measured samples/second: `n_i = B * r_i / sum(r)`. The observed 1.31 speed ratio suggests roughly 5,675 samples for the friend and 4,325 locally before batch rounding; this is an illustration, not a fixed configuration.

Use a deterministic integer-cost planner for the general case. For candidate counts predict `T_i = setup_i + n_i * train_cost_i`, `A_i = replay_setup_i + (B - n_i) * audit_cost_i` under full replay (zero for no audit), and evaluation `E_i`. Minimize `max(T_i) + max(A_i + E_i)` for the synchronized phase model. Include measured plan/transfer overhead in reported predictions, without inferring bandwidth from barrier time. A deterministic batch-transfer search starts from equal, rate-proportional, and previous counts, moves one batch between pairs while improving the objective, and chooses the best candidate with stable peer-ID tie-breaking. Stop after 10,000 evaluated candidates; all peers use identical ordering, integer arithmetic, and limits. Do not claim global optimality.

For regular windows give each member at least one batch when feasible; remainder windows use deterministic largest-remainder rounding. Bound each peer's share change to 10 percentage points per round (initial calibration allocation is exempt). Update cost estimates using EWMA alpha 0.2 over completed rounds; clamp each new per-sample observation to 0.5–2 times its previous estimate and retain valid previous rates for zero-work peers. Persist raw and filtered values. Keep the previous plan if predicted improvement is below 2%; initialize from equal if no candidate clears that threshold. Rescale shares deterministically for remainder windows.

With full replay, every peer computes approximately the entire global window as training plus auditing. Allocation may improve phase alignment but cannot remove that duplicated work or replicated evaluation. Mark `audit_limited` when the model predicts less than 2% improvement over equal; retaining equal work in that case is a valid outcome. Switching audits off automatically is expressly excluded.

### 5. Shared phase evidence and agreement

After aggregation, audit, and evaluation, each peer publishes an authenticated round-completion record with its timing measurements, resulting model digest, and accepted plan digest. All peers collect the same ordered records, quantize costs to integer units, compute the next plan, and publish its digest. Execute only after all digest acknowledgments agree. This adds an explicit end-of-round planning barrier, making phase alignment deterministic; account for its waiting separately so apparent gains cannot result from moving wait time out of the old metric. Equal version-2 controls use the same phase exchange to isolate the allocation policy effect. The proposing CLI is not a permanent scheduler and may still exit after proposal.

The first plan uses the same acknowledgment procedure after initial evaluation, with the complete calibration record set bound into the run config. Calibration uses `min(512, train_limit)` examples for small datasets, records actual counts, and obeys the existing 120-second backend deadline per execution. Replay setup cost is summed over the nonempty remote contributions, since each requires a separate backend invocation.

Records are immutable per peer/round. Mismatched digests, changed records, or missing members fail the run using bounded waits (180 seconds, consistent with current peer waits); no silent membership changes or plan fallback after agreement. Apply deadline bounds also to calibration and acknowledgments. Model/audit integrity checks remain independent of self-reported speed; timing is authenticated but not hardware attested.

### 6. Report the actual outcome

Retain existing metrics and add counts, cost estimates, plan versions/digests, actual kernel throughput, calibration time, update-readiness polling time, successful transfer time, validation time, completion/plan waiting, and total round wall time. These intervals must have documented boundaries; do not subtract all network time from barrier time or call that CPU idle time. Display assigned work and waiting breakdown in peer status. Use monotonic local durations, not unsynchronized cross-host timestamps.

Add an explicit allocation comparison mode to the report CLI. Ordinary placement comparison rejects policy/index-version mismatches. Allocation comparison allows the intended policy difference but requires matching data, initial tensors, seed, global budgets, optimizer, threads, peer hardware/identity, backend, numerics, audit policy, and indexing version. Summarize observed distributions rather than requiring identical output models across policies. Require identical aggregate hashes among peers within each run.

## Risks / Trade-offs

- Full replay dominates work → preserve it and expose its limiting effect; report no speedup honestly.
- New phase agreement adds overhead → include it in both version-2 controls and all end-to-end results.
- Unequal local steps can affect convergence → compare held-out accuracy/loss and time-to-quality, not runtime alone.
- Timing noise and thermal load → warm-up, smoothing, bounded changes, randomized paired trial order.
- Determinism across Windows/Linux → canonical integer planning, explicit indexing version, cross-platform fixtures and existing portable numerics.
- Firecracker startup distorts short probes → include measured setup cost, bound tasks, update guest artifacts, test host/guest assignment parity.
- Tiny final windows or extreme imbalance → deterministic zero-work handling and exact coverage tests.

## Migration Plan

Implement pure allocation/indexing first, then backend payload validation, peer agreement, CLI and reporting. Update both laptops before requesting compute-aware mode; rebuild CIFAR Firecracker images when that backend is used. Preserve existing stored artifacts and default equal mode. Roll back by starting a new equal run; never change a running experiment's policy.

Validate with synthetic rate/delay scenarios and a small two-peer audited run before full trials. Run three paired seeds (42, 43, 44), alternating policy order, using the same initial tensors per pair and identical resource budgets. First compare version-2 equal versus compute-aware under full audit, the user's current mode. An explicitly selected trusted `audit none` experiment can separately isolate training balance; never compare different audit policies as evidence of allocator speedup.

Correctness gates: exact coverage, no rejected audit/update accepted, all per-round peer model hashes equal, bounded deterministic plans, and explicit failure on disagreement. Synthetic 2:1 speed tests without audit target at least 50% lower synchronization waiting and lower total time after warm-up. Physical release target: at least 10% median end-to-end runtime improvement including calibration and at most 1 percentage point mean final accuracy regression over the paired seeds. If full-audit trials miss the target, keep the feature experimental and report the measured limit; zero waiting is not an acceptance criterion.
