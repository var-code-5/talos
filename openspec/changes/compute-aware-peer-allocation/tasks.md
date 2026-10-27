# Tasks

## 1. Deterministic allocation and sample coverage

- [x] 1.1 Add a pure integer-cost allocation module with proportional initialization, phase-cost optimization, stable rounding, EWMA, and adjustment bounds; verify fixtures for equal, 2:1, extreme, noisy, and audit-limited rates.
- [x] 1.2 Add versioned global-window indexing and round-budget validation while preserving legacy indexing; verify exact once-per-epoch coverage across 1–8 peers, nondivisible sizes, and zero-work remainders.
- [x] 1.3 Document planner constants, optimization objective, indexing versions, and full-audit limits in the CIFAR guide; verify examples match allocator fixtures.

## 2. Calibration and protocol admission

- [x] 2.1 Add bounded authenticated CIFAR calibration with warm-up, medians, actual sample counts, initial-state isolation, and setup/evaluation timing; verify authentication, invalid measurements, busy peers, timeouts, and unchanged model/momentum.
- [x] 2.2 Add allocation/indexing capabilities and CLI/config fields in `__main__.py`, `experiment.py`, and peer admission; verify unsupported peers are rejected and omitted policy preserves legacy compatibility.
- [x] 2.3 Bind complete calibration records and planner settings into the configuration digest and persist startup overhead; verify tampering rejection and calibration-inclusive timing in a small run.
- [x] 2.4 Document commands for a version-2 equal control and compute-aware run; verify CLI parsing and recorded settings match the documented examples.

## 3. Variable training and independent agreement

- [x] 3.1 Extend host and Firecracker task payload validation with plan identity and deterministic ranges; verify incorrect ranges, counts, stale model identities, and artifact mismatches are rejected before execution or aggregation.
- [x] 3.2 Update training, sample-weighted aggregation, and audit replay for variable ranges and persistent per-peer momentum; verify unequal full-audit rounds and zero-work rounds preserve correct optimizer state and identical peer aggregate hashes.
- [x] 3.3 Add immutable completion records and all-peer plan digest acknowledgments, including the first round; verify proposer exit, changed records, disagreement, peer loss, and bounded failure behavior with integration tests.
- [x] 3.4 Integrate bounded adaptation and retain last valid rates for zero-work peers; verify deterministic plan sequences for injected slowdowns and stable plans under noise.
- [ ] 3.5 Update CIFAR artifact hashing/guest preparation and backend parity tests; verify host/guest indexing and audit parity with prepared KVM assets, recording unavailable-platform validation explicitly.
- [x] 3.6 Document upgrade/restart requirements, Firecracker rebuilds, phase synchronization, and rollback to a new equal run; verify instructions against a local two-peer smoke run.

## 4. Timing evidence and comparisons

- [x] 4.1 Instrument separate update polling, transfer, validation, completion waiting, plan agreement, and calibration intervals; verify timing boundaries using controlled delays and ensure overlap is not double-counted as idle time.
- [x] 4.2 Expose per-peer assignment counts, measured rates, predicted costs, audit-limited status, and timing breakdown in saved artifacts and the existing peer UI; verify active/completed status and restored legacy summaries.
- [x] 4.3 Add explicit allocation-policy comparison and retain strict placement comparison guards; verify allowed policy differences, rejected workload/hardware/indexing mismatches, accuracy reporting, and legacy report compatibility.
- [x] 4.4 Document interpretation of waiting versus CPU idle time, calibration-inclusive runtime, and unequal-SGD quality differences; verify generated report labels convey these distinctions.

## 5. Integrated performance validation

- [x] 5.1 Run the CIFAR test suite and controlled 2:1-delay experiments; verify exact coverage, audit correctness, identical within-run hashes, and at least 50% reduced synchronization waiting with improved total time in the no-audit synthetic scenario.
- [ ] 5.2 Run a small Linux–Windows full-audit comparison before full experiments; verify both peers accept identical plans across allocation changes and all replay checks pass.
- [ ] 5.3 Run full-audit equal/compute-aware paired experiments for seeds 42, 43, and 44 with identical per-pair initial tensors and alternating order; save both peers' evidence and a comparison report including all waiting phases and calibration overhead.
- [ ] 5.4 Assess the 10% median end-to-end improvement and at most one percentage point mean accuracy regression targets; record whether the feature meets them or remains experimental, including any full-audit speed limit without claiming unmeasured improvements.
