# Spec Delta

## Purpose

Distribute synchronous CIFAR training across heterogeneous peers using measured execution costs while preserving auditable sample coverage and making performance trade-offs visible.

## ADDED Requirements

### Requirement: Explicit compatible allocation policy
The system SHALL offer equal and compute-aware allocation, default to equal, persist policy and version, and reject compute-aware runs when any member lacks the required protocol. Existing equal-run artifacts SHALL remain readable.

#### Scenario: Unsupported participant
- **WHEN** a compute-aware run includes an older peer
- **THEN** admission fails with an upgrade explanation before training begins.

#### Scenario: Existing experiment
- **WHEN** an older artifact has no allocation policy
- **THEN** it is interpreted as legacy equal allocation without rewriting its sample order.

### Requirement: Workload-specific calibration
Compute-aware runs SHALL measure CIFAR execution under each peer's offered resources, backend, and numerical profile before model training. Calibration SHALL leave the initial model and optimizer unchanged, record its overhead, and reject invalid measurements before admission.

#### Scenario: Faster peer
- **WHEN** calibration measures a peer at twice another peer's throughput with equal fixed overhead and no auditing
- **THEN** its proposed sample allocation is approximately twice as large, subject to batch rounding and allocation bounds.

### Requirement: Complete deterministic sample coverage
Peers SHALL agree on disjoint assignments covering every selected training example exactly once per epoch. Each round SHALL retain the configured global sample budget except the epoch remainder. Assignments SHALL be bound to run, epoch, round, membership, and model identity.

#### Scenario: Uneven remainder
- **WHEN** an epoch's final round has fewer examples than peers or a non-batch-sized remainder
- **THEN** the plan accounts for every remaining example exactly once, permits zero-work participants, and records their zero contribution weight.

### Requirement: Agreed bounded adaptation
Peers SHALL independently derive and acknowledge the same plan from authenticated shared measurements before executing it. Adaptation SHALL smooth noisy measurements and limit allocation changes. Missing participants or conflicting plans SHALL fail explicitly rather than silently change membership or train divergent assignments.

#### Scenario: Background load changes
- **WHEN** a peer's measured execution cost increases across rounds
- **THEN** subsequent agreed plans adjust within the configured limits and record the measurements and decision.

#### Scenario: Plan disagreement
- **WHEN** peers calculate different digests for the next assignment
- **THEN** no peer executes that assignment and the run reports a planning failure.

### Requirement: Audit-aware optimization
The allocator SHALL include training, replay audit, and evaluation costs in its predicted round cost, retain the selected audit policy, and expose cases where redistribution offers no predicted end-to-end improvement. It SHALL NOT claim that lower waiting alone proves faster training.

#### Scenario: Full replay limits speedup
- **WHEN** two peers each replay all of the other's samples and redistribution merely moves work between training and auditing
- **THEN** the report identifies the audit constraint and reports predicted and measured total cost without reducing audit coverage.

### Requirement: Correct variable-work updates
Aggregation SHALL weight updates by actual assigned sample counts. Audits SHALL replay the exact remote assignment and that peer's optimizer state. Accepted updates SHALL match assignment identity and sample count; per-peer momentum SHALL persist across allocation changes and zero-work rounds.

#### Scenario: Incorrect returned assignment
- **WHEN** a peer returns a valid model with a wrong count or plan identity
- **THEN** that update is rejected before aggregation.

#### Scenario: Variable assignments pass full replay
- **WHEN** peers complete unequal assignments and full audits pass
- **THEN** all peers produce the same aggregate model digest for each round.

### Requirement: Evidence and controlled comparison
Reports SHALL expose assigned samples, measured rates, predicted cost, training, audit, evaluation, plan agreement, update polling, and transfer timing, plus calibration-inclusive runtime. Explicit policy comparisons SHALL identify changed optimization trajectories and compare accuracy as well as runtime; ordinary placement comparisons SHALL reject incompatible allocation policies.

#### Scenario: Performance comparison
- **WHEN** equal and compute-aware runs use matching data, initial tensors, seed, resources, numerical profile, audit policy, and global round budget
- **THEN** the allocation comparison shows both policies, sample plans, waiting, end-to-end time, and held-out quality without claiming identical SGD trajectories.
