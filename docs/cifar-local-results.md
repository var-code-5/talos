# Observed local CIFAR-10 result

Numerical profile: **legacy `fp32-native`**. A separate `fp64-portable` control
and full two-host comparison are now available in the
[full experiment report](cifar-full-experiment.md). This document retains the
historical FP32 observations.

Tested on 26 October 2026 using two symmetric CPU peers on one x86_64 Linux
host, one pinned CPU/PyTorch thread each. Both used the process backend, full
independent auditing, PyTorch 2.13.0+cpu, NumPy 2.5.2, and seed 42.

The benchmark uses the project1 CNN and normalization with explicitly defined
synchronous local SGD, rather than Hivemind's asynchronous optimizer timing.
All 50,000 training examples were used per epoch for 10 epochs, and all 10,000
held-out images were evaluated after every round. Synchronization occurred every
10,000 global training examples (50 rounds total). Every peer trained 250,000
examples and audited the other peer's corresponding work.

| Observation | Value |
| --- | ---: |
| Held-out top-1 accuracy | 62.48% |
| Held-out top-5 accuracy | 96.22% |
| Held-out cross-entropy | 1.077109 |
| Macro F1 | 0.621233 |
| Slowest peer wall time | 306.37 seconds |
| Effective global training throughput | 1632.01 examples/second |
| Mean authenticated HTTP RTT | 1.278 ms |
| Median authenticated HTTP RTT | 0.803 ms |
| Pooled observed RTT p95 | 3.306 ms |
| Training protocol JSON body volume | 22.568 MiB |
| Sum of peer compute times | 189.53 peer-seconds |
| Sum of peer audit times | 189.16 peer-seconds |
| Sum of peer evaluation times | 218.49 peer-seconds |
| Peak process RSS | 479.26 MiB |
| Independently aggregated peer checkpoints | Identical SHA-256 hashes |

Summed peer-seconds are concurrent costs and must not be added to estimate the
experiment wall time. Network volume counts successful training HTTP JSON bodies,
including barrier polling, and excludes discovery, dataset staging, headers,
TCP retransmissions and TLS framing. RTT is HTTP latency under the observed host
load, not ICMP or an isolated wire measurement. This was a development laptop
with other applications running, not a controlled throughput benchmark.

An earlier full run before the portability refactor also reached 62.48% and the
same final checkpoint. That is a reproducibility check, not a multi-seed estimate
of training variance. The recorded final baseline uses the kernel hash
`c23272ff1ae2c92278a061a433a2a9c25b25ed7c9c29f77ae03b5d4c12072437`.

The final experiment ID is `2983b5c91394a6e0f7388154`. Detailed local artifacts:

- `results/cifar/final/2983b5c91394a6e0f7388154/experiment.json`
- `results/cifar/comparison/comparison.html`
- `results/cifar/comparison/comparison.csv`
- Per-peer round CSV/JSONL and checkpoints in `.mtcp/cifar-local/a/runs/` and
  `.mtcp/cifar-local/b/runs/`.

**Update:** Full physical two-host measurements have now completed using the
portable FP64 profile. See [the full experiment](cifar-full-experiment.md).
This legacy FP32 result is not used as its performance control. For mixed
OS/hardware, observed differences include platform and hardware effects.
More paired seeds are needed before drawing statistical conclusions.

Validation: 29 test methods passed locally, including real CIFAR guest/host
weight and momentum comparisons, a >230 KiB two-state vsock transfer, symmetric
LAN discovery/training, platform-metric branches, and existing protocol/isolation
checks. The peer status page and generated report were checked in Chromium;
mobile peer-page overflow and JavaScript errors were absent. Native Windows execution has since been validated in the portable experiments.
macOS execution and the complete configured CI matrix remain pending.
