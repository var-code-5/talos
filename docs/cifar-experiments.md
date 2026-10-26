# CIFAR-10 experiments: single-system and multi-system setups

Experiment record: 26 October 2026.

**Update:** The full 10-epoch Linux–Windows run has now completed. See the
[full placement experiment](cifar-full-experiment.md) for its matched-workload
comparison, learning curves, raw measurements, and scope for invention-disclosure
supporting evidence. The earlier two-epoch validation remains documented below.

## Findings

Two equal peers completed full CIFAR-10 training on one Linux laptop using both
the original FP32 profile and the portable FP64 profile. Both full runs reached
**62.48% held-out accuracy** after 10 epochs. A separate **Linux–Windows run**
completed two epochs on 10,000 training images and reached **29.4% accuracy** on
1,000 held-out images. Every audit in those initial portable runs passed with
zero recorded weight or momentum difference, and the two peers ended with
identical model hashes within each run.

The multi-system runs establish that discovery, authenticated connections,
training, independent auditing, aggregation, and evaluation work across the two
tested machines. The full matched-workload run reached **62.48% accuracy**
in **984.694 seconds**. All 100 independent contribution audits
passed, and 50/50 aggregate round hashes matched
the portable local control. These are single-seed observations on different
hardware, not a statistically established distributed-training speedup.

## System and training method

The application uses symmetric peers. Each peer advertises its resources,
accepts connections, trains a disjoint data shard, publishes its local update,
audits the other contribution, independently computes the same sample-weighted
average, and evaluates its aggregate model. Either peer can propose a run; there
is no permanent master or slave. Membership stays fixed for that experiment.

All completed runs documented here used two peers, one PyTorch thread per peer, CPU
execution, and the **process backend**. Their training processes were not running
inside Firecracker. Firecracker/KVM isolation and CIFAR guest/host agreement were
validated separately on Linux; those checks are not measurements of the
Linux–Windows training deployment.

| Setting | Value |
| --- | --- |
| Dataset | CIFAR-10, checksum-verified archive converted to memory-mapped arrays |
| Input | 32 × 32 RGB, normalized as `(x / 255 - 0.5) / 0.5` |
| Model | Conv(3→16, 5×5), max-pool, ReLU, Conv(16→32, 5×5), max-pool, ReLU, flatten, Linear(800→10) |
| Trainable parameters | 22,058 |
| Optimizer | SGD, learning rate 0.001, momentum 0.9 |
| Batch size | 32 |
| Seed | 42 |
| Synchronization | Every 10,000 global training examples; 5,000 per peer per round |
| Auditing | Full independent replay of the other peer's local update |
| Audit condition | Elementwise absolute error ≤ `1e-6 + 1e-5 × abs(reference)`, for weights and momentum |
| Aggregation | Sample-weighted model average; each peer retains its own momentum |
| Evaluation | Each peer evaluates the aggregate model after every round |

The model and normalization follow the earlier `project1` implementation. The
synchronous local-SGD protocol here has explicitly defined synchronization and
momentum behavior; these measurements do not reproduce Hivemind's asynchronous
optimizer timing.

`fp32-native` uses native float32 training. `fp64-portable` uses double precision
for input normalization, forward/backward operations, and SGD arithmetic, with
MKLDNN disabled. Serialized round/checkpoint weights and momentum remain float32
and are promoted when restored. Both profiles keep the same audit tolerances.

## Physical setups

### Single-system setup

Two separate peer applications ran on one x86_64 Linux laptop. Each peer was
pinned to a separate logical CPU (CPU 0 and CPU 1), with one training thread.
The portable control used loopback endpoints `127.0.0.1:9111` and
`127.0.0.1:9112` and a separate cluster key/discovery group from the live LAN peers.

The host advertised 8 logical CPUs and 7,764 MiB total RAM. The saved CPU
identification is only `x86_64`; an exact processor model was not captured.
Python was 3.13.7, PyTorch was 2.13.0+cpu, and NumPy was 2.5.2.

This is a control with **two peers on one physical system**, not a single-process
centralized training baseline. It exercises the same peer protocol as the
multi-system setup.

### Multi-system setup

| Property | Linux peer | Windows peer |
| --- | --- | --- |
| Advertised name | `varcode-laptop` | `shridhar` |
| Endpoint at validation time | `172.16.203.242:9101` | `172.16.203.120:9101` |
| OS | Linux 6.19.11-arch1-1 | Windows 11, build 26200 |
| Architecture / recorded CPU | x86_64; exact model not captured | AMD64 Family 25 Model 80 Stepping 0 |
| Logical CPUs on host | 8 | 16 |
| Total host RAM | 7,764 MiB | 15,773 MiB |
| Assigned logical CPU | 4 | 0 |
| Offered vCPUs / training threads | 1 / 1 | 1 / 1 |
| Python | 3.13.7 | 3.13.17 |
| PyTorch | 2.13.0+cpu | 2.13.0+cpu |
| NumPy | 2.5.2 | 2.5.2 |
| Backend | process | process |

Both peers advertised `accepting: true`, an offered memory value of 768 MiB,
compute information, and measured peer links. For the process backend, the
advertised memory offer is not a Firecracker memory limit.

Peers discover each other through signed LAN multicast on UDP 47767 and accept
authenticated HTTP connections on TCP 9101. Explicit endpoint handshakes are
available when multicast is unavailable. The cluster key is omitted from this
report and the saved measurement artifacts.

## Completed experiment results

These columns describe different workloads and numerical profiles. They are
listed together to document what ran, not to imply a controlled speed comparison.

| Metric | Single system: native FP32 | Single system: portable FP64 | Two systems: portable FP64 validation |
| --- | ---: | ---: | ---: |
| Physical hosts | 1 | 1 | 2 |
| Peers | 2 | 2 | 2 |
| Training images per epoch | 50,000 | 50,000 | 10,000 |
| Held-out images | 10,000 | 10,000 | 1,000 |
| Epochs | 10 | 10 | 2 |
| Rounds per peer | 50 | 50 | 2 |
| Top-1 accuracy | 62.48% | 62.48% | 29.40% |
| Top-5 accuracy | 96.22% | 96.21% | 78.30% |
| Cross-entropy loss | 1.077109 | 1.077357 | 2.006947 |
| Macro F1 | 0.621233 | 0.621097 | 0.275308 |
| Slowest peer wall time (s) | 306.371 | 959.115 | 25.361 |
| Effective global examples/s | 1632.01 | 521.31 | 788.60 |
| Training HTTP JSON body volume (MiB) | 22.568 | 22.574 | 0.905 |
| Sum of peer training times (s) | 189.534 | 572.785 | 19.292 |
| Sum of peer audit times (s) | 189.155 | 575.977 | 20.204 |
| Sum of peer evaluation times (s) | 218.493 | 757.415 | 7.064 |
| Largest reported process peak RSS (MiB) | 479.26 | 549.23 | 494.68 |
| Final peer models identical | Yes | Yes | Yes |

Effective global examples/s counts original training examples across all epochs,
divided by the slowest peer's wall time. Replay audits perform additional
training work that is not added to this numerator. Wall time includes training,
auditing, evaluation, communication, and waiting. Summed peer times represent
concurrent work and must not be added to obtain experiment duration.

Network volume counts successful training HTTP JSON request/response bodies,
including barrier polling. It excludes discovery, initialization/data staging,
HTTP headers, retransmissions, and transport framing. Peak RSS is the reported
process high-water mark, not total host RAM consumption.

### Network observations

| Run / peer | RTT samples | Mean HTTP RTT (ms) | Median (ms) | Maximum (ms) |
| --- | ---: | ---: | ---: | ---: |
| Single-system FP32 / local-b | 50 | 1.433 | 0.840 | 12.930 |
| Single-system FP32 / local-a | 50 | 1.123 | 0.755 | 4.432 |
| Single-system FP64 / portable-control-b | 50 | 1.015 | 0.757 | 6.087 |
| Single-system FP64 / portable-control-a | 50 | 0.879 | 0.772 | 3.273 |
| Two-system FP64 / shridhar | 2 | 34.468 | 34.468 | 52.782 |
| Two-system FP64 / varcode-laptop | 2 | 82.551 | 82.551 | 133.952 |

RTT is authenticated HTTP application latency under training load, not ICMP
latency or a measurement of the physical link alone. The two-system run has only
two RTT samples in each direction, so its mean is particularly sensitive to
scheduling and transient delays. A separate authenticated connectivity check
returned approximately 15 ms; this is distinct from the in-training samples.

The resource advertisements taken before the two-system run also contained these
128 KiB HTTP body-transfer probes:

| Measuring peer | RTT (ms) | Upload (Mbps) | Download (Mbps) |
| --- | ---: | ---: | ---: |
| shridhar | 28.536 | 24.321 | 19.348 |
| varcode-laptop | 12.218 | 23.023 | 11.138 |

These probes include HTTP overhead and RTT. They are observations of these peer
connections, not an Internet speed test or guaranteed bandwidth. Advertised
fixed upload/download caps were unset; the measured links supplied the observed
rates. The advertised compute microbenchmark uses a separate fixed gradient
kernel and must not be interpreted as CIFAR training throughput.

### Audit and model agreement

The full portable single-system run passed all 50 rounds on both peers; the
portable two-system validation passed both rounds on both peers. Across their
saved round records, maximum weight error and maximum momentum error were both
zero. Full replay auditing also passed in the legacy FP32 local run, whose older
artifact does not contain the newer per-round maximum-error fields.

| Run | Final aggregate model SHA-256, identical on both peers |
| --- | --- |
| Single-system FP32 | `eb50654778b19fb72335f96084f3b84ab23307f29eb0bae091722f71a966756d` |
| Single-system FP64 | `3dea161e29e5c7fe24a31a831ce2cca4ee346bf5b5b9550895bc0b8c2453f46d` |
| Two-system FP64 | `1939bab67cb0dae66f8a1c64cd2e9cbf2e31ac5eb4efde36229e2199e5b9da3e` |

The two full local runs have equal rounded accuracy but different numerical
profiles and different final model hashes. Matching accuracy does not mean the
trained tensors are identical across profiles.

## Earlier failed multi-system experiment and repair

The earlier run `a58b24409529bf46a1fb0ef6`, involving a different Windows laptop
(`aayush`), failed the independent audit at round 0. It did not complete training
and contributes no final accuracy or throughput result to the table above.
Matching dataset, seed, initial weights, and training source did not prevent
native FP32 arithmetic differences between Windows and Linux.

Replaying the saved contributions on Linux produced:

| Contribution being checked | Maximum weight error | Maximum momentum error |
| --- | ---: | ---: |
| Linux update replayed on Linux | 0 | 0 |
| Windows update replayed on Linux | 0.000218009 | 0.005269990 |

Disabling MKLDNN alone in a native-FP32 replay reduced the weight error to
0.000031779, which still failed the original tolerance. The repair introduced
`fp64-portable`, preserved strict weight and momentum checks, and saved submitted
updates and detailed audit errors for diagnosis. Deliberately corrupted updates
remain rejected by regression tests. The later successful Linux–Windows run
provides the cross-platform validation for this repair on the tested machines.

Connection testing also identified Windows inbound firewall blocks and a stale
Linux address after DHCP changed the host IP. After resolving reachability, the
application was changed to refresh automatically advertised addresses while idle
and to announce the actual proposing peer during handshakes. Endpoint membership
remains fixed while a run is active.

See [the audit investigation](cifar-audit-portability.md) for the detailed failure
record. A stopped exploratory portable control (`dfb47ca0eea004dc6759d10f`)
overlapped regression testing and was excluded from the reported benchmark.

## Reproducing the setups

Prepare the same code, PyTorch/NumPy versions, checksum-verified dataset, and a
privately shared `MTCP_CLUSTER_KEY` on each participating application. Full OS
installation instructions are in [the peer setup guide](cifar-peers.md).

For a single-system portable control, run each command in a separate terminal
with the same control key. CPU IDs must be available on the host:

```sh
.venv-cifar/bin/python -m mtcp.cifar peer --name portable-control-a \
  --port 9111 --cpus 0 --threads 1 --advertise-host 127.0.0.1 \
  --discovery-port 47768 --state-dir .mtcp/cifar-portable-control/a

.venv-cifar/bin/python -m mtcp.cifar peer --name portable-control-b \
  --port 9112 --cpus 1 --threads 1 --advertise-host 127.0.0.1 \
  --discovery-port 47768 --state-dir .mtcp/cifar-portable-control/b
```

Use a different key for the LAN experiment so the control peers cannot join it.
For the two-system setup, start one application on each host using the same LAN
key. These commands reflect the successful Linux and Windows assignments:

```sh
# Linux
.venv-cifar/bin/python -m mtcp.cifar peer --name varcode-laptop \
  --port 9101 --cpus 4 --threads 1 --state-dir .mtcp/cifar-local/a
```

```powershell
# Windows PowerShell
.\.venv-cifar\Scripts\python.exe -m mtcp.cifar peer --name shridhar `
  --port 9101 --cpus 0 --threads 1
```

When necessary, add `--connect http://OTHER_HOST_IP:9101`. Confirm that both hosts
can accept inbound TCP 9101. Endpoints shown above are historical LAN addresses
and may change.

The successful short two-system validation used:

```sh
.venv-cifar/bin/python -m mtcp.cifar run --peers 2 \
  --epochs 2 --train-limit 10000 --test-limit 1000 --seed 42 \
  --sync-samples 10000 --audit full --numerics fp64-portable \
  --initial-checkpoint benchmarks/cifar/portable-reference/initial.json \
  --output results/cifar/windows-portable-validation
```

The original invocation used `local-reference/initial.json`; the portable
reference contains identical initial tensor bytes.

The full portable single-system control uses the following workload, with
`--endpoint http://127.0.0.1:9111`. For the completed full two-system run, use
`--endpoint http://127.0.0.1:9101` instead:

```sh
.venv-cifar/bin/python -m mtcp.cifar run \
  --endpoint http://127.0.0.1:9111 --peers 2 \
  --epochs 10 --train-limit 50000 --test-limit 10000 --seed 42 \
  --batch-size 32 --sync-samples 10000 --threads 1 \
  --lr 0.001 --momentum 0.9 --audit full --numerics fp64-portable \
  --initial-checkpoint benchmarks/cifar/portable-reference/initial.json \
  --output results/cifar/portable-control
```

After a matching full two-system run, compare its collected artifact:

```sh
.venv-cifar/bin/python -m mtcp.cifar compare \
  benchmarks/cifar/portable-reference/experiment.json \
  results/cifar/TWO_HOST_RUN/experiment.json \
  --output results/cifar/full-placement-comparison
```

The comparator rejects incompatible workloads and numerical profiles. Do not use
the shorter validation artifact in a full-workload performance comparison.

## Evidence and limits

| Experiment | Run ID | Saved measurements |
| --- | --- | --- |
| Full local FP32 | `2983b5c91394a6e0f7388154` | [experiment.json](../benchmarks/cifar/local-reference/experiment.json) |
| Full local FP64 | `3572dc8f3026f03b2e1d87b7` | [experiment.json](../benchmarks/cifar/portable-reference/experiment.json) |
| Two-system FP64 validation | `885b687962267fcf2f412b59` | [experiment.json](../benchmarks/cifar/windows-validation/experiment.json) |
| Full two-system FP64 | `c30c66da00ad46ed90695d21` | [experiment.json](../benchmarks/cifar/full-two-host/experiment.json) |

These artifacts include per-peer environments, configuration, round metrics,
final evaluation metrics, per-class precision/recall/F1, confusion matrices,
resource advertisements, and final model hashes. They contain no cluster key.
The successful portable runs used the same training source digest:
`683b6304bdf4c90f0d58877b7a0de8edfcba9a0d8823c48177a263bdd306fb1d`.

The experiments used one seed on development laptops with unrelated workloads;
they do not provide confidence intervals or controlled hardware comparisons.
The Windows and Linux hosts have different CPUs, RAM, and operating systems, so
future timing differences will include those effects. Resource snapshots are
point-in-time observations. macOS execution and a multi-system Firecracker
benchmark are not established by these runs.

Local validation included 31 passing test methods covering the protocol, actual
Firecracker execution, CIFAR guest/host agreement, and numerical auditing. After
the address fix, both peer regression tests passed, including symmetric training
and idle-address refresh. These correctness checks are separate from measured
training benchmarks.
