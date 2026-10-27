# CIFAR-10 on equal peers

The CIFAR application has **no permanent master, requester, or central aggregator**.
Every peer advertises, accepts connections, trains a disjoint data shard, publishes
its update, optionally audits other updates, independently averages the same
ordered contributions, and evaluates its resulting model. A run can be proposed
from either laptop. The proposing CLI can exit after proposal; peers keep training.
Membership is fixed during a measured experiment. A missing peer fails the run;
it is not silently replaced with local work in a supposedly distributed result.

The older linear-regression coordinator demo remains a separate compatibility
example. It is not used by these CIFAR experiments.

## Prepare each laptop

Use Python 3.12+ and CPU-only PyTorch for a comparable CPU experiment:

```sh
python3 -m venv .venv-cifar
.venv-cifar/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv-cifar/bin/pip install numpy==2.5.2
```

Prepare the public official CIFAR-10 Python archive once on each laptop:

```sh
.venv-cifar/bin/python -m mtcp.cifar prepare \
  --archive /path/to/cifar-10-python.tar.gz --output .mtcp/cifar10
```

This machine's existing source archive is at
`/home/varcode/Projects/project1/data/cifar-10-python.tar.gz`. Preparation checks
the official archive checksum and creates memory-mapped arrays. Runtime and wire
formats use no pickle or executable serialization. Prepared manifests must match
on both peers. Source: [official CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Generate one experiment key and share it privately with the friend:

```sh
python3 -c 'import secrets; print(secrets.token_hex(32))'
export MTCP_CLUSTER_KEY='<same-generated-key-on-both-laptops>'
```

Do not use different keys: discovery messages, HTTP responses, and control
requests authenticate the intended experiment group. The key is not advertised.

Start this **same application** on both laptops:

```sh
.venv-cifar/bin/python -m mtcp.cifar peer --name my-laptop --port 9101 --threads 1
```

On the friend laptop change only `--name`. Each application listens on all IPv4
interfaces and advertises its LAN endpoint. Signed multicast on
`239.255.77.67:47767/UDP` discovers peers automatically on the same LAN. HTTP uses
TCP port 9101. Both directions must be reachable. No firewall rules or NAT port
forwarding are changed automatically.

If multicast is blocked, use a known reachable endpoint:

```sh
.venv-cifar/bin/python -m mtcp.cifar peer --name friend-laptop \
  --connect http://YOUR_LAPTOP_LAN_IP:9101
```

Both sides learn each other through the authenticated handshake. For an unusual
routing setup use `--advertise-host YOUR_REACHABLE_IP`. Inspect public resource
metadata at `http://localhost:9101/v1/resources` and the discovered membership at
`http://localhost:9101/v1/peers`. Measured peer links report authenticated HTTP RTT
and 128 KiB body-transfer throughput; this is not an Internet speed estimate.

## Start training from either laptop

```sh
.venv-cifar/bin/python -m mtcp.cifar run --endpoint http://127.0.0.1:9101 \
  --peers 2 --epochs 10 --seed 42 --audit full
```

Both peers train automatically after agreeing on the same run configuration.
Use `--no-wait` if the proposal command should exit immediately. Otherwise it
collects each peer's result and verifies identical final aggregate model hashes.
The default uses all 50,000 training examples and all 10,000 held-out examples.
`--train-limit` and `--test-limit` create explicit seeded subsets for smoke tests;
reports retain those limits and reject incompatible comparisons.

The model, normalization, batch size 32, learning rate 0.001, and momentum 0.9
match `project1/local.py` and `friend.py`. Synchronization is explicitly defined
here as sample-weighted synchronous local SGD: by default 10,000 total examples
per round, split across the peers. Per-peer momentum persists through model
averaging. This is not a claim to reproduce Hivemind's asynchronous timing or
optimizer semantics. Both compared placements use this same peer implementation.

Full auditing independently recomputes each other peer's local-SGD round and
checks both model weights and optimizer momentum before aggregation. It adds
real compute cost, recorded separately. `--audit none` is an explicitly trusted
performance experiment: authentication alone does not establish correct training.
Keep the same audit policy in both compared placements.

## Single-host control experiment

Run two instances on the same host, with the same group key and the same CPU
thread budgets. Example CPU IDs must exist in `os.sched_getaffinity(0)`:

```sh
.venv-cifar/bin/python -m mtcp.cifar peer --name local-a --port 9101 --cpus 0
.venv-cifar/bin/python -m mtcp.cifar peer --name local-b --port 9102 --cpus 1
```

Use a separate experiment key while the friend is online so discovery cannot
mix three nodes into a two-peer control run. Peer identities are persisted per
port/state directory. The experiment infers `single-host-peers` or
`separate-host-peers` from physical host identity metadata, rather than trusting
a user-supplied topology label.

Run the exact same training command and seed as for the physical two-host trial.
Repeat with several paired seeds for a useful estimate of variance. Keep model,
versions, data limits, batch/sync sizes, thread allocations, backend, and audit
policy fixed. Record CPU model/OS and unrelated host load when interpreting speed.

## Results

Each peer saves `config.json`, per-round JSONL/CSV, final summary, and a safe
JSON/base64 checkpoint under `.mtcp/cifar-peer-PORT/runs/RUN_ID/`. The collecting
CLI saves both peers' summaries under `results/cifar/RUN_ID/experiment.json`.

Metrics include held-out top-1/top-5 accuracy, cross-entropy, macro F1, per-class
precision/recall/F1 and confusion matrix; sample-weighted training loss/accuracy;
wall time, training throughput, kernel time, CPU time, peak process RSS, evaluation
time, audit time, barrier time, authenticated HTTP RTT, body bytes, and effective
HTTP body throughput. Bytes exclude transport headers, retransmission, and TLS
framing. Barrier/network times can overlap and should not simply be added.

```sh
.venv-cifar/bin/python -m mtcp.cifar compare \
  results/cifar/LOCAL_RUN/experiment.json \
  results/cifar/TWO_HOST_RUN/experiment.json \
  --output results/cifar/comparison
```

Open the generated `comparison.html`; CSV/JSON accompany it. Workload mismatches
are rejected. A report containing only local results explicitly says the physical
two-host comparison is pending. No remote accuracy or latency is invented.

Transport is plaintext HTTP on the trusted lab LAN. The shared key authenticates
participants but does not encrypt traffic or distinguish malicious key holders.
Use encrypted routing/TLS termination for an untrusted network. Benchmark peers
currently run fixed trusted code in their own processes; Firecracker support is
an additional backend, with its availability and configuration recorded explicitly.

## Windows and macOS

The **process backend is portable**; it does not require Linux or KVM. The pinned
stack targets Windows x86-64 and Apple Silicon macOS 14+ (the published PyTorch
2.13 wheel is ARM64). Intel Macs are not supported by this pinned PyTorch release.
Native Windows execution is validated by the [full Linux–Windows experiment](cifar-full-experiment.md).
macOS execution and the complete three-OS CI matrix remain pending; local tests
cover the platform-specific metric/affinity branches.

Windows PowerShell, using Python 3.13:

```powershell
py -3.13 -m venv .venv-cifar
.\.venv-cifar\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-cifar\Scripts\python.exe -m pip install numpy==2.5.2 psutil==7.2.2
.\.venv-cifar\Scripts\python.exe -m mtcp.cifar prepare --download
$env:MTCP_CLUSTER_KEY = '<same-generated-key>'
.\.venv-cifar\Scripts\python.exe -m mtcp peer --name friend-windows --port 9101 --backend process
```

Apple Silicon macOS, using Python 3.13:

```sh
python3.13 -m venv .venv-cifar
.venv-cifar/bin/python -m pip install torch==2.13.0 numpy==2.5.2 psutil==7.2.2
.venv-cifar/bin/python -m mtcp.cifar prepare --download
export MTCP_CLUSTER_KEY='<same-generated-key>'
.venv-cifar/bin/python -m mtcp peer --name friend-mac --port 9101 --backend process
```

Allow the Python application's incoming connections in the OS firewall if
prompted. If multicast discovery does not cross the network, append
`--connect http://OTHER_LAPTOP_IP:9101`. CPU training is used on every platform;
CUDA and Apple MPS are not enabled. macOS does not expose the same hard CPU
pinning API, so the app records `affinity_enforced=false` while still limiting
PyTorch threads. Such runs compare placement **and platform/hardware**, rather
than isolating networking alone. CPU build suffixes may differ across platforms;
peers require the same PyTorch base version and NumPy version.

The **Firecracker backend requires Linux/KVM** and is rejected explicitly on
Windows/macOS. See the [Firecracker prerequisites](https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md)
and [PyTorch platform installation documentation](https://pytorch.org/get-started/locally/).

## Firecracker CIFAR backend on Linux

The CIFAR image is separate from the small linear-regression image. It contains
CPU-only PyTorch, the fixed CIFAR kernel, and the checksum-verified public data:

```sh
python3 tools/firecracker/prepare.py
.venv-cifar/bin/python tools/firecracker/prepare_cifar.py
.venv-cifar/bin/python -m mtcp peer --name linux-peer --backend firecracker --memory-mb 768
```

Use that backend on **both peers** when comparing placements. Each local-SGD round
runs in a fresh read-only-root guest, with guest-only writable temporary storage,
no network device, a bounded vsock channel, unprivileged task execution, and the
host VMM namespace/cgroup controls. The controller enforces a 120-second guest
wall deadline. Model and optimizer momentum use bounded float32/base64 frames;
large (>230 KiB) two-state transfers are tested without pickle.

This image uses disk-backed temporary build space under `.mtcp/build-tmp`, since
its rootfs archive can exceed a `/tmp` memory-filesystem quota. Docker is used to
construct the runtime image; it is not the execution isolation boundary.

The real-guest test compares host and guest weights/momentum and then transfers
both into another fresh guest. The full 10-epoch baseline recorded for this stage
uses the **process backend**; it is not presented as a full Firecracker benchmark.

```sh
MTCP_TEST_FIRECRACKER=1 .venv-cifar/bin/python -m unittest discover -s tests -v
```

Open `http://localhost:9101` for the peer's resource, discovery, and experiment
status page. It restores completed local summaries after restart. Run control
continues to require the experiment key through the CLI. The older linear demo
at port 8087 is independent from this equal-peer CIFAR application.

To collect a run started with `--no-wait` later, keep both peer applications running:

```sh
.venv-cifar/bin/python -m mtcp.cifar collect --endpoint http://127.0.0.1:9101 --run-id RUN_ID
```

The full single-system and physical Linux–Windows results are recorded in
[cifar-full-experiment.md](cifar-full-experiment.md). Both placements completed
10 epochs, all audits passed, and all 50 aggregate round hashes matched.
The complete Windows/macOS CI matrix has not been executed from this workspace.

For a cross-platform comparison, copy the control run's `initial.json` and add
`--initial-checkpoint /path/to/initial.json` when proposing the separate-host run.
The run configuration carries the exact initial tensor bytes to every peer,
independent of their platform's RNG implementation. Every peer still trains and
aggregates independently. The comparison tool rejects different initial models
for the same seed. Full audit tolerances are `atol=1e-6`, `rtol=1e-5`; the tested portable profile
passed full cross-platform auditing without relaxing these bounds.

A compact portable control and the verified initial model are committed under
`benchmarks/cifar/portable-reference/`. To match that control trial from either laptop:

```sh
python -m mtcp.cifar run --peers 2 --epochs 10 --seed 42 --audit full --numerics fp64-portable \
  --initial-checkpoint benchmarks/cifar/portable-reference/initial.json
```

Signed multicast discovery tolerates at most ten seconds of clock skew. Keep
system clocks synchronized, or use the authenticated `--connect` fallback.

## Cross-platform full audits and numerical profiles

The first Windows/Linux full-data attempt exposed a native FP32 replay mismatch.
The same Linux update reproduced exactly, while recomputing the Windows update
on Linux differed by a maximum `0.000218` in weights and `0.005270` in optimizer
momentum after 157 local SGD batches. Matching source, seed, and dataset does not
guarantee equal floating-point paths across operating systems/CPU libraries.
See [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

New runs default to **`--numerics fp64-portable`**: input normalization, forward /
backward operations, and SGD momentum arithmetic use float64; MKLDNN is disabled.
The shared round/checkpoint format remains float32, and restored momentum is
promoted back to float64 before the next local round. This reduces accumulation
of platform-specific roundoff while retaining the existing audit tolerances
(`atol=1e-6`, `rtol=1e-5`). It is a reproducibility measure, not a mathematical
cross-platform guarantee. The full Linux–Windows validation is recorded in
[the experiment report](cifar-full-experiment.md).

`fp32-native` remains an explicit profile for the original baseline and homogeneous
experiments. The comparison tool rejects mixed numerical profiles. The original
62.48% reference is FP32 and cannot serve as the FP64 performance baseline.
A separate portable-profile control and full two-host run are now measured
and archived under `benchmarks/cifar/portable-reference/` and `full-two-host/`.

After updating and restarting **both** peer applications, first check two complete
5,000-example local rounds, including momentum restoration:

```sh
python -m mtcp.cifar run --peers 2 --epochs 2 --train-limit 10000 \
  --test-limit 1000 --sync-samples 10000 --audit full --numerics fp64-portable \
  --initial-checkpoint benchmarks/cifar/local-reference/initial.json
```

Then use the full 10-epoch command with the same numerical profile. There are no
new dependencies or dataset conversions for this update. Firecracker users must
rebuild the CIFAR guest image because the execution artifact changed.

Audit failure messages now include maximum errors. Per-round evidence is saved
under the peer's `runs/RUN_ID/audits/`, and submitted local states are persisted in
`commits/` for investigation after restart. No failed audited update is aggregated.

## Compute-aware allocation (experimental)

Update and restart **every peer** before using the new policy. Existing commands
still use equal allocation and legacy strided data indexing. Compute-aware runs
measure the actual CIFAR kernel at each peer's offered CPU/thread budget, backend,
and numerical profile; hardware core counts and the linear benchmark do not set
training shares.

```sh
.venv-cifar/bin/python -m mtcp.cifar run --peers 2 --epochs 10 --seed 42 \
  --audit full --numerics fp64-portable --allocation compute-aware
```

For a controlled allocation comparison, use a new equal run with the same global
window indexing and phase agreement protocol:

```sh
.venv-cifar/bin/python -m mtcp.cifar run --peers 2 --epochs 10 --seed 42 \
  --audit full --numerics fp64-portable --allocation equal --indexing-version 2
```

Copy that control's `initial.json` and add `--initial-checkpoint PATH` to the
compute-aware command. Keep peers, CPU affinities, threads, data, numerical
profile, audit policy and global round budget identical. Compare the collected
artifacts explicitly:

```sh
.venv-cifar/bin/python -m mtcp.cifar compare \
  results/cifar/EQUAL_RUN/experiment.json results/cifar/AWARE_RUN/experiment.json \
  --allocation-comparison --output results/cifar/allocation-comparison
```

Calibration uses one warm-up plus three measured executions (up to 512 examples)
and an evaluation, with a 120-second deadline per isolated task. It never updates
the experiment model or momentum. Interpreter startup is counted as calibration
overhead, while Firecracker boot is a real recurring task setup cost. Admission
binds each peer's own calibration into the configuration. Invalid, missing or
unsupported measurements fail before training. Calibration and discovery overhead
are included in `end_to_end_seconds`; `wall_seconds` measures peer execution.

Version-2 indexing shuffles the selected examples globally per epoch, then splits
each synchronization window into disjoint ranges. Exactly every selected example
is trained once per epoch; tiny remainder windows can assign zero samples to a
peer. That peer preserves momentum and contributes zero aggregation weight.
Per-peer momentum persists when shares change. Unequal numbers of local SGD steps
change the optimization trajectory: compare accuracy and loss alongside runtime,
without expecting the equal-policy final model to match the compute-aware model.
All peers within a single run must still agree on every aggregate model hash.

Without audits and with negligible fixed cost, shares approach measured samples
per second. With full audits the planner minimizes the predicted training barrier
plus the slowest replay-and-evaluation phase. Every peer recomputes all remote
samples under full auditing, so redistribution may offer little or no speedup;
`audit_limited` reports this condition. Auditing is never reduced automatically.
The planner uses integer microsecond costs, deterministic batch rounding, a
10,000-candidate search limit, EWMA alpha 0.2, per-observation cost clipping to
0.5–2 times the previous estimate, and a 2% minimum predicted improvement. Share
changes are limited to 10 percentage points (rounded to a batch when necessary)
relative to the previous proportions. Initial calibration allocation is exempt.
Short epoch tails preserve the last full-window shares and do not update
per-sample rates from less than one batch of work.
All peers acknowledge the same next plan; a missing peer or disagreement fails
the run instead of silently changing membership.

The status page shows assigned counts, predicted compute phases and update,
completion and plan waits. `update_polling_seconds` includes unsuccessful HTTP
polls, retry delays and polling predicate overhead; `update_transfer_seconds`
measures the successful authenticated update request. Local update validation is
recorded separately. `barrier_seconds` covers their enclosing update exchange.
Completion and plan-agreement waits include their HTTP exchange. Network durations
can overlap those enclosing phases; **do not add them twice or interpret these
measurements as CPU idle time**. Runtime comparisons include all phase waits and
calibration. This feature remains experimental until physical paired trials
establish runtime and quality results.

Firecracker users must rebuild with `tools/firecracker/prepare_cifar.py`; the
training artifact now binds both kernel and allocation code. Roll back by starting
a new equal run, without changing policy on a running experiment. Archived legacy
runs remain readable and keep their original indexing semantics.

To run the full paired validation automatically after both peers are updated:

```sh
.venv-cifar/bin/python -m tools.cifar.paired_allocation
```

This runs both policies for seeds 42, 43 and 44, alternates their order, reuses
exact initial tensors within each pair, and saves `assessment.json` plus HTML,
CSV and JSON comparisons under `results/cifar-allocation-paired/`. Its release
assessment requires physical peers, the full workload, at least a 10% median
runtime reduction including calibration, and at most one percentage point mean
accuracy regression. A small connectivity/audit trial uses:

```sh
.venv-cifar/bin/python -m tools.cifar.paired_allocation --epochs 2 \
  --train-limit 10000 --test-limit 1000 --seeds 42 \
  --output results/cifar-allocation-smoke
```

For a repeatable local speed-imbalance check with a synthetic delay kernel:

```sh
.venv-cifar/bin/python -m tools.cifar.validate_allocation
```

That check produces `results/cifar-allocation-validation/synthetic.json`. It is
not a CIFAR accuracy result or a physical two-laptop performance measurement.
