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
Native Windows/macOS validation is pending; the repository includes a three-OS CI
matrix, and local tests cover the platform-specific metric/affinity branches.

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

The local full-data result for this development stage is recorded in
[cifar-local-results.md](cifar-local-results.md). The physical two-host run is
pending the friend's application becoming reachable. Native Windows/macOS CI has
been configured, not executed from this local workspace.

For a cross-platform comparison, copy the control run's `initial.json` and add
`--initial-checkpoint /path/to/initial.json` when proposing the separate-host run.
The run configuration carries the exact initial tensor bytes to every peer,
independent of their platform's RNG implementation. Every peer still trains and
aggregates independently. The comparison tool rejects different initial models
for the same seed. Full audit tolerances are `atol=1e-6`, `rtol=1e-5`; native
cross-platform numerical behavior remains part of the pending validation.

A compact measured baseline and the verified initial model are committed under
`benchmarks/cifar/local-reference/`. To match that control trial from either laptop:

```sh
python -m mtcp.cifar run --peers 2 --epochs 10 --seed 42 --audit full \
  --initial-checkpoint benchmarks/cifar/local-reference/initial.json
```

Signed multicast discovery tolerates at most ten seconds of clock skew. Keep
system clocks synchronized, or use the authenticated `--connect` fallback.
