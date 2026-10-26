# MTCP Compute Lab

A runnable implementation of the protocol in
[VIT_IDF_B_MTCP_Complete_Draft.md](VIT_IDF_B_MTCP_Complete_Draft.md).
Volunteers advertise resources and accept authenticated inbound work. The requester
trains locally, audits remote gradients, recovers failures, and updates one model.

## Run the Firecracker cluster

Host requirements: **x86_64 Linux, Python 3.12+, usable `/dev/kvm`, bubblewrap,
user systemd with CPU/memory/pids/cpuset controllers**, and e2fsprogs. Artifact
preparation also uses Docker and downloads pinned upstream assets. No sudo or
privileged container is required by this implementation.

```sh
python3 tools/firecracker/prepare.py
python3 -m mtcp --demo --nodes 3 --backend kvm-microvm --port 8087
```

Open **http://127.0.0.1:8087**. Choose **Firecracker KVM microVM** as the required
isolation, select a scenario, and start training. Preparation creates a 160 MiB
read-only root filesystem image under `.mtcp/firecracker/`. The CLI's
`--backend` chooses the volunteers' actual capability; the dashboard's isolation
selector chooses what the requester requires. Incompatible requests are declined.

The three local volunteers use separate processes, identities, inbound addresses
(`127.0.0.2`–`127.0.0.4`), ports, CPU affinities, and offered bandwidth budgets.
On a host with enough CPUs they offer 1, 2, and 1 vCPUs respectively. A fresh
Firecracker guest is booted for every accepted training assignment. These are
independent protocol nodes on **one shared physical host**, rather than three
physical computers. Ctrl+C stops the coordinator and demo volunteers.

For the fast process-only demo (no artifact preparation):

```sh
python3 -m mtcp --demo --nodes 3 --port 8087
```

Choose **Bounded fixed-kernel process** in the dashboard for this backend. The
process backend is a development option and does not provide VM isolation.

## Discover compute and bandwidth

Every volunteer serves:

```text
GET  /health                 Basic health and node identity
GET  /v1/resources           Public resource advertisement
POST /v1/probe               Authenticated identity and bandwidth probe
POST /v1/assignments         Authenticated inbound work admission
```

The coordinator publishes `GET /v1/resources` for its own resources and
`GET /v1/nodes` for the discovered volunteers. The dashboard shows:

- Host CPU count and memory in the API, offered vCPUs, allowed CPU IDs, guest
  memory budget, one execution slot, and its current availability.
- Measured **host fixed-kernel samples/second**, with the benchmark and timestamp.
  This is neither a FLOPS estimate nor guest training throughput.
- Owner-configured upload/download budgets and separately measured HTTP transfer
  rates and round-trip time. The HTTP server paces application bytes to its
  bandwidth budget. This is not kernel packet shaping or an Internet speed test.
- Actual isolation backend, inbound endpoint, connection status, and heartbeat.

The coordinator authenticates each endpoint using a random challenge and MAC
before dispatching, measures 128 KiB transfers in each direction, and refreshes
those measurements every 30 seconds. Transfer rates include HTTP/setup overhead;
loopback measurements are not predictions of a remote WAN link. Resource and
capability advertisements are self-reports, not hardware attestation.

## Workload and fault scenarios

The fixed Python kernel fits `y = w1*x1 + w2*x2 + b` on 1,000 deterministic public
synthetic samples. The target weights are `2.4, -1.7, 0.6` with small added noise.
The requester computes 400 samples; the other 600 are split across **all selected
online volunteers**, proportional to their offered vCPU budgets. A requester and
one volunteer are supported for the later two-machine test.

All gradients use the same model checkpoint and are weighted by their actual
partition sizes. Contracts authenticate job/task/attempt/model identity, input and
artifact digests, CPU/memory limits, isolation, data classification, and expiration.
Selected results are recomputed independently only after complete submission.
Rejected or timed-out partitions get one trusted local recovery attempt; every
logical partition enters each update exactly once.

Fault scenarios affect the last selected volunteer at step 3:

- **Corrupt:** add an incorrect value to the returned gradient.
- **Replay:** submit the previous model version.
- **Timeout:** return after the assignment deadline.

Full auditing deterministically detects the injected corruption. Adaptive auditing
uses the draft's illustrative 60%/15%/100% policy and can miss unsampled corruption.
The default deadline is 12 seconds for KVM tasks and 3 seconds for process tasks.

## What the Firecracker backend enforces

- Pinned Firecracker binary and guest kernel checksums; pinned Docker image
  digest for the guest Python runtime. A local manifest verifies the constructed
  image and checks that the bundled kernel and guest service match the source.
- A new microVM per task, bounded vCPUs and guest RAM, read-only root drive,
  **no virtio network device**, and a narrow host-to-guest vsock request channel.
- A guest child running as UID/GID 65534, no new privileges, 2 CPU seconds,
  96 MiB process address-space limit, and no file/core output.
- A rootless bubblewrap wrapper around the VMM: separate user/mount/PID/network
  and other namespaces; only the binary, kernel, rootfs, private run directory,
  `/dev/kvm`, private `/dev`, and private `/proc` are exposed. Default Firecracker
  seccomp filters stay enabled.
- A systemd user cgroup per VM with CPU quota and allowed CPUs, `TasksMax=32`,
  and `MemoryMax=guest RAM + 256 MiB` for the VMM overhead. A wall deadline stops
  the scope and tears down the VMM and private runtime directory. A parent-death
  signal kills the launcher if its controller crashes; a systemd runtime limit
  provides an independent deadline as well.

A KVM volunteer starts its listener and advertises capability **only after a real
boot/probe succeeds**. Missing KVM, namespaces, cgroup support, or artifacts causes
startup failure; it does not fall back to ordinary processes. This rootless wrapper
is not Firecracker's official privileged jailer and is not a production security
certification. The workload remains fixed; arbitrary Python uploads are not enabled.
A malicious volunteer host can inspect guest plaintext and fabricate results.

## Later: run on two physical systems

See [docs/two-machine-lab.md](docs/two-machine-lab.md) for exact coordinator and
volunteer commands, endpoint routing, credentials, and connection checks. Both
sides must be reachable: coordinator → volunteer for work; volunteer → coordinator
for heartbeats and results. NAT traversal/automatic port forwarding is not included.

Transport is HTTP for the local lab. For cross-machine testing use a trusted
isolated LAN or SSH tunnels in **both directions**; do not expose plaintext bearer
credentials to an untrusted network. HTTPS client verification is supported, but
TLS termination/certificates must be supplied externally. Public discovery exposes
resource metadata only; assignments and probes require the per-volunteer secret.

## Persistence and verification

SQLite stores snapshots, assignments, events, and model weights in
`.mtcp/state.sqlite3`. Inspect saved runs and export JSON in the dashboard.
Interrupted runs are marked `interrupted`; automatic resume and crash-atomic
exactly-once processing are not implemented. Worker audit history resets when the
coordinator restarts.

```sh
# Fast protocol + HTTP/process tests; real-KVM tests skip unless explicitly enabled.
python3 -m unittest discover -s tests -v

# Real Firecracker guests, three inbound volunteers, adversarial runs, and limits.
MTCP_TEST_FIRECRACKER=1 python3 -m unittest discover -s tests -p test_firecracker.py -v

# Optional Playwright check against an already-running demo.
MTCP_URL=http://127.0.0.1:8087 python3 tests/browser_smoke.py
```

Playwright is optional. Set `MTCP_CHROMIUM` to an existing Chromium executable if
needed. Screenshots are written to `/tmp/mtcp-*.png`. The KVM integration tests
write observed run results to `.mtcp/firecracker-test-report.json`.

See [docs/architecture.md](docs/architecture.md) for the source map, trust
boundaries, remaining work, and upstream references. No training-speedup,
patentability, or general security guarantee is claimed by this small experiment.
