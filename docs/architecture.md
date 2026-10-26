# Architecture and boundaries

MTCP separates the volunteer's admission decision from the requester's result
acceptance decision. The current system uses a fixed deterministic Python training
kernel, public synthetic data, authenticated control messages, and independent
auditing. Firecracker provides a real guest boundary for the KVM backend.

## Source map

| File | Responsibility |
| --- | --- |
| `mtcp/__main__.py` | CLI, coordinator HTTP server, public discovery, localhost dashboard, demo processes |
| `mtcp/coordinator.py` | Endpoint verification, network probes, capacity-weighted partitions, dispatch, verification, recovery, snapshots |
| `mtcp/worker.py` | Public capacity/listener, authenticated inbound admission, execution, heartbeat/result callbacks |
| `mtcp/resources.py` | Host capacity, offered budgets, fixed-kernel host benchmark |
| `mtcp/transport.py` | Bounded HTTP reads, no proxy/redirect forwarding, normal HTTPS verification |
| `mtcp/firecracker.py` | Verified assets, rootless VMM namespace/cgroup, fresh guest boot, vsock, teardown |
| `mtcp/vmm_launcher.py` | Parent-death signal and race check before launching the VMM scope |
| `mtcp/protocol.py` | Canonical JSON HMACs, artifact identity, owner-policy checks |
| `mtcp/kernel.py` | Deterministic dataset, mean gradients, loss, process fallback limits |
| `guest/init.py` | Guest PID 1, vsock listener, unprivileged bounded task child, fixed diagnostic probes |
| `tools/firecracker/prepare.py` | Pinned binary/kernel/image downloads and rootless ext4 image construction |
| `tests/test_firecracker.py` | Opt-in real KVM isolation, resource/listener checks, three-node failure cases |
| `web/` | Dashboard, resources, model loss, tasks, events, historical runs |

## Connections and execution

```mermaid
sequenceDiagram
    participant R as Requester / coordinator
    participant V as Volunteer listener
    participant F as Fresh Firecracker guest
    V->>R: Register endpoint, owner policy, compute/bandwidth budgets
    R->>V: Authenticated random challenge and transfer probes
    V->>R: MAC-bound identity proof
    R->>R: Compute local 400-sample gradient
    R->>V: POST authenticated assignment at model v
    V->>V: Check policy, expiry, artifact, CPU/memory budgets
    V->>F: Start read-only, no-NIC guest; send bounded request over vsock
    F->>F: Drop task child privileges and enforce limits
    F->>V: Gradient over vsock
    V->>V: Stop cgroup and destroy guest
    V->>R: Commit authenticated result
    R->>R: Validate bindings; select private audit after commitment
    R->>R: Recompute audited results; recover failed partitions locally
    R->>R: Sample-weighted model update and SQLite snapshot
```

The requester selects all online accepting volunteers for a run and fixes their
partitions for that run. Budgets determine partition sizes; measured link rates
are displayed but not yet used for automatic offload-cost optimization. Failures
recover locally rather than rebalancing onto a different remote volunteer.
Heartbeats occur once per second, expire after five seconds, and remain active
during guest execution. The listener has one active execution slot and rejects
concurrent assignments with HTTP 409. The coordinator retries transient dispatch
failures/busy responses within the assignment deadline. An identical assignment
cannot execute twice in one running listener. Legacy pull routes remain for
compatibility, but new demo workers use inbound push connections.

## Trust and guarantees

Canonical compact sorted JSON disallows non-finite values. Pairwise pre-shared
HMAC-SHA256 keys authenticate contracts/results; they are not asymmetric signing
identities or attestation. Assignment bindings include job, logical task, attempt,
model version, input digest, and kernel artifact digest. The coordinator serializes
commitment and aggregation under one lock. Private audit coins are drawn after
copying the complete result. Numerical checks use relative tolerance `1e-8` and
absolute tolerance `1e-9`.

Each KVM execution has a distinct VM process/cgroup/run directory. The guest has
only a read-only root disk and vsock, with no host filesystem mount or virtio NIC.
The VMM itself sees only the explicitly bound assets and private runtime paths in
rootless namespaces. The host user systemd manager enforces aggregate VMM limits;
the guest kernel enforces task limits and privilege drop. No root host daemon,
privileged container, or official Firecracker jailer is used by this prototype.
The host kernel, user namespace implementation, systemd, bubblewrap, VMM, and guest
kernel are part of the trusted computing base. A VM boundary is not a proof
against host/hypervisor vulnerabilities.

The VMM launcher inherits a controller parent-death signal without using
`preexec_fn` in the multithreaded worker. Bubblewrap's parent-death handling carries
termination into its sandbox. An independent systemd `RuntimeMaxSec` limit bounds
the scope even if controller cleanup cannot run. Real-KVM tests kill a controller
with SIGKILL and verify its VM scope disappears. A crash may leave a private
temporary directory containing configuration/log files; it does not leave a live
guest. Normal completion and wall-timeout cleanup also remove the directory.

Self-reported capabilities can be dishonest. Authenticated results can be wrong.
A malicious host sees submitted data. Sampled audits may miss corruption. The
fixed guest diagnostic operations are testing tools, not arbitrary code execution
support. The application does not enable GPU passthrough, guest outbound network,
NAT traversal, certificate issuance, incentives, or decentralized scheduling.

The process backend has no VM/filesystem/network isolation and remains explicitly
identified in contracts, advertisements, and execution receipts. It never
satisfies a `kvm-microvm` requirement. Requester-side local recovery is recorded
as attempt 2, rather than presented as remote isolated execution.

SQLite snapshots support inspection/checkpoints, not multi-coordinator scheduling
or transactional crash recovery. Active runs become `interrupted` on restart.
Listener replay history and requester audit reputation are in memory. Physical
multi-host operation still needs the lab validation in
[two-machine-lab.md](two-machine-lab.md).

## Upstream references

The implementation follows Firecracker's documented
[boot configuration and KVM requirements](https://github.com/firecracker-microvm/firecracker/blob/v1.17.0/docs/getting-started.md)
and [host-initiated vsock handshake](https://github.com/firecracker-microvm/firecracker/blob/v1.17.0/docs/vsock.md).
The official [production host setup guidance](https://github.com/firecracker-microvm/firecracker/blob/v1.17.0/docs/prod-host-setup.md)
and [jailer documentation](https://github.com/firecracker-microvm/firecracker/blob/v1.17.0/docs/jailer.md)
remain relevant before accepting untrusted uploaded programs in a production service.
