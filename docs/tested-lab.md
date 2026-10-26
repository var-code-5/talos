# Local validation record

Observed on 26 October 2026, on this x86_64 Linux host with 8 logical CPUs,
approximately 7.6 GiB RAM, usable KVM API 12, and Firecracker v1.17.0.
Guest kernel: 6.1.188. These are local correctness/isolation checks, not WAN
performance or production-security certification.

## Automated checks

All **21 test methods passed**:

- 10 protocol tests: authentication, tampering, resource/policy consistency,
  partition weighting, committed-result immutability, replay/duplicate rejection,
  expiration, audit policy, and saved snapshots.
- 7 live HTTP/process tests: authentication/origin restrictions, honest training,
  corruption/replay recovery, timeout recovery, incompatible-isolation rejection,
  cancellation, and a requester with one inbound volunteer.
- 4 real-KVM tests: controller SIGKILL cleanup; guest isolation and resource
  limits; public resource advertisements and inbound authentication; three-node
  honest/corrupt/replayed/timed-out training against a trusted baseline.

The optional Playwright browser check also passed: actual microVM-backed
corruption recovery, task inspector, pagination, saved-run navigation, return to
latest run, no JavaScript errors, and no page overflow at a 390 px mobile width.

## Observed guest boundary

The actual guest reported UID 65534, only the loopback interface, a read-only root
filesystem, and guest memory within its 128 MiB allocation. It could not read a
file created only on the host, write the root filesystem, or connect to an external
IP. A 256 MiB allocation failed under the guest task limit. A busy loop was killed
by the CPU limit; a stalled task hit the host wall deadline. A new VM executed
normally afterward. Killing a controller removed its live VM scope within the
crash test's four-second bound; no scope remained after test teardown.

## Observed three-node computation

The three volunteers offered 1, 2, and 1 vCPUs, on distinct allowed CPU sets, with
separate listeners on 127.0.0.2, 127.0.0.3, and 127.0.0.4. Their node metadata
correctly identified one shared physical host. The requester processed 400 samples;
volunteers processed 150, 300, and 150 samples per step.

For four steps with full independent auditing:

| Scenario | Accepted remote results | Rejected | Timed out | Locally recovered |
| --- | ---: | ---: | ---: | ---: |
| Honest | 12 | 0 | 0 | 0 |
| Corrupted gradient | 11 | 1 | 0 | 1 |
| Replayed model version | 11 | 1 | 0 | 1 |
| Missed deadline | 11 | 0 | 1 | 1 |

All four final checkpoints matched trusted full-batch training to the test's
10-decimal-place tolerance. Observed coefficients were approximately
`[1.46040971155, -1.06278103153, 0.61728008849]`. This is an intermediate four-step
model, not a claim of convergence or speedup. The detailed most recent observed
run results are stored locally in `.mtcp/firecracker-test-report.json`.

Physical two-host networking, WAN characteristics, arbitrary uploaded workloads,
and production multi-tenant hardening remain outside this validation. Follow
[two-machine-lab.md](two-machine-lab.md) for the next physical-host experiment.
