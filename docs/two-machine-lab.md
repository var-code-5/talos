# Two-machine lab

The local demo has been tested with multiple listeners and real Firecracker guests
on one host. Physical two-host networking is a **later validation step**, not a
result claimed by the current tests.

## Choose endpoints and secrets

Example isolated-LAN addresses (replace with your actual addresses):

- Machine A: requester/coordinator, `192.168.50.10:8087`.
- Machine B: volunteer, `192.168.50.20:9101`.

Generate a 32-byte key once with `python3 -c 'import secrets; print(secrets.token_hex(32))'`
and provision the same key to both terminals as `<worker-key>`. Do not commit it.
Both systems need the same code, especially `mtcp/kernel.py`. The volunteer needs
KVM and the prepared Firecracker artifacts; the requester does not need KVM.
Keep both system clocks synchronized because contract expirations use wall time.

Machine A:

```sh
export MTCP_WORKER_TOKENS='{"volunteer-b":"<worker-key>"}'
python3 -m mtcp --host 0.0.0.0 --port 8087
```

Machine B:

```sh
python3 tools/firecracker/prepare.py
export MTCP_WORKER_TOKEN='<worker-key>'
python3 -m mtcp worker --id volunteer-b \
  --url http://192.168.50.10:8087 \
  --listen-host 0.0.0.0 --listen-port 9101 --advertise-host 192.168.50.20 \
  --backend kvm-microvm --cpus 0 --memory-mb 128 \
  --upload-mbps 25 --download-mbps 50
```

Choose a CPU ID from `python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))'`.
Configured bandwidth is the owner's offered application pacing budget. Choose
values appropriate to the volunteer; these values do not measure the LAN link.

Open the dashboard on Machine A at **http://127.0.0.1:8087**. The dashboard is
intentionally restricted to localhost even when the worker control plane listens
on all interfaces. Start a KVM run after the volunteer appears as connected.

One volunteer receives all 600 remote samples; the requester computes 400. With
additional volunteers, remote samples split in proportion to offered vCPUs.

## Verify both connection directions

From Machine A:

```sh
curl -fsS http://192.168.50.20:9101/v1/resources
curl -fsS http://127.0.0.1:8087/v1/nodes
```

From Machine B:

```sh
curl -fsS http://192.168.50.10:8087/health
curl -fsS http://192.168.50.10:8087/v1/resources
```

The coordinator's node listing should show the volunteer endpoint and the local
dashboard should show a verified connection with transfer rates. A health check
alone does not prove that authenticated dispatch works; run honest training and
confirm accepted/audited remote work, then inject corruption and inspect recovery.

Configure your lab firewall to allow Machine A to reach B:9101 and B to reach
A:8087. No automatic firewall changes are made. A wildcard listening address is
not a routable advertised address; `--advertise-host` must identify the address
that the coordinator can actually reach.

## SSH-tunnel alternative

If B can SSH to A, keep services on loopback and run this tunnel from B:

```sh
ssh -N -L 18087:127.0.0.1:8087 -R 19101:127.0.0.1:9101 user@machine-a
```

Run A's coordinator on loopback port 8087. Run B's worker with:

```sh
export MTCP_WORKER_TOKEN='<worker-key>'
python3 -m mtcp worker --id volunteer-b \
  --url http://127.0.0.1:18087 \
  --listen-host 127.0.0.1 --listen-port 9101 --advertise-host 127.0.0.1 --advertise-port 19101 \
  --backend kvm-microvm --cpus 0
```

The worker's locally bound port and externally advertised port differ through the
tunnel. Supply `--advertise-port 19101` in the worker command above so registration
publishes the coordinator-side reverse-forward port. The SSH server must permit
forwarding. This setup encrypts both control directions without exposing the HTTP
listeners to the LAN.

## Before declaring two-host support validated

Record the host CPU/memory/OS and KVM availability, advertised and measured
bandwidth, artifact manifest hashes, honest/reference model weights, injected
failure results, and behavior when either connection disappears. The tiny seeded
workload is for correctness testing, not proof of useful distributed speedup.
