"""Rootless Firecracker backend: private namespaces + cgroup + fresh guest per task."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from .protocol import ARTIFACT


class Firecracker:
    def __init__(self, assets=".mtcp/firecracker", cpus=None, memory_mb=128):
        self.assets = Path(assets).resolve()
        self.cpus = cpus or sorted(os.sched_getaffinity(0))[:1]
        self.memory_mb = memory_mb
        for command in ("bwrap", "systemd-run", "systemctl"):
            if not shutil.which(command):
                raise RuntimeError(f"Firecracker backend requires {command}")
        with open("/dev/kvm", "rb+") as kvm:
            if fcntl.ioctl(kvm.fileno(), 0xAE00, 0) != 12:
                raise RuntimeError("KVM API unavailable")
        manifest = json.loads((self.assets / "manifest.json").read_text())
        if manifest["artifact_digest"] != ARTIFACT:
            raise RuntimeError("Guest training artifact is stale; rerun tools/firecracker/prepare.py")
        guest_source = Path(__file__).resolve().parent.parent / "guest/init.py"
        if hashlib.sha256(guest_source.read_bytes()).hexdigest() != manifest["guest_digest"]:
            raise RuntimeError("Guest service is stale; rerun tools/firecracker/prepare.py")
        for name, expected in manifest["files"].items():
            with open(self.assets / name, "rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"Firecracker artifact integrity check failed: {name}")
        subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=384M",
                        "-p", "TasksMax=32", "true"], check=True, capture_output=True, timeout=5)

    def execute(self, payload=None, timeout=8, operation="train", sentinel=None, memory_mb=None):
        memory_mb = self.memory_mb if memory_mb is None else memory_mb
        if type(memory_mb) is not int or not 128 <= memory_mb <= self.memory_mb:
            raise ValueError("Guest memory exceeds the owner allocation")
        unit = "mtcp-vm-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="mtcp-vm-") as directory:
            run = Path(directory)
            config = {
                "boot-source": {"kernel_image_path": "/kernel", "boot_args":
                    "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda ro init=/sbin/mtcp-init"},
                "drives": [{"drive_id": "root", "path_on_host": "/rootfs", "is_root_device": True, "is_read_only": True}],
                "machine-config": {"vcpu_count": len(self.cpus), "mem_size_mib": memory_mb, "smt": False},
                "vsock": {"guest_cid": 3, "uds_path": "/run/vsock"},
            }
            (run / "config.json").write_text(json.dumps(config))
            command = ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit,
                "-p", f"RuntimeMaxSec={math.ceil(timeout) + 2}s",
                "-p", f"MemoryMax={memory_mb + 256}M", "-p", "TasksMax=32",
                "-p", f"CPUQuota={100 * len(self.cpus)}%", "-p", f"AllowedCPUs={','.join(map(str, self.cpus))}",
                "bwrap", "--unshare-all", "--die-with-parent", "--clearenv", "--setenv", "PATH", "/",
                "--ro-bind", str(self.assets / "firecracker"), "/firecracker",
                "--ro-bind", str(self.assets / "vmlinux"), "/kernel",
                "--ro-bind", str(self.assets / "rootfs.ext4"), "/rootfs",
                "--bind", str(run), "/run", "--dev", "/dev", "--dev-bind", "/dev/kvm", "/dev/kvm",
                "--proc", "/proc", "--tmpfs", "/tmp", "--chdir", "/run",
                "/firecracker", "--no-api", "--config-file", "/run/config.json"]
            started = time.monotonic()
            with open(run / "console.log", "wb") as log:
                launcher = Path(__file__).with_name("vmm_launcher.py")
                process = subprocess.Popen([sys.executable, "-I", str(launcher), str(os.getpid()), *command],
                                           stdout=log, stderr=log, start_new_session=True)
                try:
                    while time.monotonic() - started < timeout:
                        if process.poll() is not None:
                            raise RuntimeError("Firecracker boot failed: " + (run / "console.log").read_text(errors="replace")[-2000:])
                        try:
                            stream = None
                            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            sock.settimeout(.2)
                            sock.connect(str(run / "vsock"))
                            sock.sendall(b"CONNECT 5000\n")
                            stream = sock.makefile("rwb", buffering=0)
                            if not stream.readline(100).startswith(b"OK "):
                                raise ConnectionError("Guest listener not ready")
                            break
                        except (OSError, ConnectionError):
                            if stream is not None:
                                stream.close()
                            sock.close()
                            time.sleep(.03)
                    else:
                        raise TimeoutError("Firecracker boot deadline exceeded")
                    with sock, stream:
                        sock.settimeout(max(.01, timeout - (time.monotonic() - started)))
                        stream.write(json.dumps(dict(operation=operation, payload=payload, sentinel=sentinel)).encode() + b"\n")
                        raw = stream.readline(16385)
                        if len(raw) > 16384 or not raw.endswith(b"\n"):
                            raise RuntimeError("Invalid guest response")
                        result = json.loads(raw)
                        if "error" in result:
                            raise RuntimeError(result["error"])
                        return dict(result, isolation="kvm-microvm", vmm="firecracker", fresh_vm=True,
                                    wall_ms=(time.monotonic() - started) * 1000)
                finally:
                    try:
                        subprocess.run(["systemctl", "--user", "stop", unit + ".scope"],
                                       capture_output=True, timeout=5)
                    finally:
                        if process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        process.wait(timeout=5)
