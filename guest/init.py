#!/usr/local/bin/python3
"""PID 1 in the read-only microVM. One bounded request, no network interface."""
import ctypes
import json
import os
import resource
import socket
import sys

sys.path.insert(0, "/opt/mtcp")
from kernel import gradient, samples


def mount(source, target, kind):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mount(source.encode(), target.encode(), kind.encode(), 0, None) != 0:
        raise OSError(ctypes.get_errno(), f"mount {target}")


def calculate(request):
    cifar = request.get("operation") == "cifar-train"
    cpu_limit = 120 if cifar else 2
    address_limit = 4 * 1024**3 if cifar else 96 * 1024**2
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
    file_limit = 16 * 1024**2 if cifar else 0
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    # No new privilege gain through exec, even within the guest.
    ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)
    if cifar:
        sys.path.insert(0, "/opt/app")
        os.environ.update(HOME="/tmp", TMPDIR="/tmp", TORCH_HOME="/tmp/torch")
        from mtcp.cifar.training import train
        return train(request["payload"], "/data")
    if request.get("operation") == "probe":
        checks = dict(uid=os.getuid(), interfaces=os.listdir("/sys/class/net"),
                      kernel=os.uname().release, pid=os.getpid(), vcpus=os.cpu_count(),
                      rootfs_readonly=bool(os.statvfs("/").f_flag & os.ST_RDONLY),
                      guest_memory_mb=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 1024**2)
        try:
            with open(request.get("sentinel") or "/host/mtcp-sentinel") as source:
                checks["host_file_visible"] = bool(source.read())
        except OSError:
            checks["host_file_visible"] = False
        try:
            with open("/mtcp-write-probe", "w") as target:
                target.write("unsafe")
            checks["rootfs_writable"] = True
        except OSError:
            checks["rootfs_writable"] = False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(.3)
                sock.connect(("1.1.1.1", 80))
            checks["outbound_network"] = True
        except OSError:
            checks["outbound_network"] = False
        try:
            bytearray(256 * 1024**2)
            checks["memory_limit_enforced"] = False
        except MemoryError:
            checks["memory_limit_enforced"] = True
        return checks
    if request.get("operation") == "cpu-probe":
        while True:
            pass
    if request.get("operation") == "stall-probe":
        import time
        time.sleep(120)
    p = request["payload"]
    return {"gradient": gradient(p["weights"], samples(p["seed"])[p["start"]:p["end"]])}


def main():
    mount("proc", "/proc", "proc")
    mount("sysfs", "/sys", "sysfs")
    mount("tmpfs", "/tmp", "tmpfs")
    os.chmod("/tmp", 0o1777)
    listener = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    listener.bind((socket.VMADDR_CID_ANY, 5000))
    listener.listen(1)
    connection, _ = listener.accept()
    with connection:
        stream = connection.makefile("rwb", buffering=0)
        raw = stream.readline(1048577)
        if len(raw) > 1048576 or not raw.endswith(b"\n"):
            return
        request = json.loads(raw)
        reader, writer = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(reader)
            connection.close()
            listener.close()
            try:
                result = calculate(request)
            except Exception as exc:
                result = {"error": str(exc)}
            with os.fdopen(writer, "wb") as output:
                output.write(json.dumps(result, allow_nan=False).encode())
            os._exit(0)
        os.close(writer)
        with os.fdopen(reader, "rb") as pipe:
            result = pipe.read(1048576)
        _, status = os.waitpid(pid, 0)
        if status or not result:
            result = json.dumps({"error": "Guest task terminated by resource limit", "wait_status": status}).encode()
        connection.sendall(result + b"\n")
    # Keep PID 1 alive until the controller tears down the VM.
    while True:
        import time
        time.sleep(60)


if __name__ == "__main__":
    main()
