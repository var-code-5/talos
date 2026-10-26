#!/usr/bin/env python3
"""Provision local microVM artifacts. Downloads pinned upstream release/image."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
VERSION = "v1.17.0"
RELEASE_SHA = "06094a1108ae9e82aa4c23a775aa92758f53f1175d422270d9d6162cb9ade558"
BINARY_SHA = "99ad0f5cd0514a88aad0e9ae8cfdb3cc3b4ab9d190e1194602406c786b5de7a5"
IMAGE = "python:3.12-alpine@sha256:b658b05ce9c4eb97ec6fc19424818a1f31993659c2b505a532f470e83807815d"
KERNEL_URL = "https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/20261023-dd7c24b0d165-0/x86_64/vmlinux-6.1.188"
KERNEL_SHA = "fc7a33d23a629d64db61ddf12789c737c12edb1702a5ce9d5546b95ca21dc6ca"


def sha(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / ".mtcp/firecracker"))
    args = parser.parse_args()
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if os.uname().machine != "x86_64":
        parser.error("This pinned artifact recipe currently supports x86_64 only")
    with tempfile.TemporaryDirectory(prefix="mtcp-build-") as temporary:
        work = Path(temporary)
        archive = work / "firecracker.tgz"
        print("Downloading pinned Firecracker release", flush=True)
        if not (target / "firecracker").exists() or sha(target / "firecracker") != BINARY_SHA:
            urllib.request.urlretrieve(f"https://github.com/firecracker-microvm/firecracker/releases/download/{VERSION}/firecracker-{VERSION}-x86_64.tgz", archive)
            if sha(archive) != RELEASE_SHA:
                raise RuntimeError("Firecracker release checksum mismatch")
            with tarfile.open(archive) as bundle:
                member = bundle.getmember(f"release-{VERSION}-x86_64/firecracker-{VERSION}-x86_64")
                (target / "firecracker").write_bytes(bundle.extractfile(member).read())
        (target / "firecracker").chmod(0o755)
        print("Downloading pinned guest kernel", flush=True)
        if not (target / "vmlinux").exists() or sha(target / "vmlinux") != KERNEL_SHA:
            urllib.request.urlretrieve(KERNEL_URL, target / "vmlinux")
        if sha(target / "vmlinux") != KERNEL_SHA:
            raise RuntimeError("Guest kernel checksum mismatch")
        subprocess.run(["docker", "pull", IMAGE], check=True)
        container = subprocess.check_output(["docker", "create", IMAGE], text=True).strip()
        try:
            subprocess.run(["docker", "export", "-o", str(work / "root.tar"), container], check=True)
        finally:
            subprocess.run(["docker", "rm", container], check=True, stdout=subprocess.DEVNULL)
        rootfs = work / "root"
        rootfs.mkdir()
        # Rewrite image-absolute symlinks to root-relative links for safe extraction.
        def image_filter(member, destination):
            if member.issym() and member.linkname.startswith("/"):
                member.linkname = os.path.relpath(member.linkname.lstrip("/"), str(Path(member.name).parent))
            return tarfile.data_filter(member, destination)
        with tarfile.open(work / "root.tar") as bundle:
            bundle.extractall(rootfs, filter=image_filter)
        for folder in ("proc", "sys", "dev", "tmp", "opt/mtcp", "sbin"):
            (rootfs / folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "mtcp/kernel.py", rootfs / "opt/mtcp/kernel.py")
        shutil.copyfile(ROOT / "guest/init.py", rootfs / "sbin/mtcp-init")
        (rootfs / "sbin/mtcp-init").chmod(0o755)
        with open(target / "rootfs.ext4", "wb") as image:
            image.truncate(160 * 1024**2)
        subprocess.run(["mke2fs", "-q", "-t", "ext4", "-F", "-d", str(rootfs), str(target / "rootfs.ext4")], check=True)
    manifest = dict(version=VERSION, kernel_url=KERNEL_URL, image=IMAGE,
                    artifact_digest=sha(ROOT / "mtcp/kernel.py"), guest_digest=sha(ROOT / "guest/init.py"),
                    files={name: sha(target / name) for name in ("firecracker", "vmlinux", "rootfs.ext4")})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Artifacts ready at {target}")


if __name__ == "__main__":
    main()
