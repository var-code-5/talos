#!/usr/bin/env python3
"""Build a CPU PyTorch guest with prepared CIFAR data, using pinned dependencies."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mtcp.cifar.data import sha,load_manifest
from mtcp.cifar.training import ARTIFACT
from mtcp.protocol import ARTIFACT as LINEAR_ARTIFACT

IMAGE='python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='.mtcp/cifar10');parser.add_argument('--output',default='.mtcp/firecracker-cifar')
    parser.add_argument('--base-assets',default='.mtcp/firecracker')
    parser.add_argument('--work-dir',default='.mtcp/build-tmp',help='Disk-backed build space; large rootfs archives can exceed /tmp tmpfs quota')
    args=parser.parse_args();data=Path(args.data).resolve();target=Path(args.output).resolve()
    manifest=load_manifest(data,verify=True);target.mkdir(parents=True,exist_ok=True)
    base=Path(args.base_assets).resolve()
    if not (base/'manifest.json').exists():raise RuntimeError('Run tools/firecracker/prepare.py first')
    base_manifest=json.loads((base/'manifest.json').read_text())
    for name in ('firecracker','vmlinux'):
        if sha(base/name)!=base_manifest['files'][name]:raise ValueError('Base asset checksum mismatch')
        shutil.copyfile(base/name,target/name)
    (target/'firecracker').chmod(0o755)
    work_dir=Path(args.work_dir).resolve();work_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mtcp-cifar-build-',dir=work_dir) as temporary:
        work=Path(temporary);context=work/'context';context.mkdir()
        (context/'mtcp').mkdir();(context/'mtcp/__init__.py').write_text('')
        shutil.copytree(ROOT/'mtcp/cifar',context/'mtcp/cifar',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(ROOT/'mtcp/kernel.py',context/'kernel.py')
        shutil.copyfile(ROOT/'mtcp/host.py',context/'mtcp/host.py')
        shutil.copyfile(ROOT/'guest/init.py',context/'mtcp-init')
        # Keep data out of the Docker context until dependency layers are built.
        (context/'Dockerfile').write_text(f'''FROM {IMAGE}
RUN pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy==2.5.2
COPY mtcp /opt/app/mtcp
COPY kernel.py /opt/mtcp/kernel.py
COPY mtcp-init /sbin/mtcp-init
RUN chmod 755 /sbin/mtcp-init && mkdir -p /data /proc /sys /dev /tmp
''')
        subprocess.run(['docker','build','-t','mtcp-cifar-guest:local',str(context)],check=True)
        container=subprocess.check_output(['docker','create','mtcp-cifar-guest:local'],text=True).strip()
        try:subprocess.run(['docker','export','-o',str(work/'root.tar'),container],check=True)
        finally:subprocess.run(['docker','rm',container],check=True,stdout=subprocess.DEVNULL)
        root=work/'root';root.mkdir()
        import os
        def safe(member,destination):
            if member.issym() and member.linkname.startswith('/'):
                member.linkname=os.path.relpath(member.linkname.lstrip('/'),str(Path(member.name).parent))
            return tarfile.data_filter(member,destination)
        with tarfile.open(work/'root.tar') as archive:archive.extractall(root,filter=safe)
        shutil.copytree(data,root/'data',dirs_exist_ok=True)
        with open(target/'rootfs.ext4','wb') as stream:stream.truncate(1536*1024**2)
        subprocess.run(['mke2fs','-q','-t','ext4','-F','-d',str(root),str(target/'rootfs.ext4')],check=True)
        guest_digest=sha(context/'mtcp-init')
        cifar_digest=sha(context/'mtcp/cifar/training.py')
    result=dict(version=base_manifest['version'],image=IMAGE,artifact_digest=LINEAR_ARTIFACT,
        guest_digest=guest_digest,cifar_artifact_digest=cifar_digest,dataset_id=manifest['dataset_id'],
        files={name:sha(target/name) for name in ('firecracker','vmlinux','rootfs.ext4')})
    (target/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Prepared CPU CIFAR Firecracker image at {target}')


if __name__=='__main__':main()
