#!/usr/bin/env python3
"""Create a source-only peer update; excludes data, credentials and run state."""
from pathlib import Path
import zipfile


def main():
    root=Path(__file__).resolve().parents[2]
    target=root/'results/cifar-allocation-update.zip';target.parent.mkdir(parents=True,exist_ok=True)
    files={root/'.gitattributes',root/'README.md',root/'requirements-cifar.txt',root/'docs/cifar-peers.md'}
    for pattern in ('mtcp/**/*.py','guest/*.py','tools/cifar/*.py','tools/firecracker/*.py','web/*'):
        files.update(p for p in root.glob(pattern) if p.is_file())
    instructions='''Compute-aware peer update

1. Stop the existing peer with Ctrl+C.
2. Extract this archive into the existing project folder, replacing source files.
   It contains no dataset, keys, virtual environment or previous training state.
   PowerShell: Expand-Archive .\\cifar-allocation-update.zip -DestinationPath . -Force
3. Restart the peer using the same shared MTCP_CLUSTER_KEY and prior CPU settings.
   Windows example (from the project folder):
   .\\.venv-cifar\\Scripts\\python.exe -m mtcp.cifar peer --name shridhar --port 9101 --threads 1 --backend process
4. Update and restart both laptops before starting a compute-aware run.
5. From either laptop, in a terminal with the shared cluster key:
   python -m mtcp.cifar run --peers 2 --epochs 2 --train-limit 10000 --test-limit 1000 --audit full --allocation compute-aware
   Use that laptop's virtual-environment Python path in place of python.

See docs/cifar-peers.md for paired performance validation and audit limitations.
Firecracker users must rebuild the CIFAR image before restarting that backend.
'''
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,path.relative_to(root))
        archive.writestr('ALLOCATION-UPDATE.txt',instructions)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise RuntimeError('Update archive failed integrity check')
        names=archive.namelist()
        assert all(not n.startswith(('.mtcp/','.venv','.git/')) for n in names)
        assert 'mtcp/cifar/allocation.py' in names and 'mtcp/cifar/execution.py' in names
    print(f'{target} ({target.stat().st_size:,} bytes; {len(names)} files)')


if __name__=='__main__':main()
