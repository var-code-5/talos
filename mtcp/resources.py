"""Self-reported capacity and a repeatable fixed-kernel throughput measurement."""
import hashlib
import math
import os
from pathlib import Path
import platform
import time
from .kernel import gradient, samples


def advertise(cpus=None, memory_mb=128, upload_mbps=100, download_mbps=100, backend='bounded-process'):
    cpus = cpus or sorted(os.sched_getaffinity(0))[:1]
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    data = samples()
    before = time.perf_counter()
    for _ in range(20):
        gradient([0., 0., 0.], data)
    throughput = 20000 / (time.perf_counter() - before)
    machine = Path('/etc/machine-id').read_text().strip() if Path('/etc/machine-id').exists() else platform.node()
    return dict(host_id=hashlib.sha256(machine.encode()).hexdigest()[:12], architecture=platform.machine(),
        logical_cpus=os.cpu_count(), host_memory_mb=int(meminfo['MemTotal'].split()[0]) // 1024,
        host_available_memory_mb=int(meminfo['MemAvailable'].split()[0]) // 1024,
        offered_vcpus=len(cpus), cpu_affinity=cpus, offered_memory_mb=memory_mb,
        offered_upload_mbps=upload_mbps, offered_download_mbps=download_mbps,
        bandwidth_source='owner-configured application pacing budget',
        gradient_samples_per_second=round(throughput), benchmark='host fixed-kernel gradient; 20 x 1,000 samples',
        isolation=backend, accepting=True, slots=1, measured_at=time.time())


def validate_resources(value):
    if not isinstance(value, dict):
        raise ValueError('Expected resource advertisement')
    for key, maximum in (('offered_vcpus', 64), ('offered_memory_mb', 1048576),
                         ('offered_upload_mbps', 1000000), ('offered_download_mbps', 1000000),
                         ('gradient_samples_per_second', 1e12)):
        number = value.get(key)
        if type(number) not in (float, int) or not math.isfinite(number) or not 0 < number <= maximum:
            raise ValueError(f'Invalid resource field: {key}')
    if type(value.get('accepting')) is not bool:
        raise ValueError('Invalid accepting flag')
    if type(value['offered_vcpus']) is not int or type(value['offered_memory_mb']) is not int:
        raise ValueError('CPU count and memory budget must be integers')
    return value
